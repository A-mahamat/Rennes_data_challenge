"""
Agent IA Comparateur de Produits
=================================
Recherche des produits sur internet à partir d'un texte
et retourne une liste structurée avec prix, liens, images, etc.
"""

import anthropic
import json
import re
import sys
from typing import Optional
from pydantic import BaseModel, Field


# ── Modèles de données ────────────────────────────────────────────────────────

class Product(BaseModel):
    name: str = Field(description="Nom complet du produit")
    price: str = Field(description="Prix avec devise (ex: '299,99 €')")
    url: str = Field(description="Lien direct vers la page produit")
    image_url: Optional[str] = Field(None, description="URL de l'image du produit")
    description: Optional[str] = Field(None, description="Description courte du produit")
    rating: Optional[str] = Field(None, description="Note clients (ex: '4.5/5 - 234 avis')")
    store: Optional[str] = Field(None, description="Nom du site marchand")

class SearchResult(BaseModel):
    query: str = Field(description="Requête de recherche originale")
    products: list[Product] = Field(description="Liste des produits trouvés")
    summary: str = Field(description="Résumé comparatif")


# ── Prompts ───────────────────────────────────────────────────────────────────

SEARCH_SYSTEM_PROMPT = """Tu es un agent expert en comparaison de produits en ligne.

Ta mission:
1. Utilise web_search pour trouver les produits correspondant à la requête sur plusieurs sites (Amazon.fr, Fnac, Cdiscount, Darty, Boulanger, etc.)
2. Utilise web_fetch pour accéder aux pages produits et extraire les détails exacts (prix, images, descriptions)
3. Compile les résultats dans un JSON structuré

IMPORTANT: À la fin de ta réponse, retourne OBLIGATOIREMENT un bloc JSON valide entouré de ```json et ``` avec cette structure:
{
  "query": "requête originale",
  "products": [
    {
      "name": "Nom complet du produit",
      "price": "Prix avec devise",
      "url": "URL directe vers la page produit",
      "image_url": "URL de l'image principale ou null",
      "description": "Description courte (1-2 phrases)",
      "rating": "Note ex: 4.3/5 (128 avis) ou null",
      "store": "amazon.fr"
    }
  ],
  "summary": "Résumé comparatif en 2-3 phrases"
}

Cherche sur plusieurs sites marchands pour offrir une vraie comparaison."""

STRUCTURE_SYSTEM_PROMPT = """Tu es un extracteur de données structurées.
Analyse le texte fourni et extrait les informations produits dans le format JSON demandé.
Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""


# ── Agent principal ───────────────────────────────────────────────────────────

class ProductComparatorAgent:
    """Agent comparateur de produits utilisant Claude + recherche web."""

    def __init__(self, verbose: bool = True):
        self.client = anthropic.Anthropic()
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def search(self, query: str, max_products: int = 5) -> SearchResult:
        """
        Recherche des produits correspondant au texte et retourne une liste structurée.

        Args:
            query: Texte décrivant le produit recherché
            max_products: Nombre maximum de produits à retourner

        Returns:
            SearchResult avec la liste des produits
        """
        self._log(f"\n🔍 Recherche: '{query}'")
        self._log("🌐 L'agent navigue sur internet...\n")

        # Phase 1: Recherche web avec Claude
        gathered_info = self._gather_product_info(query, max_products)

        # Phase 2: Extraction du JSON depuis la réponse
        result = self._extract_structured_data(gathered_info, query, max_products)

        self._log(f"\n✅ {len(result.products)} produit(s) trouvé(s)")
        return result

    def _gather_product_info(self, query: str, max_products: int) -> str:
        """Phase 1: Utilise Claude avec les outils web pour collecter les infos produits."""
        user_message = (
            f"Recherche et compare {max_products} produits correspondant à: '{query}'\n"
            f"Cherche sur Amazon.fr, Fnac, Cdiscount, Darty et autres sites français.\n"
            f"Pour chaque produit, récupère le prix exact, le lien, l'image et la note clients."
        )

        full_text = ""

        with self.client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=8000,
            system=SEARCH_SYSTEM_PROMPT,
            tools=[
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260209", "name": "web_fetch"},
            ],
            messages=[{"role": "user", "content": user_message}],
            thinking={"type": "adaptive"},
        ) as stream:
            for event in stream:
                # Afficher la progression
                if hasattr(event, "type"):
                    if event.type == "content_block_start" and hasattr(event, "content_block"):
                        block = event.content_block
                        if block.type == "server_tool_use":
                            tool = block.name
                            icon = "🔎" if tool == "web_search" else "📄"
                            self._log(f"  {icon} {tool}...")
                        elif block.type == "text":
                            self._log("  💬 Analyse des résultats...")

            final = stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                full_text = block.text
                break

        return full_text

    def _extract_structured_data(
        self, raw_text: str, query: str, max_products: int
    ) -> SearchResult:
        """Phase 2: Extrait les données structurées depuis la réponse brute."""

        # Tenter d'extraire le JSON directement depuis la réponse
        result = self._parse_json_from_text(raw_text, query)
        if result and result.products:
            return result

        # Fallback: demander à Claude de structurer les données
        self._log("  🔄 Structuration des données...")
        try:
            response = self.client.messages.parse(
                model="claude-opus-4-6",
                max_tokens=4000,
                system=STRUCTURE_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Extrait les informations produits de ce texte "
                            f"(maximum {max_products} produits) pour la requête '{query}':\n\n"
                            f"{raw_text[:6000]}"
                        ),
                    }
                ],
                output_format=SearchResult,
            )
            return response.parsed_output
        except Exception:
            return SearchResult(
                query=query,
                products=[],
                summary="Aucun produit structuré n'a pu être extrait.",
            )

    def _parse_json_from_text(self, text: str, query: str) -> Optional[SearchResult]:
        """Tente de parser le JSON depuis le texte brut de Claude."""
        # Chercher un bloc ```json ... ```
        pattern = r"```json\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                data = json.loads(match)
                if "products" in data:
                    products = [
                        Product(
                            name=p.get("name", "Produit inconnu"),
                            price=p.get("price", "Prix non disponible"),
                            url=p.get("url", ""),
                            image_url=p.get("image_url"),
                            description=p.get("description"),
                            rating=p.get("rating"),
                            store=p.get("store"),
                        )
                        for p in data.get("products", [])
                        if p.get("name") and p.get("url")
                    ]
                    return SearchResult(
                        query=data.get("query", query),
                        products=products,
                        summary=data.get("summary", ""),
                    )
            except (json.JSONDecodeError, Exception):
                continue

        # Chercher du JSON inline { ... }
        json_match = re.search(r'\{\s*"query"[\s\S]*?"products"[\s\S]*?\}(?=\s*$)', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "products" in data:
                    products = [
                        Product(**{k: v for k, v in p.items() if k in Product.model_fields})
                        for p in data.get("products", [])
                    ]
                    return SearchResult(
                        query=data.get("query", query),
                        products=products,
                        summary=data.get("summary", ""),
                    )
            except Exception:
                pass

        return None


# ── Affichage CLI ─────────────────────────────────────────────────────────────

def display_results(result: SearchResult):
    """Affiche les résultats de manière formatée dans le terminal."""
    print("\n" + "═" * 60)
    print(f"  📦 RÉSULTATS : {result.query}")
    print("═" * 60)

    if not result.products:
        print("  ❌ Aucun produit trouvé.")
        return

    for i, product in enumerate(result.products, 1):
        print(f"\n  [{i}] {product.name}")
        print(f"      💰 Prix    : {product.price}")
        print(f"      🏪 Site    : {product.store or 'N/A'}")
        if product.rating:
            print(f"      ⭐ Note    : {product.rating}")
        if product.description:
            desc = product.description[:100] + "..." if len(product.description) > 100 else product.description
            print(f"      📝 Desc.   : {desc}")
        print(f"      🔗 Lien    : {product.url}")
        if product.image_url:
            print(f"      🖼️  Image   : {product.image_url}")

    if result.summary:
        print("\n" + "─" * 60)
        print(f"  💡 RÉSUMÉ : {result.summary}")

    print("\n" + "═" * 60)


def export_to_json(result: SearchResult, filepath: str):
    """Exporte les résultats en JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"💾 Résultats exportés : {filepath}")


def export_to_html(result: SearchResult, filepath: str):
    """Exporte les résultats en HTML avec images."""
    cards = ""
    for p in result.products:
        img_tag = (
            f'<img src="{p.image_url}" alt="{p.name}" onerror="this.style.display=\'none\'">'
            if p.image_url
            else '<div class="no-img">📦</div>'
        )
        rating_html = f'<div class="rating">⭐ {p.rating}</div>' if p.rating else ""
        desc_html = f'<p class="desc">{p.description}</p>' if p.description else ""
        store_html = f'<span class="store">{p.store}</span>' if p.store else ""

        cards += f"""
        <div class="card">
            <div class="img-wrap">{img_tag}</div>
            <div class="info">
                <h3><a href="{p.url}" target="_blank">{p.name}</a></h3>
                {desc_html}
                <div class="meta">
                    <span class="price">{p.price}</span>
                    {store_html}
                </div>
                {rating_html}
                <a href="{p.url}" target="_blank" class="btn">Voir le produit →</a>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Comparateur – {result.query}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f2f5; color: #333; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 8px; font-size: 1.8rem; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 24px; font-size: 0.95rem; }}
  .summary {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px;
              border-radius: 6px; margin-bottom: 24px; font-style: italic; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
           gap: 20px; }}
  .card {{ background: white; border-radius: 12px; overflow: hidden;
           box-shadow: 0 2px 8px rgba(0,0,0,.1); transition: transform .2s; }}
  .card:hover {{ transform: translateY(-4px); box-shadow: 0 6px 20px rgba(0,0,0,.15); }}
  .img-wrap {{ height: 200px; display: flex; align-items: center; justify-content: center;
               background: #f8f9fa; overflow: hidden; }}
  .img-wrap img {{ max-height: 100%; max-width: 100%; object-fit: contain; }}
  .no-img {{ font-size: 4rem; }}
  .info {{ padding: 16px; }}
  .info h3 {{ font-size: 1rem; margin-bottom: 8px; line-height: 1.4; }}
  .info h3 a {{ text-decoration: none; color: #1a1a2e; }}
  .info h3 a:hover {{ color: #0066cc; }}
  .desc {{ font-size: 0.85rem; color: #666; margin-bottom: 10px; line-height: 1.5; }}
  .meta {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
  .price {{ font-size: 1.3rem; font-weight: bold; color: #e63946; }}
  .store {{ font-size: 0.8rem; background: #e8f4fd; color: #0066cc;
            padding: 2px 8px; border-radius: 20px; }}
  .rating {{ font-size: 0.85rem; color: #f4a261; margin-bottom: 12px; }}
  .btn {{ display: block; text-align: center; background: #0066cc; color: white;
          padding: 10px; border-radius: 8px; text-decoration: none; font-size: 0.9rem;
          transition: background .2s; }}
  .btn:hover {{ background: #0052a3; }}
  footer {{ text-align: center; margin-top: 30px; color: #999; font-size: 0.8rem; }}
</style>
</head>
<body>
  <h1>🛒 Comparateur de produits</h1>
  <p class="subtitle">Résultats pour : <strong>{result.query}</strong> — {len(result.products)} produit(s) trouvé(s)</p>
  {"<div class='summary'>💡 " + result.summary + "</div>" if result.summary else ""}
  <div class="grid">{cards}</div>
  <footer>Généré par l'Agent IA Comparateur de Produits • Claude Opus 4.6</footer>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 Rapport HTML exporté : {filepath}")


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python product_comparator.py \"<requête produit>\" [nombre_max]")
        print('Exemple: python product_comparator.py "casque audio bluetooth" 5')
        sys.exit(1)

    query = sys.argv[1]
    max_products = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    agent = ProductComparatorAgent(verbose=True)
    result = agent.search(query, max_products=max_products)

    display_results(result)

    # Export automatique
    safe_name = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:30]
    export_to_json(result, f"results_{safe_name}.json")
    export_to_html(result, f"results_{safe_name}.html")
