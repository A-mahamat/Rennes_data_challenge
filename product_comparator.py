"""
Agent IA Comparateur de Produits — Gemini Flash + Google Search
===============================================================
Gemini navigue lui-même sur internet via son outil Google Search intégré.
→ UNE seule clé API gratuite : https://aistudio.google.com

Pas de SerpAPI, pas d'Anthropic. 100% gratuit.
"""

import os
import re
import sys
import json
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ── Modèles de données ────────────────────────────────────────────────────────

class Product(BaseModel):
    name: str = Field(description="Nom complet du produit")
    price: str = Field(description="Prix affiché (ex: '299,99 €')")
    url: str = Field(description="Lien direct vers la page produit")
    image_url: Optional[str] = Field(None, description="URL de l'image")
    description: Optional[str] = Field(None, description="Description courte")
    rating: Optional[str] = Field(None, description="Note clients")
    store: Optional[str] = Field(None, description="Nom du site marchand")


class SearchResult(BaseModel):
    query: str
    products: list[Product]
    summary: str


# ── Prompt ────────────────────────────────────────────────────────────────────

SEARCH_PROMPT = """Tu es un agent expert en comparaison de produits.

Utilise Google Search pour trouver {n} produits correspondant à : "{query}"

Instructions :
- Cherche sur plusieurs sites : Amazon.fr, Fnac, Cdiscount, Darty, Boulanger, Rakuten, etc.
- Pour chaque produit, récupère le prix actuel, le lien direct, l'image et la note clients
- Varie les sources pour avoir une vraie comparaison

Retourne UNIQUEMENT un bloc JSON valide (sans texte avant ni après) avec cette structure :
{{
  "query": "{query}",
  "products": [
    {{
      "name": "Nom complet du produit",
      "price": "Prix avec devise ex: 299,99 €",
      "url": "https://lien-direct-vers-produit.com",
      "image_url": "https://url-image.jpg ou null",
      "description": "Description courte en 1 phrase",
      "rating": "4.5/5 (234 avis) ou null",
      "store": "amazon.fr"
    }}
  ],
  "summary": "Résumé comparatif en 2-3 phrases : fourchette de prix, meilleur rapport qualité/prix, conseil"
}}"""


# ── Agent principal ───────────────────────────────────────────────────────────

class ProductComparatorAgent:
    """
    Agent comparateur utilisant Gemini Flash avec Google Search intégré.
    Gemini navigue lui-même sur internet — aucune autre API requise.
    """

    def __init__(self, gemini_key: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "Clé Gemini manquante.\n"
                "→ Gratuit sur https://aistudio.google.com\n"
                "→ Définissez GEMINI_API_KEY ou passez-la au constructeur."
            )
        self.client = genai.Client(api_key=key)

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def search(self, query: str, max_products: int = 5) -> SearchResult:
        """
        Recherche des produits sur internet et retourne une liste structurée.

        Args:
            query:        Description du produit (ex: "casque bluetooth réduction bruit")
            max_products: Nombre de produits à retourner

        Returns:
            SearchResult avec produits + résumé comparatif
        """
        self._log(f"\n🔍 Recherche : '{query}'")
        self._log("🌐 Gemini navigue sur internet...\n")

        prompt = SEARCH_PROMPT.format(query=query, n=max_products)

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                ),
            )
            raw_text = response.text or ""
        except Exception as e:
            self._log(f"  ❌ Erreur Gemini : {e}")
            return SearchResult(query=query, products=[], summary=str(e))

        self._log("  📦 Extraction des produits...")
        result = self._parse_response(raw_text, query)
        self._log(f"✅ {len(result.products)} produit(s) trouvé(s)\n")
        return result

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(self, text: str, query: str) -> SearchResult:
        """Extrait le JSON de la réponse de Gemini."""

        # Chercher un bloc ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        json_str = match.group(1) if match else text

        # Chercher le JSON brut si pas de bloc markdown
        if not match:
            obj_match = re.search(r"\{[\s\S]*\"products\"[\s\S]*\}", text)
            if obj_match:
                json_str = obj_match.group()

        try:
            data = json.loads(json_str)
            products = [
                Product(
                    name=p.get("name", "Produit inconnu"),
                    price=p.get("price", "Prix non disponible"),
                    url=p.get("url", ""),
                    image_url=p.get("image_url") or None,
                    description=p.get("description") or None,
                    rating=p.get("rating") or None,
                    store=p.get("store") or None,
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
            # Fallback : demander à Gemini de structurer sans search
            return self._fallback_structure(text, query)

    def _fallback_structure(self, raw: str, query: str) -> SearchResult:
        """Si le JSON est malformé, Gemini re-structure sans recherche web."""
        self._log("  🔄 Re-structuration des données...")
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=(
                    f"Extrait les produits de ce texte en JSON strict "
                    f"(query, products[], summary) pour '{query}':\n\n{raw[:4000]}"
                ),
                config=types.GenerateContentConfig(temperature=0),
            )
            return self._parse_response(resp.text or "", query)
        except Exception:
            return SearchResult(query=query, products=[], summary="Aucun résultat structuré.")


# ── Affichage CLI ─────────────────────────────────────────────────────────────

def display_results(result: SearchResult):
    print("\n" + "═" * 65)
    print(f"  📦 RÉSULTATS : {result.query}")
    print("═" * 65)

    if not result.products:
        print("  ❌ Aucun produit trouvé.")
        return

    for i, p in enumerate(result.products, 1):
        print(f"\n  [{i}] {p.name}")
        print(f"      💰 Prix   : {p.price}")
        print(f"      🏪 Site   : {p.store or 'N/A'}")
        if p.rating:
            print(f"      ⭐ Note   : {p.rating}")
        if p.description:
            desc = p.description[:110] + ("…" if len(p.description) > 110 else "")
            print(f"      📝 Desc.  : {desc}")
        print(f"      🔗 Lien   : {p.url}")
        if p.image_url:
            print(f"      🖼️  Image  : {p.image_url}")

    if result.summary:
        print("\n" + "─" * 65)
        print(f"  💡 RÉSUMÉ IA : {result.summary}")
    print("\n" + "═" * 65)


def export_json(result: SearchResult, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    print(f"💾 JSON exporté : {path}")


def export_html(result: SearchResult, path: str):
    cards = ""
    for p in result.products:
        img = (
            f'<img src="{p.image_url}" alt="" onerror="this.style.display=\'none\'">'
            if p.image_url else '<div class="no-img">📦</div>'
        )
        rating_html = f'<div class="rating">⭐ {p.rating}</div>' if p.rating else ""
        desc_html = f'<p class="desc">{p.description}</p>' if p.description else ""
        store_html = f'<span class="store">{p.store}</span>' if p.store else ""

        cards += f"""
        <div class="card">
          <div class="img-wrap">{img}</div>
          <div class="body">
            <h3><a href="{p.url}" target="_blank">{p.name}</a></h3>
            {desc_html}
            <div class="meta"><span class="price">{p.price}</span>{store_html}</div>
            {rating_html}
            <a class="btn" href="{p.url}" target="_blank">Voir le produit →</a>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Comparateur – {result.query}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f0f2f5;color:#333;padding:24px}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:6px}}
.sub{{text-align:center;color:#666;margin-bottom:20px}}
.summary{{background:#fff3cd;border-left:4px solid #ffc107;padding:12px 16px;border-radius:6px;margin-bottom:24px;font-style:italic}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:18px}}
.card{{background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.1);display:flex;flex-direction:column;transition:transform .2s}}
.card:hover{{transform:translateY(-4px)}}
.img-wrap{{height:190px;display:flex;align-items:center;justify-content:center;background:#f8f9fa;padding:10px}}
.img-wrap img{{max-height:100%;max-width:100%;object-fit:contain}}
.no-img{{font-size:3.5rem}}
.body{{padding:14px;display:flex;flex-direction:column;gap:6px;flex:1}}
.body h3{{font-size:.95rem;line-height:1.4}}
.body h3 a{{text-decoration:none;color:#1a1a2e}}
.desc{{font-size:.82rem;color:#666;line-height:1.5}}
.meta{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px}}
.price{{font-size:1.35rem;font-weight:700;color:#e63946}}
.store{{font-size:.75rem;background:#e8f4fd;color:#0066cc;padding:2px 8px;border-radius:20px}}
.rating{{font-size:.83rem;color:#f4a261}}
.btn{{margin-top:auto;display:block;text-align:center;background:#0066cc;color:#fff;padding:10px;border-radius:8px;text-decoration:none;font-size:.88rem}}
.btn:hover{{background:#0052a3}}
footer{{text-align:center;margin-top:28px;color:#999;font-size:.78rem}}
</style></head><body>
<h1>🛒 Comparateur de produits</h1>
<p class="sub">Résultats pour : <strong>{result.query}</strong> — {len(result.products)} produit(s)</p>
{"<div class='summary'>💡 " + result.summary + "</div>" if result.summary else ""}
<div class="grid">{cards}</div>
<footer>Propulsé par Gemini Flash + Google Search · Agent Comparateur IA</footer>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML exporté : {path}")


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage : python product_comparator.py "<produit>" [nombre]')
        print('Exemple: python product_comparator.py "casque bluetooth" 5')
        sys.exit(1)

    query = sys.argv[1]
    max_products = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    agent = ProductComparatorAgent()
    result = agent.search(query, max_products)
    display_results(result)

    safe = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:30]
    export_json(result, f"results_{safe}.json")
    export_html(result, f"results_{safe}.html")
