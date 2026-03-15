"""
Agent IA Comparateur de Produits — SerpAPI + Gemini Flash
==========================================================
• SerpAPI  (gratuit : 100 recherches/mois) → Google Shopping
• Gemini Flash (gratuit illimité*)          → Analyse & résumé IA

Clés API gratuites :
  SerpAPI  → https://serpapi.com          (100 req/mois sans CB)
  Gemini   → https://aistudio.google.com  (gratuit, pas de CB)
"""

import os
import re
import sys
import json
import requests
from typing import Optional

import google.generativeai as genai
from pydantic import BaseModel, Field


# ── Modèles de données ────────────────────────────────────────────────────────

class Product(BaseModel):
    name: str = Field(description="Nom complet du produit")
    price: str = Field(description="Prix affiché (ex: '299,99 €')")
    url: str = Field(description="Lien direct vers la page produit")
    image_url: Optional[str] = Field(None, description="URL de l'image miniature")
    description: Optional[str] = Field(None, description="Description courte")
    rating: Optional[str] = Field(None, description="Note clients formatée")
    store: Optional[str] = Field(None, description="Nom du site marchand")
    extracted_price: Optional[float] = Field(None, description="Prix numérique pour tri")


class SearchResult(BaseModel):
    query: str
    products: list[Product]
    summary: str


# ── Agent principal ───────────────────────────────────────────────────────────

class ProductComparatorAgent:
    """
    Comparateur de produits utilisant :
      - SerpAPI Google Shopping pour récupérer les produits (prix, images, liens)
      - Gemini Flash pour générer un résumé comparatif IA
    """

    SERPAPI_URL = "https://serpapi.com/search"

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        verbose: bool = True,
    ):
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY", "")
        gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")
        self.verbose = verbose

        if not self.serpapi_key:
            raise ValueError(
                "Clé SerpAPI manquante.\n"
                "→ Obtenez-en une gratuitement sur https://serpapi.com\n"
                "→ Définissez SERPAPI_KEY ou passez-la au constructeur."
            )
        if not gemini_key:
            raise ValueError(
                "Clé Gemini manquante.\n"
                "→ Obtenez-en une gratuitement sur https://aistudio.google.com\n"
                "→ Définissez GEMINI_API_KEY ou passez-la au constructeur."
            )

        genai.configure(api_key=gemini_key)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")

    def _log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(self, query: str, max_products: int = 5) -> SearchResult:
        """
        Recherche des produits et retourne une liste structurée.

        Args:
            query:        Texte décrivant le produit (ex: "casque bluetooth réduction bruit")
            max_products: Nombre max de produits à retourner (2–10)

        Returns:
            SearchResult contenant la liste des produits + résumé IA
        """
        self._log(f"\n🔍 Recherche : '{query}'")

        products = self._fetch_from_serpapi(query, max_products)
        summary = self._generate_summary(query, products)

        self._log(f"✅ {len(products)} produit(s) trouvé(s)\n")
        return SearchResult(query=query, products=products, summary=summary)

    # ── SerpAPI ───────────────────────────────────────────────────────────────

    def _fetch_from_serpapi(self, query: str, max_products: int) -> list[Product]:
        """Appelle l'API Google Shopping de SerpAPI et retourne les produits."""
        self._log("  🛒 Google Shopping via SerpAPI...")

        params = {
            "engine": "google_shopping",
            "q": query,
            "api_key": self.serpapi_key,
            "hl": "fr",      # langue : français
            "gl": "fr",      # pays   : France
            "num": max(max_products + 5, 10),  # marge pour filtrer les doublons
        }

        try:
            resp = requests.get(self.SERPAPI_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            self._log(f"  ❌ Erreur SerpAPI : {e}")
            return []

        raw_items = data.get("shopping_results", [])
        if not raw_items:
            self._log("  ⚠️  Aucun résultat Google Shopping.")
            return []

        products = []
        seen_names = set()

        for item in raw_items:
            name = item.get("title", "").strip()
            url = item.get("link", "").strip()

            # Ignorer les doublons et les entrées incomplètes
            if not name or not url or name in seen_names:
                continue
            seen_names.add(name)

            # Formater la note
            rating_str = None
            if item.get("rating"):
                rating_str = f"{item['rating']:.1f}/5"
                reviews = item.get("reviews")
                if reviews:
                    rating_str += f"  ({reviews:,} avis)"

            products.append(
                Product(
                    name=name,
                    price=item.get("price", "Prix non disponible"),
                    url=url,
                    image_url=item.get("thumbnail"),
                    description=item.get("snippet"),
                    rating=rating_str,
                    store=item.get("source"),
                    extracted_price=item.get("extracted_price"),
                )
            )

            if len(products) >= max_products:
                break

        return products

    # ── Gemini Flash ──────────────────────────────────────────────────────────

    def _generate_summary(self, query: str, products: list[Product]) -> str:
        """Génère un résumé comparatif avec Gemini Flash (gratuit)."""
        if not products:
            return "Aucun produit trouvé pour cette recherche."

        self._log("  💬 Analyse Gemini Flash...")

        lines = []
        for p in products:
            note = p.rating or "N/A"
            lines.append(f"• {p.name} | {p.price} | {p.store or '?'} | Note : {note}")

        prompt = f"""Tu es un expert en comparaison de produits. Voici les résultats pour "{query}" :

{chr(10).join(lines)}

Rédige un résumé comparatif en 2-3 phrases maximum incluant :
- la fourchette de prix
- le meilleur rapport qualité/prix si tu peux l'identifier
- un conseil d'achat concret"""

        try:
            response = self.gemini.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Résumé non disponible ({e})"


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
            if p.image_url
            else '<div class="no-img">📦</div>'
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
            <div class="meta">
              <span class="price">{p.price}</span>{store_html}
            </div>
            {rating_html}
            <a class="btn" href="{p.url}" target="_blank">Voir le produit →</a>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Comparateur – {result.query}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#f0f2f5;color:#333;padding:24px}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:6px}}
.sub{{text-align:center;color:#666;margin-bottom:20px}}
.summary{{background:#fff3cd;border-left:4px solid #ffc107;padding:12px 16px;
          border-radius:6px;margin-bottom:24px;font-style:italic}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:18px}}
.card{{background:#fff;border-radius:12px;overflow:hidden;
       box-shadow:0 2px 10px rgba(0,0,0,.1);display:flex;flex-direction:column;
       transition:transform .2s}}
.card:hover{{transform:translateY(-4px);box-shadow:0 6px 20px rgba(0,0,0,.15)}}
.img-wrap{{height:190px;display:flex;align-items:center;justify-content:center;
           background:#f8f9fa;padding:10px;overflow:hidden}}
.img-wrap img{{max-height:100%;max-width:100%;object-fit:contain}}
.no-img{{font-size:3.5rem}}
.body{{padding:14px;display:flex;flex-direction:column;gap:6px;flex:1}}
.body h3{{font-size:.95rem;line-height:1.4}}
.body h3 a{{text-decoration:none;color:#1a1a2e}}
.body h3 a:hover{{color:#0066cc}}
.desc{{font-size:.82rem;color:#666;line-height:1.5}}
.meta{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px}}
.price{{font-size:1.35rem;font-weight:700;color:#e63946}}
.store{{font-size:.75rem;background:#e8f4fd;color:#0066cc;padding:2px 8px;border-radius:20px}}
.rating{{font-size:.83rem;color:#f4a261}}
.btn{{margin-top:auto;display:block;text-align:center;background:#0066cc;color:#fff;
      padding:10px;border-radius:8px;text-decoration:none;font-size:.88rem;
      transition:background .2s}}
.btn:hover{{background:#0052a3}}
footer{{text-align:center;margin-top:28px;color:#999;font-size:.78rem}}
</style></head><body>
<h1>🛒 Comparateur de produits</h1>
<p class="sub">Résultats pour : <strong>{result.query}</strong> — {len(result.products)} produit(s)</p>
{"<div class='summary'>💡 " + result.summary + "</div>" if result.summary else ""}
<div class="grid">{cards}</div>
<footer>Propulsé par SerpAPI + Gemini Flash · Agent Comparateur IA</footer>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"🌐 HTML exporté : {path}")


# ── Point d'entrée CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python product_comparator.py \"<produit>\" [nombre]")
        print('Exemple: python product_comparator.py "casque bluetooth" 5')
        sys.exit(1)

    query = sys.argv[1]
    max_products = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    agent = ProductComparatorAgent()          # lit SERPAPI_KEY + GEMINI_API_KEY
    result = agent.search(query, max_products)

    display_results(result)

    safe = re.sub(r"[^\w\s-]", "", query).strip().replace(" ", "_")[:30]
    export_json(result, f"results_{safe}.json")
    export_html(result, f"results_{safe}.html")
