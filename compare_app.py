"""
Interface Web Gradio — Agent Comparateur de Produits
=====================================================
Lance avec: python compare_app.py
"""

import json
import re
import gradio as gr
from product_comparator import ProductComparatorAgent, SearchResult


# ── Rendu HTML pour Gradio ────────────────────────────────────────────────────

def result_to_html(result: SearchResult) -> str:
    """Convertit un SearchResult en HTML pour l'affichage Gradio."""
    if not result.products:
        return "<p style='color:#e63946; font-size:1.1rem;'>❌ Aucun produit trouvé. Essayez une autre requête.</p>"

    cards = ""
    for p in result.products:
        img_tag = (
            f'<img src="{p.image_url}" alt="{p.name}" '
            f'style="max-height:180px; max-width:100%; object-fit:contain;" '
            f'onerror="this.style.display=\'none\'">'
            if p.image_url
            else '<div style="font-size:3.5rem; text-align:center;">📦</div>'
        )
        rating_html = f'<div style="color:#f4a261; font-size:.85rem; margin:4px 0;">⭐ {p.rating}</div>' if p.rating else ""
        desc_html = (
            f'<p style="font-size:.85rem; color:#666; line-height:1.4; margin:8px 0;">'
            f'{p.description[:120]}{"..." if p.description and len(p.description) > 120 else ""}</p>'
            if p.description else ""
        )
        store_badge = (
            f'<span style="background:#e8f4fd; color:#0066cc; padding:2px 8px; '
            f'border-radius:20px; font-size:.75rem;">{p.store}</span>'
            if p.store else ""
        )

        cards += f"""
        <div style="background:white; border-radius:12px; overflow:hidden;
                    box-shadow:0 2px 12px rgba(0,0,0,.1); display:flex; flex-direction:column;">
            <div style="height:190px; display:flex; align-items:center; justify-content:center;
                        background:#f8f9fa; padding:10px;">
                {img_tag}
            </div>
            <div style="padding:14px; flex:1; display:flex; flex-direction:column; gap:6px;">
                <a href="{p.url}" target="_blank"
                   style="font-weight:600; font-size:.95rem; color:#1a1a2e;
                          text-decoration:none; line-height:1.3;">
                    {p.name}
                </a>
                {desc_html}
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:4px;">
                    <span style="font-size:1.4rem; font-weight:700; color:#e63946;">{p.price}</span>
                    {store_badge}
                </div>
                {rating_html}
                <a href="{p.url}" target="_blank"
                   style="margin-top:auto; display:block; text-align:center;
                          background:#0066cc; color:white; padding:9px; border-radius:8px;
                          text-decoration:none; font-size:.9rem;">
                    Voir le produit →
                </a>
            </div>
        </div>"""

    summary_html = ""
    if result.summary:
        summary_html = f"""
        <div style="background:#fff3cd; border-left:4px solid #ffc107; padding:12px 16px;
                    border-radius:6px; margin-bottom:20px; font-style:italic; color:#555;">
            💡 {result.summary}
        </div>"""

    header = f"""
    <div style="margin-bottom:20px;">
        <h2 style="margin:0 0 4px; font-size:1.3rem;">
            🛒 {len(result.products)} produit(s) trouvé(s) pour : <em>{result.query}</em>
        </h2>
    </div>"""

    grid = f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr));
                gap:16px;">
        {cards}
    </div>"""

    return header + summary_html + grid


def format_json_output(result: SearchResult) -> str:
    """Formate le JSON des résultats."""
    return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


# ── Logique principale ────────────────────────────────────────────────────────

agent = ProductComparatorAgent(verbose=False)


def search_products(query: str, max_products: int, progress=gr.Progress()):
    """Fonction de recherche appelée par Gradio."""
    if not query.strip():
        return (
            "<p style='color:#e63946'>⚠️ Veuillez entrer une requête de recherche.</p>",
            "",
        )

    progress(0.1, desc="Initialisation de l'agent...")

    # Capture la progression via un agent verbeux
    logs = []
    original_search = agent._gather_product_info

    def tracked_gather(q, mp):
        progress(0.3, desc="Recherche sur internet...")
        result = original_search(q, mp)
        progress(0.8, desc="Structuration des données...")
        return result

    agent._gather_product_info = tracked_gather

    try:
        result = agent.search(query.strip(), max_products=int(max_products))
    finally:
        agent._gather_product_info = original_search

    progress(1.0, desc="Terminé!")

    html_output = result_to_html(result)
    json_output = format_json_output(result)

    return html_output, json_output


# ── Interface Gradio ──────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1200px !important; }
#search-btn { background: #0066cc !important; color: white !important; }
"""

with gr.Blocks(
    title="🛒 Comparateur de Produits IA",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        # 🛒 Agent Comparateur de Produits IA
        > Entrez une description de produit — l'agent navigue sur internet et retourne une liste
        > comparée avec **prix**, **liens** et **images**.
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="🔍 Produit recherché",
                placeholder='ex: "casque audio bluetooth réduction de bruit", "laptop gaming 1000€"...',
                lines=2,
            )
        with gr.Column(scale=1):
            max_products = gr.Slider(
                label="Nombre de produits",
                minimum=2,
                maximum=10,
                value=5,
                step=1,
            )

    search_btn = gr.Button("🚀 Lancer la recherche", variant="primary", elem_id="search-btn")

    gr.Examples(
        examples=[
            ["casque audio bluetooth réduction de bruit", 5],
            ["aspirateur robot silencieux", 4],
            ["trottinette électrique adulte", 5],
            ["friteuse à air chaud", 4],
            ["écran PC 27 pouces 144Hz", 5],
        ],
        inputs=[query_input, max_products],
    )

    with gr.Tabs():
        with gr.Tab("🃏 Résultats visuels"):
            html_output = gr.HTML(label="Produits")

        with gr.Tab("📋 JSON brut"):
            json_output = gr.Code(language="json", label="Données JSON")

    search_btn.click(
        fn=search_products,
        inputs=[query_input, max_products],
        outputs=[html_output, json_output],
    )
    query_input.submit(
        fn=search_products,
        inputs=[query_input, max_products],
        outputs=[html_output, json_output],
    )

    gr.Markdown(
        """
        ---
        *Propulsé par [Claude Opus 4.6](https://anthropic.com) avec recherche web dynamique.*
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
