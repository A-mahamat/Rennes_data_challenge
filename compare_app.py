"""
Interface Web Gradio — Comparateur de Produits (Gemini Flash + Google Search)
=============================================================================
Lance avec : python compare_app.py
Clé gratuite : https://aistudio.google.com
"""

import os
import json
import gradio as gr
from product_comparator import ProductComparatorAgent, SearchResult

DEFAULT_GEMINI = os.getenv("GEMINI_API_KEY", "")


# ── Rendu HTML ────────────────────────────────────────────────────────────────

def result_to_html(result: SearchResult) -> str:
    if not result.products:
        return (
            "<div style='padding:20px;color:#e63946;font-size:1.1rem;'>"
            "❌ Aucun produit trouvé. Essayez une autre requête.</div>"
        )

    cards = ""
    for p in result.products:
        img_tag = (
            f'<img src="{p.image_url}" alt="{p.name}" '
            f'style="max-height:175px;max-width:100%;object-fit:contain;" '
            f'onerror="this.style.display=\'none\'">'
            if p.image_url
            else '<div style="font-size:3rem;text-align:center;">📦</div>'
        )
        rating_html = (
            f'<div style="color:#f4a261;font-size:.83rem;margin:4px 0;">⭐ {p.rating}</div>'
            if p.rating else ""
        )
        desc_html = (
            f'<p style="font-size:.82rem;color:#666;line-height:1.4;margin:6px 0;">'
            f'{(p.description or "")[:120]}{"…" if p.description and len(p.description) > 120 else ""}</p>'
            if p.description else ""
        )
        store_badge = (
            f'<span style="background:#e8f4fd;color:#0066cc;padding:2px 8px;'
            f'border-radius:20px;font-size:.73rem;">{p.store}</span>'
            if p.store else ""
        )

        cards += f"""
        <div style="background:white;border-radius:12px;overflow:hidden;
                    box-shadow:0 2px 10px rgba(0,0,0,.1);display:flex;
                    flex-direction:column;">
            <div style="height:185px;display:flex;align-items:center;
                        justify-content:center;background:#f8f9fa;padding:10px;">
                {img_tag}
            </div>
            <div style="padding:14px;display:flex;flex-direction:column;gap:5px;flex:1;">
                <a href="{p.url}" target="_blank"
                   style="font-weight:600;font-size:.93rem;color:#1a1a2e;
                          text-decoration:none;line-height:1.35;">{p.name}</a>
                {desc_html}
                <div style="display:flex;justify-content:space-between;
                            align-items:center;flex-wrap:wrap;gap:4px;">
                    <span style="font-size:1.35rem;font-weight:700;color:#e63946;">{p.price}</span>
                    {store_badge}
                </div>
                {rating_html}
                <a href="{p.url}" target="_blank"
                   style="margin-top:auto;display:block;text-align:center;
                          background:#0066cc;color:white;padding:9px;
                          border-radius:8px;text-decoration:none;font-size:.88rem;">
                    Voir le produit →
                </a>
            </div>
        </div>"""

    summary_block = (
        f"<div style='background:#fff3cd;border-left:4px solid #ffc107;"
        f"padding:12px 16px;border-radius:6px;margin-bottom:20px;"
        f"font-style:italic;color:#555;'>💡 {result.summary}</div>"
        if result.summary else ""
    )

    header = (
        f"<h2 style='margin:0 0 16px;font-size:1.2rem;'>"
        f"🛒 <strong>{len(result.products)}</strong> produit(s) pour : "
        f"<em>{result.query}</em></h2>"
    )

    grid = (
        f"<div style='display:grid;"
        f"grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:16px;'>"
        f"{cards}</div>"
    )

    return header + summary_block + grid


# ── Fonction principale ───────────────────────────────────────────────────────

def search_products(gemini_key: str, query: str, max_products: int, progress=gr.Progress()):
    query = query.strip()
    gemini_key = gemini_key.strip()

    if not query:
        return "<p style='color:#e63946;padding:12px;'>⚠️ Entrez une requête.</p>", ""
    if not gemini_key:
        return (
            "<p style='color:#e63946;padding:12px;'>⚠️ Clé Gemini manquante.<br>"
            "Gratuit sur <a href='https://aistudio.google.com' target='_blank'>aistudio.google.com</a></p>",
            "",
        )

    progress(0.1, desc="Initialisation de l'agent...")
    try:
        agent = ProductComparatorAgent(gemini_key=gemini_key, verbose=False)
        progress(0.3, desc="Gemini recherche sur internet...")
        result = agent.search(query, max_products=int(max_products))
        progress(1.0, desc="Terminé !")
    except Exception as e:
        return f"<p style='color:#e63946;padding:12px;'>❌ Erreur : {e}</p>", ""

    return result_to_html(result), json.dumps(result.model_dump(), ensure_ascii=False, indent=2)


# ── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="🛒 Comparateur de Produits IA") as demo:

    gr.Markdown(
        """
        # 🛒 Comparateur de Produits IA
        > L'agent **navigue lui-même sur internet** (Amazon, Fnac, Cdiscount…) et extrait prix, images et notes.
        > **100% gratuit** — une seule clé API Gemini suffit.
        """
    )

    with gr.Accordion("🔑 Clé API Gemini (gratuite, sans CB)", open=not DEFAULT_GEMINI):
        gr.Markdown(
            "1. Aller sur **[aistudio.google.com](https://aistudio.google.com)**\n"
            "2. Cliquer **Get API key** → **Create API key**\n"
            "3. Coller la clé ci-dessous"
        )
        gemini_input = gr.Textbox(
            label="Clé Gemini API",
            placeholder="AIzaSy...",
            value=DEFAULT_GEMINI,
            type="password",
        )

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="🔍 Produit recherché",
                placeholder='ex: "casque bluetooth réduction de bruit", "laptop gaming 1000€"...',
                lines=2,
            )
        with gr.Column(scale=1):
            max_products = gr.Slider(
                label="Nombre de produits", minimum=2, maximum=10, value=5, step=1
            )

    search_btn = gr.Button("🚀 Comparer les produits", variant="primary", size="lg")

    gr.Examples(
        examples=[
            ["casque audio bluetooth réduction de bruit", 5],
            ["aspirateur robot silencieux", 4],
            ["trottinette électrique adulte", 5],
            ["friteuse à air chaud", 4],
            ["écran PC 27 pouces 144Hz gaming", 5],
            ["montre connectée sport GPS", 4],
        ],
        inputs=[query_input, max_products],
        label="💡 Exemples",
    )

    with gr.Tabs():
        with gr.Tab("🃏 Résultats visuels"):
            html_output = gr.HTML()
        with gr.Tab("📋 JSON brut"):
            json_output = gr.Code(language="json", label="Données complètes")

    inputs = [gemini_input, query_input, max_products]
    outputs = [html_output, json_output]
    search_btn.click(fn=search_products, inputs=inputs, outputs=outputs)
    query_input.submit(fn=search_products, inputs=inputs, outputs=outputs)

    gr.Markdown("---\n*Propulsé par [Gemini Flash](https://aistudio.google.com) + Google Search intégré*")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
