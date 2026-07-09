# -*- coding: utf-8 -*-
"""
Composants d'interface — identité visuelle de "Tonalité".
--------------------------------------------------------------
Centralise :
    - inject_design_system() : CSS injecté une fois en tête de page
      (typographie, couleurs, restylage des composants Streamlit).
    - sentiment_gauge() : composant signature -- jauge tri-zone
      (négatif | neutre | positif) avec marqueur positionné selon le
      résultat, réutilisée partout où un sentiment est affiché.
    - result_card() : carte de résultat cohérente avec l'identité visuelle.

Toutes les fonctions retournent du HTML à passer à st.markdown(...,
unsafe_allow_html=True), ou l'injectent directement pour le CSS global.
"""

import html

import streamlit as st

# --- Tokens de design (source unique de vérité pour les couleurs) ---
COLORS = {
    "bg": "#FBFBFA",
    "surface": "#FFFFFF",
    "border": "#E4E7EC",
    "ink": "#14181F",
    "ink_muted": "#5B6472",
    "brand": "#3452EB",
    "brand_soft": "#EEF0FE",
    "positif": "#12805C",
    "positif_soft": "#E6F4EE",
    "neutre": "#B7791F",
    "neutre_soft": "#FBF3E4",
    "négatif": "#C23B3B",
    "négatif_soft": "#FBEAEA",
}

_ZONE_ORDER = ["négatif", "neutre", "positif"]
_ZONE_POSITION = {"négatif": 0.14, "neutre": 0.5, "positif": 0.86}


def inject_design_system() -> None:
    """Injecte le CSS global. À appeler UNE FOIS en tête de app.py, juste
    après st.set_page_config()."""
    c = COLORS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --bg: {c['bg']}; --surface: {c['surface']}; --border: {c['border']};
            --ink: {c['ink']}; --ink-muted: {c['ink_muted']}; --brand: {c['brand']};
        }}

        html, body, [class*="css"] {{
            font-family: 'IBM Plex Sans', sans-serif;
            color: {c['ink']};
        }}

        .stApp {{ background: {c['bg']}; }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
            color: {c['ink']} !important;
        }}

        /* Nombres / scores en mono pour l'esprit "instrument de mesure" */
        .tn-mono {{ font-family: 'IBM Plex Mono', monospace; }}

        /* -- En-tête de marque -- */
        .tn-header {{
            display: flex; align-items: baseline; justify-content: space-between;
            padding-bottom: 0.75rem; margin-bottom: 1.25rem;
            border-bottom: 1px solid {c['border']};
        }}
        .tn-wordmark {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700;
            font-size: 1.5rem; color: {c['ink']}; letter-spacing: -0.02em;
        }}
        .tn-wordmark span {{ color: {c['brand']}; }}
        .tn-tagline {{ color: {c['ink_muted']}; font-size: 0.85rem; margin-top: 0.1rem; }}
        .tn-status {{ display: flex; gap: 0.5rem; }}
        .tn-pill {{
            font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace;
            padding: 0.22rem 0.6rem; border-radius: 999px; border: 1px solid {c['border']};
            color: {c['ink_muted']}; background: {c['surface']};
            white-space: nowrap;
        }}
        .tn-pill.on {{ color: {c['positif']}; border-color: {c['positif_soft']}; background: {c['positif_soft']}; }}
        .tn-pill.off {{ color: {c['ink_muted']}; }}

        /* -- Cartes -- */
        .tn-card {{
            background: {c['surface']}; border: 1px solid {c['border']};
            border-radius: 10px; padding: 1.1rem 1.2rem; margin-bottom: 0.9rem;
        }}
        .tn-card-title {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 600;
            font-size: 0.95rem; margin-bottom: 0.65rem; color: {c['ink']};
            display: flex; align-items: center; gap: 0.4rem;
        }}
        .tn-empty {{
            color: {c['ink_muted']}; font-size: 0.88rem; font-style: normal;
            padding: 0.4rem 0;
        }}

        /* -- Jauge tri-zone (composant signature) -- */
        .tn-gauge-wrap {{ margin: 0.3rem 0 0.5rem 0; }}
        .tn-gauge-track {{
            position: relative; height: 10px; border-radius: 999px;
            background: linear-gradient(to right,
                {c['négatif_soft']} 0%, {c['négatif_soft']} 30%,
                {c['neutre_soft']} 30%, {c['neutre_soft']} 70%,
                {c['positif_soft']} 70%, {c['positif_soft']} 100%);
            border: 1px solid {c['border']};
            overflow: visible;
        }}
        .tn-gauge-marker {{
            position: absolute; top: 50%; width: 16px; height: 16px;
            border-radius: 50%; border: 2.5px solid {c['surface']};
            box-shadow: 0 1px 3px rgba(20,24,31,0.25);
            transform: translate(-50%, -50%);
            transition: left 0.35s ease;
        }}
        .tn-gauge-labels {{
            display: flex; justify-content: space-between;
            font-size: 0.68rem; color: {c['ink_muted']}; margin-top: 0.3rem;
            font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.02em;
        }}

        /* -- Badge de label de sentiment -- */
        .tn-badge {{
            display: inline-flex; align-items: center; gap: 0.35rem;
            font-family: 'Space Grotesk', sans-serif; font-weight: 600;
            font-size: 0.95rem; padding: 0.15rem 0.7rem; border-radius: 7px;
        }}
        .tn-badge .dot {{ width: 8px; height: 8px; border-radius: 50%; }}

        /* -- Boutons Streamlit -- */
        .stButton > button {{
            font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
            border-radius: 8px !important; border: 1px solid {c['brand']} !important;
        }}
        .stButton > button[kind="primary"] {{
            background: {c['brand']} !important; border-color: {c['brand']} !important;
        }}

        /* -- Tabs -- */
        button[data-baseweb="tab"] {{
            font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
            font-size: 0.92rem !important;
        }}

        /* -- Réduction de la marque Streamlit pour un rendu plus "produit" -- */
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}

        /* -- Focus clavier visible (accessibilité) -- */
        *:focus-visible {{ outline: 2px solid {c['brand']} !important; outline-offset: 2px; }}

        /* -- Respect de prefers-reduced-motion -- */
        @media (prefers-reduced-motion: reduce) {{
            .tn-gauge-marker {{ transition: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(active_model: str, spacy_available: bool, neural_available: bool) -> None:
    spacy_pill = f'<span class="tn-pill {"on" if spacy_available else "off"}">spaCy {"actif" if spacy_available else "fallback"}</span>'
    neural_pill = f'<span class="tn-pill {"on" if neural_available else "off"}">neuronal {"actif" if neural_available else "inactif"}</span>'
    model_pill = f'<span class="tn-pill">stat. : {html.escape(active_model)}</span>'

    st.markdown(
        f"""
        <div class="tn-header">
            <div>
                <div class="tn-wordmark">Tonal<span>ité</span></div>
                <div class="tn-tagline">Analyse de sentiment comparative — corpus francophones</div>
            </div>
            <div class="tn-status">{model_pill}{spacy_pill}{neural_pill}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _gauge_html(zone: str, intensity: float) -> str:
    """zone : 'négatif' | 'neutre' | 'positif'. intensity : 0 à 1
    (contrôle la taille/l'opacité du marqueur -> code visuel de la
    confiance ou de l'intensité du score)."""
    position = _ZONE_POSITION.get(zone, 0.5)
    color = COLORS.get(zone, COLORS["ink_muted"])
    size = 12 + round(8 * max(0.0, min(intensity, 1.0)))
    opacity = 0.55 + 0.45 * max(0.0, min(intensity, 1.0))

    return f"""
    <div class="tn-gauge-wrap">
        <div class="tn-gauge-track">
            <div class="tn-gauge-marker" style="left:{position*100:.1f}%; width:{size}px; height:{size}px;
                 background:{color}; opacity:{opacity:.2f};"></div>
        </div>
        <div class="tn-gauge-labels"><span>négatif</span><span>neutre</span><span>positif</span></div>
    </div>
    """


def sentiment_badge_html(label: str) -> str:
    color = COLORS.get(label, COLORS["ink_muted"])
    return (
        f'<span class="tn-badge" style="background:{color}18; color:{color};">'
        f'<span class="dot" style="background:{color};"></span>{html.escape(label)}</span>'
    )


def result_card(title: str, label: str | None, intensity: float, detail_html: str = "", icon: str = "") -> None:
    """Carte de résultat standardisée : titre, badge de sentiment, jauge
    tri-zone, puis contenu détaillé libre (mots contributeurs, etc.)."""
    title_html = f"{icon} {html.escape(title)}" if icon else html.escape(title)

    if label is None:
        st.markdown(
            f"""
            <div class="tn-card">
                <div class="tn-card-title">{title_html}</div>
                <div class="tn-empty">{detail_html or "Résultat non disponible."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    gauge = _gauge_html(label, intensity)
    badge = sentiment_badge_html(label)
    st.markdown(
        f"""
        <div class="tn-card">
            <div class="tn-card-title">{title_html}</div>
            {badge}
            {gauge}
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def contribution_chip_html(word: str, polarity: float, negated: bool) -> str:
    zone = "négatif" if polarity < 0 else ("positif" if polarity > 0 else "neutre")
    if negated:
        zone = "négatif" if zone == "positif" else "positif"
    color = COLORS.get(zone, COLORS["ink_muted"])
    sign = "+" if polarity > 0 else ""
    negated_marker = " ↩" if negated else ""
    return (
        f'<span class="tn-mono" style="display:inline-block; margin:0.15rem 0.25rem 0 0; '
        f'padding:0.1rem 0.45rem; border-radius:6px; font-size:0.78rem; '
        f'background:{color}14; color:{color};">{html.escape(word)} {sign}{polarity:.1f}{negated_marker}</span>'
    )
