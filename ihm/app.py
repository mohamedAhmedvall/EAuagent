"""
IHM Streamlit — SOMEI Plan de Renouvellement du Réseau d'Eau Potable
=====================================================================
Pages :
  1. 📊 Tableau de bord réseau
  2. 🔍 Explorer les tronçons
  3. 🎯 Scorer un tronçon
  4. ⚙️  Optimisation du plan
  5. 🔄 Analyse What-If

Lancement :
  streamlit run ihm/app.py
"""

import math
import os
import sys

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────
API_URL   = os.getenv("SOMEI_API_URL", "http://localhost:8000")
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_CSV = os.path.join(BASE_DIR, "models", "scoring_troncons.csv")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
DEVISE    = "EUR"
COUT_CURATIF_RATIO = 3.5   # casse urgente coûte en moyenne 3,5× le renouvellement préventif
_COUT_KM_EUR = {
    "FT": 8_000_000, "FTG": 7_500_000, "FTVI": 8_500_000,
    "PEHD": 6_000_000, "PVC": 5_500_000, "BTM": 7_000_000,
    "POLY": 6_500_000, "AC": 9_000_000, "AUTRE": 7_000_000,
}
ANNEE_COURANTE = 2026

st.set_page_config(
    page_title="SOMEI — Plan de Renouvellement",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
    margin: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  .metric-card-warn {
    background: linear-gradient(135deg, #922b21 0%, #e74c3c 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
    margin: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  .metric-card-ok {
    background: linear-gradient(135deg, #1a5e2a 0%, #27ae60 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
    margin: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  .metric-card-orange {
    background: linear-gradient(135deg, #784212 0%, #e67e22 100%);
    border-radius: 12px; padding: 16px; color: white; text-align: center;
    margin: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }
  .metric-value { font-size: 2rem; font-weight: bold; }
  .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
  .metric-trend-up   { color: #2ecc71; font-size: 0.8rem; margin-top: 4px; }
  .metric-trend-down { color: #e74c3c; font-size: 0.8rem; margin-top: 4px; }
  .metric-trend-neu  { color: #f1c40f; font-size: 0.8rem; margin-top: 4px; }
  .urgent-badge {
    background: #e74c3c; color: white; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem; font-weight: bold;
  }
  .ok-badge {
    background: #27ae60; color: white; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem;
  }
  .alert-n3 {
    background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
    border-radius: 10px; padding: 12px 16px; color: white;
    margin: 6px 0; font-size: 0.9rem;
  }
  h1 { color: #1e3a5f; }
  .stButton>button { background: #2980b9; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ────────────────────────────────────────────────

RHO_WEIBULL = 2.78

@st.cache_data(ttl=600)
def charger_scoring() -> pd.DataFrame:
    """Chargement direct du CSV + calcul P_casse_1an."""
    df = pd.read_csv(SCORE_CSV)
    df["age_actuel"] = 2026 - df["DDP_year"]

    # P(casse dans la prochaine année | survie jusqu'à aujourd'hui)
    def _p1an(row):
        med = row["duree_mediane_pred"]
        age = row["age_actuel"]
        if med <= 0 or age < 0:
            return 0.0
        rho = RHO_WEIBULL
        lam = med / (np.log(2) ** (1.0 / rho))
        if lam <= 0:
            return 1.0
        def S(t):
            return float(np.exp(-((max(t, 0) / lam) ** rho))) if t > 0 else 1.0
        s_now = S(age)
        s_next = S(age + 1)
        if s_now < 1e-12:
            return 1.0
        return max(0.0, min(1.0, 1.0 - s_next / s_now))

    df["P_casse_1an"] = df.apply(_p1an, axis=1)
    return df


def api_get(path: str, params=None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def couleur_decile(d):
    if d >= 9:   return "#c0392b"
    elif d >= 7: return "#e67e22"
    elif d >= 5: return "#f1c40f"
    elif d >= 3: return "#2ecc71"
    else:        return "#27ae60"


def badge_risque(p1an: float) -> str:
    """Badge basé sur P_casse_1an (probabilité de casse dans la prochaine année)."""
    if p1an >= 0.05:   return "🔴 CRITIQUE  (P≥5%/an)"
    elif p1an >= 0.01: return "🟠 ÉLEVÉ     (P≥1%/an)"
    elif p1an >= 0.001:return "🟡 MODÉRÉ    (P≥0.1%/an)"
    else:              return "🟢 FAIBLE    (P<0.1%/an)"


# ─── Sidebar ──────────────────────────────────────────────────────────────

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("## 💧 SOMEI")
    st.caption("Plan de Renouvellement — Réseau Eau Potable")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Tableau de bord",
            "🔍 Explorer les tronçons",
            "🎯 Scorer un tronçon",
            "⚙️ Optimisation du plan",
            "🔄 Analyse What-If",
            "🧠 Comparaison & Explicabilité",
            "🗺️ Carte du réseau",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"API : `{API_URL}`")
    # Vérification API
    _, err = api_get("/")
    if err:
        st.warning("⚠️ API hors ligne\n(données locales utilisées)")
    else:
        st.success("✅ API connectée")
    st.divider()
    st.caption(f"Devise : **{DEVISE}**")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tableau de bord
# ═══════════════════════════════════════════════════════════════════════════

if page == "📊 Tableau de bord":
    st.title("📊 Tableau de bord du réseau")

    df = charger_scoring()
    total_km = df["LNG"].sum() / 1000
    km_min_loi = total_km * 1.0 / 100  # 1% réglementaire

    # KPIs ligne 1
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Tronçons", f"{len(df):,}", "Total analysés"),
        ("Réseau", f"{total_km:.0f} km", "Longueur totale"),
        ("Min légal/an", f"{km_min_loi:.0f} km", "Loi : 1%/an obligatoire"),
        ("Top 10% risque", f"{df['top10_pourcent'].sum():,}", "Tronçons prioritaires"),
        ("Fuites actives", f"{(df['nb_fuites_detectees']>=1).sum():,}", "Tronçons avec fuites"),
    ]
    for col, (label, val, sub) in zip([col1, col2, col3, col4, col5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{label}<br><small>{sub}</small></div>
            </div>""", unsafe_allow_html=True)

    # KPIs P_casse_1an ligne 2
    st.divider()
    st.subheader("Probabilité de casse dans la prochaine année — P_casse_1an")
    st.caption("Hazard conditionnel Weibull AFT : P(casse dans [age, age+1] | survie jusqu'à aujourd'hui)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P_casse_1an moyen", f"{df['P_casse_1an'].mean():.4%}")
    c2.metric("P_casse_1an médian", f"{df['P_casse_1an'].median():.4%}")
    c3.metric("Tronçons P≥1%/an", f"{(df['P_casse_1an']>=0.01).sum():,}",
              help="Priorité ÉLEVÉE ou CRITIQUE")
    c4.metric("Tronçons P≥5%/an", f"{(df['P_casse_1an']>=0.05).sum():,}",
              help="Priorité CRITIQUE — renouvellement immédiat")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution des déciles de risque")
        decile_counts = df["decile_risque"].value_counts().sort_index()
        colors = [couleur_decile(d) for d in decile_counts.index]
        fig = go.Figure(go.Bar(
            x=decile_counts.index.astype(str),
            y=decile_counts.values,
            marker_color=colors,
            text=decile_counts.values,
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Décile de risque (1=faible, 10=élevé)",
            yaxis_title="Nombre de tronçons",
            height=350, showlegend=False,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Répartition par matériau")
        mat_counts = df["MAT_grp"].value_counts()
        fig2 = px.pie(
            values=mat_counts.values,
            names=mat_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Score de risque par matériau")
        risk_mat = df.groupby("MAT_grp")["risk_score_50ans"].agg(
            ["mean", "median", "count"]
        ).round(3).sort_values("mean", ascending=False)
        risk_mat.columns = ["Moy", "Médiane", "N"]
        st.dataframe(risk_mat, use_container_width=True)

    with col_d:
        st.subheader("Distribution des âges")
        fig3 = px.histogram(
            df, x="age_actuel", nbins=40,
            color_discrete_sequence=["#3498db"],
            labels={"age_actuel": "Âge actuel (ans)"},
        )
        fig3.update_layout(height=300, plot_bgcolor="white",
                           bargap=0.05, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Heatmap matériau × décile
    st.subheader("Heatmap risque : Matériau × Décile")
    mats_main = ["FT", "FTG", "FTVI", "PEHD", "PVC", "BTM", "POLY"]
    heatmap_data = {}
    for mat in mats_main:
        heatmap_data[mat] = [
            df[(df["MAT_grp"] == mat) & (df["decile_risque"] == d)].shape[0]
            for d in range(1, 11)
        ]
    hm_df = pd.DataFrame(heatmap_data, index=range(1, 11)).T
    # normaliser par ligne
    hm_pct = hm_df.div(hm_df.sum(axis=1), axis=0).round(3) * 100
    fig4 = px.imshow(
        hm_pct,
        labels={"x": "Décile de risque", "y": "Matériau", "color": "% du matériau"},
        color_continuous_scale="RdYlGn_r",
        text_auto=".0f",
        aspect="auto",
    )
    fig4.update_layout(height=320)
    st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # KPIs AVANCÉS — Taux d'urgences, Vétusté, Alerte N+3
    # ══════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("Indicateurs avancés — Urgences & Vétusté")

    n_urgences_tot = int((
        df["MAT_grp"].isin(["FTVI", "AC"]) |
        (df["nb_fuites_detectees"] >= 1) |
        (df["age_actuel"] >= 60)
    ).sum())
    taux_urg = n_urgences_tot / len(df) * 100
    n_50 = int((df["age_actuel"] >= 50).sum())
    n_70 = int((df["age_actuel"] >= 70).sum())

    # Alerte N+3 : tronçons qui passeront en P ≥ 1%/an dans 3 ans
    def _p1an_futur(row, delta=3):
        med = row["duree_mediane_pred"]
        age = row["age_actuel"] + delta
        if med <= 0 or age < 0:
            return 0.0
        rho = RHO_WEIBULL
        lam = med / (np.log(2) ** (1.0 / rho))
        if lam <= 0:
            return 1.0
        def S(t): return float(np.exp(-((max(t, 0) / lam) ** rho))) if t > 0 else 1.0
        sn, ss = S(age), S(age + 1)
        return max(0.0, min(1.0, 1.0 - ss / sn)) if sn > 1e-12 else 1.0

    with st.spinner("Calcul alerte N+3 …"):
        df_n3 = df.copy()
        df_n3["P_n3"] = df_n3.apply(_p1an_futur, axis=1)
    n_alerte_n3 = int(((df_n3["P_casse_1an"] < 0.01) & (df_n3["P_n3"] >= 0.01)).sum())

    ka1, ka2, ka3, ka4 = st.columns(4)
    ka1.metric(
        "Taux d'urgences actives",
        f"{taux_urg:.1f}%",
        delta=f"{n_urgences_tot:,} tronçons",
        delta_color="inverse",
        help="Matériaux urgents (FTVI/AC) + fuites détectées + âge > 60 ans",
    )
    ka2.metric(
        "Indice vétusté (> 50 ans)",
        f"{n_50 / len(df) * 100:.1f}%",
        delta=f"{n_50:,} tronçons",
        delta_color="inverse",
    )
    ka3.metric(
        "Vétusté critique (> 70 ans)",
        f"{n_70 / len(df) * 100:.1f}%",
        delta=f"{n_70:,} tronçons",
        delta_color="inverse",
    )
    ka4.metric(
        "🔔 Alerte prédictive N+3",
        f"{n_alerte_n3:,} tronçons",
        delta="passeront P ≥ 1%/an d'ici 2029",
        delta_color="inverse",
        help="Tronçons actuellement < 1%/an qui atteindront ce seuil critique dans 3 ans",
    )

    # ── Coût de non-remplacement ──────────────────────────────────────────
    st.divider()
    st.subheader(f"Coût de non-remplacement — Curatif vs Préventif ({DEVISE})")
    st.caption(
        f"Hypothèse : une casse urgente coûte **×{COUT_CURATIF_RATIO}** le renouvellement préventif "
        "(mobilisation d'urgence, réparations secondaires, coupures de service)."
    )

    df_urg = df[
        df["MAT_grp"].isin(["FTVI", "AC"]) |
        (df["nb_fuites_detectees"] >= 1) |
        (df["age_actuel"] >= 60)
    ].copy()
    df_urg["cout_preventif"] = df_urg.apply(
        lambda r: _COUT_KM_EUR.get(r["MAT_grp"], 7_000_000) * r["LNG"] / 1000, axis=1
    )
    total_preventif = df_urg["cout_preventif"].sum()
    total_curatif   = total_preventif * COUT_CURATIF_RATIO

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric(
        "Coût préventif total (urgences)",
        f"{total_preventif / 1e6:.0f} M {DEVISE}",
        delta=f"{len(df_urg):,} tronçons urgents",
    )
    cc2.metric(
        "Coût curatif estimé (sans plan)",
        f"{total_curatif / 1e6:.0f} M {DEVISE}",
        delta=f"×{COUT_CURATIF_RATIO} — casses non anticipées",
        delta_color="inverse",
    )
    cc3.metric(
        "Économies potentielles",
        f"{(total_curatif - total_preventif) / 1e6:.0f} M {DEVISE}",
        delta="en adoptant un plan préventif",
        delta_color="normal",
    )

    # ── Alerte N+3 — tableau détaillé ────────────────────────────────────
    st.divider()
    st.subheader("🔔 Alerte prédictive N+3 — Tronçons à surveiller")
    st.caption(
        f"**{n_alerte_n3:,} tronçons** passeront le seuil critique P ≥ 1%/an d'ici {ANNEE_COURANTE + 3}. "
        "À intégrer dans le prochain cycle de planification."
    )
    if n_alerte_n3 > 0:
        df_n3_alert = df_n3[
            (df_n3["P_casse_1an"] < 0.01) & (df_n3["P_n3"] >= 0.01)
        ].sort_values("P_n3", ascending=False)
        cols_n3 = [c for c in ["GID", "MAT_grp", "DIAMETRE_imp", "LNG",
                                "age_actuel", "P_casse_1an", "P_n3", "decile_risque"]
                   if c in df_n3_alert.columns]
        st.dataframe(
            df_n3_alert[cols_n3].head(300).rename(columns={
                "MAT_grp": "Matériau", "DIAMETRE_imp": "Ø mm", "LNG": "Long. m",
                "age_actuel": "Âge", "P_casse_1an": "P actuel",
                "P_n3": f"P en N+3 ({ANNEE_COURANTE + 3})", "decile_risque": "Décile",
            }).style.format({
                "P actuel": "{:.3%}",
                f"P en N+3 ({ANNEE_COURANTE + 3})": "{:.3%}",
            }).background_gradient(subset=[f"P en N+3 ({ANNEE_COURANTE + 3})"], cmap="Oranges"),
            use_container_width=True,
            height=350,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — Explorer les tronçons
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔍 Explorer les tronçons":
    st.title("🔍 Explorer les tronçons")

    df = charger_scoring()

    # Filtres
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dec_min = st.slider("Décile risque min", 1, 10, 7)
    with col2:
        mats = ["Tous"] + sorted(df["MAT_grp"].unique().tolist())
        mat_sel = st.selectbox("Matériau", mats)
    with col3:
        top10_only = st.checkbox("Top 10% seulement", value=False)
    with col4:
        fuites_only = st.checkbox("Fuites détectées", value=False)

    filtered = df[df["decile_risque"] >= dec_min].copy()
    if mat_sel != "Tous":
        filtered = filtered[filtered["MAT_grp"] == mat_sel]
    if top10_only:
        filtered = filtered[filtered["top10_pourcent"] == 1]
    if fuites_only:
        filtered = filtered[filtered["nb_fuites_detectees"] >= 1]

    filtered = filtered.sort_values("P_casse_1an", ascending=False)

    st.caption(f"**{len(filtered):,}** tronçons — triés par P_casse_1an (priorité annuelle)")

    # Affichage tableau avec P_casse_1an en premier
    cols_affich = ["GID", "MAT_grp", "DIAMETRE_imp", "LNG", "age_actuel",
                   "P_casse_1an", "risk_score_50ans", "decile_risque",
                   "nb_fuites_detectees", "nb_anomalies", "duree_mediane_pred"]
    cols_ok = [c for c in cols_affich if c in filtered.columns]

    df_display = filtered[cols_ok].copy()
    rename_map = {
        "GID": "GID", "MAT_grp": "Matériau", "DIAMETRE_imp": "Diamètre (mm)",
        "LNG": "Longueur (m)", "age_actuel": "Âge (ans)",
        "P_casse_1an": "P(casse/an) ★",
        "risk_score_50ans": "Score 50 ans", "decile_risque": "Décile",
        "nb_fuites_detectees": "Fuites détectées",
        "nb_anomalies": "Anomalies", "duree_mediane_pred": "Durée médiane (ans)",
    }
    df_display = df_display.rename(columns={k: v for k, v in rename_map.items() if k in df_display.columns})

    grad_cols = [c for c in ["P(casse/an) ★", "Score 50 ans"] if c in df_display.columns]
    st.dataframe(
        df_display.head(500).style.background_gradient(subset=grad_cols[:1], cmap="RdYlGn_r")
                  .format({c: "{:.4%}" for c in ["P(casse/an) ★"] if c in df_display.columns}),
        use_container_width=True,
        height=400,
    )

    # Scatter P_casse_1an vs âge
    st.subheader("P(casse prochaine année) vs Âge du tronçon")
    fig = px.scatter(
        filtered.sample(min(3000, len(filtered))),
        x="age_actuel", y="P_casse_1an",
        color="MAT_grp", size="DIAMETRE_imp",
        hover_data=["GID", "LNG", "decile_risque", "risk_score_50ans"],
        opacity=0.6,
        labels={"age_actuel": "Âge (ans)", "P_casse_1an": "P(casse dans 1 an)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(height=400, plot_bgcolor="white",
                      yaxis_tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)

    # Export
    csv = filtered[cols_ok].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Télécharger la sélection (CSV)", csv,
                       "troncons_selectionnes.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — Scorer un tronçon
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🎯 Scorer un tronçon":
    st.title("🎯 Scorer un tronçon ad hoc")
    st.caption("Évaluer le risque d'un tronçon avec ses caractéristiques via le modèle Weibull AFT.")

    with st.form("form_score"):
        col1, col2 = st.columns(2)
        with col1:
            mat = st.selectbox("Matériau", ["FT", "FTG", "FTVI", "PEHD", "PVC", "BTM", "POLY", "AC"])
            diametre = st.number_input("Diamètre (mm)", 25, 1000, 100)
            lng = st.number_input("Longueur (m)", 1.0, 5000.0, 80.0)
            annee_pose = st.number_input("Année de pose", 1900, 2025, 1980)
        with col2:
            nb_anomalies = st.number_input("Nb anomalies", 0, 100, 0)
            nb_fuites_sig = st.number_input("Fuites signalées", 0, 50, 0)
            nb_fuites_det = st.number_input("Fuites détectées", 0, 50, 0)
        submitted = st.form_submit_button("🔬 Calculer le score", use_container_width=True)

    if submitted:
        payload = {
            "MAT_grp": mat,
            "DIAMETRE_imp": diametre,
            "LNG": lng,
            "DDP_year": annee_pose,       # utilisé pour l'âge actuel uniquement
            "nb_anomalies": nb_anomalies,
            "nb_fuites_signalees": nb_fuites_sig,
            "nb_fuites_detectees": nb_fuites_det,
        }
        with st.spinner("Calcul en cours …"):
            result, err = api_post("/score", payload)

        if err:
            st.error(f"Erreur API : {err}\n\nVérifiez que l'API est démarrée : `uvicorn api.main:app --reload`")
        else:
            st.divider()
            # Métrique principale : P_casse_1an
            p1an = result.get("P_casse_1an", 0)
            col0, col_r1, col_r2, col_r3 = st.columns(4)
            col0.metric("P(casse cette année) ★",
                        f"{p1an:.3%}",
                        help="Probabilité conditionnelle de casse dans la prochaine année")
            col_r1.metric("Score risque 50 ans", f"{result['risk_score_50ans']:.3f}")
            col_r2.metric("Durée médiane prédite", f"{result['duree_mediane_pred']:.1f} ans")
            col_r3.metric("Décile de risque", f"{result['decile_risque']}/10")

            badge = badge_risque(p1an)
            st.markdown(f"### {badge}")
            st.info(result["interpretation"])

            # Courbe de survie + P_casse_1an projetée dans le temps
            col_surv, col_haz = st.columns(2)

            with col_surv:
                st.subheader("Courbe de survie S(t)")
                horizons = [10, 20, 30, 50, 70]
                probs = [result[f"P_survie_{h}ans"] for h in horizons]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=horizons, y=probs,
                    mode="lines+markers+text",
                    text=[f"{p:.0%}" for p in probs],
                    textposition="top center",
                    line=dict(color="#2980b9", width=3),
                    marker=dict(size=10),
                ))
                fig.update_layout(
                    xaxis_title="Horizon (années depuis la pose)",
                    yaxis_title="P(survie)",
                    yaxis_range=[0, 1.05], height=300, plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_haz:
                st.subheader("P(casse dans 1 an) par âge futur")
                st.caption("Le score change d'année en année si non renouvelé")
                med = result["duree_mediane_pred"]
                age_base = 2026 - annee_pose
                ages_proj = list(range(max(0, age_base - 5), age_base + 16))

                def p1an_from_med(med, age):
                    rho = RHO_WEIBULL
                    if med <= 0 or age < 0: return 0.0
                    lam = med / (np.log(2) ** (1.0 / rho))
                    S = lambda t: float(np.exp(-((max(t,0)/lam)**rho))) if t > 0 else 1.0
                    sn, ss = S(age), S(age+1)
                    return max(0.0, min(1.0, 1 - ss/sn)) if sn > 1e-12 else 1.0

                p1an_proj = [p1an_from_med(med, a) for a in ages_proj]
                colors_haz = ["#e74c3c" if a >= age_base else "#95a5a6" for a in ages_proj]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=ages_proj, y=[p*100 for p in p1an_proj],
                    marker_color=colors_haz,
                    text=[f"{p:.2%}" for p in p1an_proj],
                    textposition="outside",
                ))
                fig2.add_vline(x=age_base, line_dash="dash", line_color="#2c3e50",
                               annotation_text=f"Aujourd'hui ({age_base} ans)")
                fig2.update_layout(
                    xaxis_title="Âge du tronçon (ans)",
                    yaxis_title="P(casse dans 1 an) %",
                    height=300, plot_bgcolor="white", showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Optimisation du plan
# ═══════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Optimisation du plan":
    st.title("⚙️ Optimisation du plan de renouvellement")

    # ── Sélecteur d'horizon — mis en avant ───────────────────────────────
    st.subheader("📅 Horizon du plan")
    col_h1, col_h2, col_h3 = st.columns([2, 2, 3])

    with col_h1:
        horizon_choice = st.radio(
            "Choisir l'horizon",
            options=[1, 3, 5, 10],
            index=2,
            format_func=lambda x: f"{x} an{'s' if x > 1 else ''}",
            horizontal=True,
        )

    with col_h2:
        annee_debut = st.number_input("Année de début", 2025, 2035, 2026)

    with col_h3:
        st.markdown(f"""
        **Ce qui change selon l'horizon :**
        - **Scores (P_casse_1an, risk_score_50ans)** : calculés à aujourd'hui, identiques
        - **Bénéfice du renouvellement** : dynamique — P_casse_1an(age + t) augmente avec l'âge
          → renouveler en année 3 est évalué avec le score du tronçon en 2029
        - **Budget total** : horizon × budget annuel = enveloppe pluriannuelle
        - **Km traités** : plus d'horizon = plus de tronçons planifiés
        - **Loi 1%/an** : s'applique à **chaque** année du plan
        """)

    st.divider()

    with st.expander("💰 Contraintes financières", expanded=True):
        col1, col2 = st.columns(2)
        budget_max = col1.number_input("Budget annuel max (M EUR)", 50, 5000, 500) * 1_000_000
        budget_min = col2.number_input("Budget annuel min (M EUR)", 0, 500, 50) * 1_000_000

    with st.expander("🏗️ Contraintes opérationnelles"):
        col1, col2 = st.columns(2)
        km_max = col1.number_input("Km max renouvelables/an", 10, 500, 150)
        km_min = col2.number_input("Km min/an (taux cible opérationnel)", 0, 200, 10)
        chantiers_max = st.number_input("Chantiers simultanés max", 1, 100, 10)
        st.caption("ℹ️ La loi impose en plus un minimum de **1%/an** (~79 km/an sur 7920 km), "
                   "défini dans 'Taux réglementaire' ci-dessous.")

    with st.expander("🎯 Priorisation & réglementaire"):
        col1, col2 = st.columns(2)
        decile_prio = col1.slider("Décile prioritaire min", 1, 10, 7)
        age_max = col2.number_input("Age max sans renouvellement (ans)", 30, 120, 60)
        taux_regul = st.number_input("Taux min réglementaire (%/an)", 0.0, 10.0, 1.0, step=0.1,
                                     help="La loi impose 1%/an minimum (~79 km/an)")
        lissage = st.slider("Lissage budget (variation max %)", 0, 100, 30) / 100

    with st.expander("🎛️ Objectif & performance"):
        objectif = st.selectbox(
            "Objectif d'optimisation",
            ["maximiser_reduction_risque", "minimiser_cout", "equilibre"],
            format_func=lambda x: {
                "maximiser_reduction_risque": "Maximiser la réduction de P(casse)",
                "minimiser_cout": "Minimiser le coût total",
                "equilibre": "Équilibre coût / réduction de risque",
            }[x],
        )
        top_n = st.number_input(
            "Limiter aux N tronçons les plus risqués (0 = tous)", 0, 194000, 5000
        )

    if st.button("🚀 Lancer l'optimisation", use_container_width=True, type="primary"):
        payload = {
            "contraintes": {
                "budget_annuel_max": budget_max,
                "budget_annuel_min": budget_min,
                "km_max_par_an": km_max,
                "km_min_par_an": km_min,
                "chantiers_simultanes_max": chantiers_max,
                "seuil_decile_prioritaire": decile_prio,
                "age_max_sans_renouvellement": age_max,
                "taux_renouvellement_min_pct": taux_regul,
                "lissage_budget_pct": lissage,
                "horizon_plan": horizon_choice,
                "annee_debut": annee_debut,
            },
            "top_n_troncons": int(top_n) if top_n > 0 else None,
            "objectif": objectif,
        }

        with st.spinner("Optimisation en cours (MILP) …"):
            result, err = api_post("/optimiser", payload)

        if err:
            st.error(f"Erreur API : {err}")
        else:
            st.success(f"✅ {result['message']}")

            g = result["resume_global"]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Tronçons planifiés", f"{g.get('nb_troncons_planifies',0):,}")
            col2.metric("Km renouvelés", f"{g.get('km_total_renouveles',0):.1f} km",
                        delta=f"min légal {g.get('km_min_reglementaire_par_an',0):.0f} km/an")
            col3.metric("Budget total", f"{g.get('budget_total_engage',0)/1e6:.1f} M EUR")
            col4.metric("P(casse/an) évitée ★",
                        f"{g.get('p_casse_1an_evitee',0):.3f}",
                        help="Somme des P_casse_1an des tronçons planifiés")
            col5.metric("Risque résiduel (50 ans)", f"{g.get('risque_residuel_pct',100):.1f}%")

            # Résumé par année
            st.subheader("Plan annuel")
            annees_data = result["resume_par_annee"]
            if annees_data:
                df_annees = pd.DataFrame(annees_data)
                df_annees["budget_engage"] = df_annees["budget_engage"] / 1e6

                fig = go.Figure()
                fig.add_bar(
                    x=df_annees["annee"], y=df_annees["km_renouveles"],
                    name="Km renouvelés", marker_color="#3498db",
                    yaxis="y",
                )
                fig.add_trace(go.Scatter(
                    x=df_annees["annee"], y=df_annees["budget_engage"],
                    name="Budget (M EUR)", mode="lines+markers",
                    marker_color="#e74c3c", yaxis="y2",
                ))
                fig.update_layout(
                    yaxis=dict(title="Km renouvelés"),
                    yaxis2=dict(title="Budget (M EUR)", overlaying="y", side="right"),
                    height=350, plot_bgcolor="white", legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df_annees.rename(columns={
                        "annee": "Année", "nb_troncons": "Tronçons",
                        "km_renouveles": "Km", "budget_engage": "Budget (M EUR)",
                        "reduction_risque_totale": "Réd. risque",
                    }),
                    use_container_width=True,
                )

            # Plan détaillé
            if result["plan_detaille"]:
                st.subheader("Plan détaillé par tronçon")
                df_plan = pd.DataFrame(result["plan_detaille"])
                df_plan["cout_estime"] = df_plan["cout_estime"] / 1e6
                rename_plan = {
                    "GID": "GID", "annee_prevue": "Année",
                    "MAT_grp": "Matériau", "DIAMETRE_imp": "Diamètre",
                    "LNG_km": "Longueur (km)", "cout_estime": f"Coût (M {DEVISE})",
                    "risk_score_50ans": "Score risque",
                    "decile_risque": "Décile",
                    "raison_priorite": "Raison",
                }
                st.dataframe(
                    df_plan.rename(columns=rename_plan),
                    use_container_width=True, height=400,
                )

                # ── Exports ──────────────────────────────────────────────
                exp_csv, exp_xl = st.columns(2)
                csv_plan = df_plan.to_csv(index=False).encode("utf-8")
                exp_csv.download_button(
                    "⬇️ CSV", csv_plan,
                    "plan_renouvellement.csv", "text/csv",
                )
                try:
                    import io as _io
                    buf_xl = _io.BytesIO()
                    df_annees_xl = pd.DataFrame(annees_data).copy()
                    df_annees_xl["budget_engage"] = df_annees_xl["budget_engage"] / 1e6
                    with pd.ExcelWriter(buf_xl, engine="openpyxl") as writer:
                        df_annees_xl.rename(columns={
                            "annee": "Année", "nb_troncons": "Tronçons",
                            "km_renouveles": "Km", "budget_engage": f"Budget (M {DEVISE})",
                            "reduction_risque_totale": "Réd. risque",
                        }).to_excel(writer, sheet_name="Résumé annuel", index=False)
                        df_plan.rename(columns=rename_plan).to_excel(
                            writer, sheet_name="Plan détaillé", index=False)
                    exp_xl.download_button(
                        "📊 Excel",
                        buf_xl.getvalue(),
                        "plan_renouvellement.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception:
                    exp_xl.caption("openpyxl requis pour Excel")

            # ── ROI du renouvellement ────────────────────────────────────
            st.divider()
            st.subheader(f"ROI estimé du renouvellement ({DEVISE})")
            budget_total_plan = g.get("budget_total_engage", 0)
            casses_evitees    = g.get("p_casse_1an_evitee", 0)
            nb_troncons_plan  = max(g.get("nb_troncons_planifies", 1), 1)
            cout_moy_troncon  = budget_total_plan / nb_troncons_plan
            economies_est     = casses_evitees * cout_moy_troncon * COUT_CURATIF_RATIO * horizon_choice
            roi_est           = (economies_est - budget_total_plan) / max(budget_total_plan, 1) * 100

            roi1, roi2, roi3 = st.columns(3)
            roi1.metric(
                f"Budget préventif total",
                f"{budget_total_plan / 1e6:.1f} M {DEVISE}",
            )
            roi2.metric(
                "Économies curatif évitées",
                f"{economies_est / 1e6:.1f} M {DEVISE}",
                delta=f"×{COUT_CURATIF_RATIO} ratio urgence × {horizon_choice} ans",
            )
            roi3.metric(
                "ROI estimé",
                f"{roi_est:.0f}%",
                delta="plan rentable" if roi_est > 0 else "déficitaire",
                delta_color="normal" if roi_est > 0 else "inverse",
            )
            st.caption(
                f"Hypothèse : une casse urgente coûte **×{COUT_CURATIF_RATIO}** "
                "le renouvellement préventif."
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — Analyse What-If
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔄 Analyse What-If":
    st.title("🔄 Analyse What-If paramétrique")
    st.caption(
        "Faites varier une ou plusieurs contraintes et comparez les résultats "
        "de chaque scénario pour guider vos décisions."
    )

    # Scénarios prédéfinis
    scenarios_predefinis = {
        "Impact du budget": {
            "parametres_variables": [
                {"nom": "budget_annuel_max", "valeurs": [200e6, 350e6, 500e6, 700e6, 1000e6]}
            ],
        },
        "Budget vs Capacité km": {
            "parametres_variables": [
                {"nom": "budget_annuel_max", "valeurs": [300e6, 500e6, 800e6]},
                {"nom": "km_max_par_an",     "valeurs": [50, 80, 120]},
            ],
        },
        "Urgences FTVI d'abord": {
            "parametres_variables": [
                {"nom": "budget_annuel_max", "valeurs": [300e6, 500e6, 800e6]}
            ],
            "contraintes_base_extra": {"materiaux_urgence": ["FTVI"]},
        },
        "Horizon 3 vs 5 vs 10 ans": {
            "parametres_variables": [
                {"nom": "horizon_plan", "valeurs": [3, 5, 10]}
            ],
        },
    }

    mode = st.radio("Mode", ["Scénario prédéfini", "Paramétrage manuel"],
                    horizontal=True)

    if mode == "Scénario prédéfini":
        scenario_nom = st.selectbox("Choisir un scénario", list(scenarios_predefinis.keys()))
        scenario_cfg = scenarios_predefinis[scenario_nom]
        params_var = scenario_cfg["parametres_variables"]
        st.caption(f"Paramètres variables : {[p['nom'] for p in params_var]}")

    else:
        st.subheader("Définir les paramètres variables")
        nb_params = st.number_input("Nombre de paramètres à faire varier", 1, 4, 1)
        params_var = []
        for i in range(nb_params):
            col1, col2 = st.columns([1, 2])
            nom = col1.selectbox(
                f"Paramètre {i+1}",
                ["budget_annuel_max", "km_max_par_an", "horizon_plan",
                 "age_max_sans_renouvellement", "seuil_decile_prioritaire",
                 "taux_renouvellement_min_pct", "lissage_budget_pct"],
                key=f"nom_{i}",
            )
            vals_str = col2.text_input(
                f"Valeurs (séparées par virgule)", "300000000,500000000,800000000",
                key=f"vals_{i}",
            )
            try:
                vals = [float(v.strip()) for v in vals_str.split(",")]
                params_var.append({"nom": nom, "valeurs": vals})
            except Exception:
                st.warning(f"Valeurs invalides pour {nom}")

    top_n_wi = st.number_input("Tronçons analysés (pour performance)", 500, 50000, 3000)

    if st.button("▶️ Lancer l'analyse What-If", use_container_width=True, type="primary"):
        contraintes_base = {}
        if mode == "Scénario prédéfini" and "contraintes_base_extra" in scenario_cfg:
            contraintes_base.update(scenario_cfg["contraintes_base_extra"])

        payload = {
            "contraintes_base": contraintes_base,
            "parametres_variables": params_var,
            "top_n_troncons": int(top_n_wi),
        }

        with st.spinner("Calcul des scénarios …"):
            result, err = api_post("/whatif", payload)

        if err:
            st.error(f"Erreur API : {err}")
        else:
            st.success(f"✅ {result['nb_scenarios']} scénarios calculés")
            st.info(f"**Recommandation :** {result['recommandation']}")

            # Tableau des scénarios
            rows = []
            for s in result["scenarios"]:
                row = {**s["parametres"],
                       "Tronçons planifiés": s["nb_troncons_planifies"],
                       "Km renouvelés": s["km_renouveles_total"],
                       "Budget total (M EUR)": round(s["budget_total"] / 1e6, 1),
                       "Réd. risque": round(s["reduction_risque_totale"], 2),
                       "Risque résiduel (%)": s["risque_residuel_pct"],
                       "Statut": s["statut"]}
                rows.append(row)

            df_wi = pd.DataFrame(rows)
            best_id = result["meilleur_scenario"]["scenario_id"]

            st.dataframe(
                df_wi.style.highlight_min(
                    subset=["Risque résiduel (%)"], color="#abebc6"
                ),
                use_container_width=True,
            )

            # Graphe interactif
            if len(result["parametres_testes"]) == 1:
                param_x = result["parametres_testes"][0]
                fig = go.Figure()
                x_vals = [s["parametres"][param_x] for s in result["scenarios"]]
                y_risk = [s["risque_residuel_pct"] for s in result["scenarios"]]
                y_km   = [s["km_renouveles_total"] for s in result["scenarios"]]

                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_risk, mode="lines+markers",
                    name="Risque résiduel (%)", line=dict(color="#e74c3c", width=2),
                    yaxis="y",
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_km, mode="lines+markers",
                    name="Km renouvelés", line=dict(color="#2980b9", width=2, dash="dash"),
                    yaxis="y2",
                ))
                fig.update_layout(
                    xaxis_title=param_x,
                    yaxis=dict(title=dict(text="Risque résiduel (%)", font=dict(color="#e74c3c"))),
                    yaxis2=dict(title=dict(text="Km renouvelés", font=dict(color="#2980b9")),
                                overlaying="y", side="right"),
                    height=380, plot_bgcolor="white",
                    title=f"Sensibilité : {param_x}",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif len(result["parametres_testes"]) == 2:
                p1, p2 = result["parametres_testes"]
                x_vals = sorted(set(s["parametres"][p1] for s in result["scenarios"]))
                y_vals = sorted(set(s["parametres"][p2] for s in result["scenarios"]))
                z_risk = np.zeros((len(y_vals), len(x_vals)))
                for s in result["scenarios"]:
                    xi = x_vals.index(s["parametres"][p1])
                    yi = y_vals.index(s["parametres"][p2])
                    z_risk[yi, xi] = s["risque_residuel_pct"]
                fig = px.imshow(
                    z_risk,
                    x=[str(v) for v in x_vals],
                    y=[str(v) for v in y_vals],
                    labels={"x": p1, "y": p2, "color": "Risque résiduel (%)"},
                    color_continuous_scale="RdYlGn_r",
                    text_auto=".1f",
                    title="Heatmap risque résiduel selon les 2 paramètres",
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # ── Tableau de bord comparatif scénarios ─────────────────────
            st.divider()
            st.subheader("Tableau de bord comparatif des scénarios")

            # Cartes synthèse : meilleur / médian / pire
            scen_sorted = sorted(result["scenarios"], key=lambda s: s["risque_residuel_pct"])
            valides_wi = [s for s in scen_sorted if s["statut"] == "OK"]
            if len(valides_wi) >= 2:
                best_wi = valides_wi[0]
                worst_wi = valides_wi[-1]
                mid_wi   = valides_wi[len(valides_wi) // 2]

                def _wi_card(col, titre, s, bg):
                    params_str = " | ".join(f"{k}={v:.0f}" for k, v in s["parametres"].items())
                    col.markdown(f"""
                    <div style="background:{bg};border-radius:10px;padding:14px;color:white;text-align:center">
                      <b>{titre}</b><br>
                      <span style="font-size:1.4rem;font-weight:bold">{s['risque_residuel_pct']:.1f}%</span>
                      <span style="font-size:0.75rem"> risque résiduel</span><br>
                      <small>{s['km_renouveles_total']:.0f} km · {s['budget_total']/1e6:.0f} M {DEVISE}</small><br>
                      <small style="opacity:0.8">{params_str[:40]}</small>
                    </div>""", unsafe_allow_html=True)

                c_best, c_mid, c_worst = st.columns(3)
                _wi_card(c_best,  "Meilleur scénario",  best_wi,  "#27ae60")
                _wi_card(c_mid,   "Scénario médian",    mid_wi,   "#2980b9")
                _wi_card(c_worst, "Scénario le plus risqué", worst_wi, "#e74c3c")

                # Radar chart — top 4 scénarios normalisés
                top4 = valides_wi[:min(4, len(valides_wi))]
                max_km_wi  = max(s["km_renouveles_total"] for s in valides_wi) or 1
                max_rr_wi  = max(s["reduction_risque_totale"] for s in valides_wi) or 1
                max_nb_wi  = max(s["nb_troncons_planifies"] for s in valides_wi) or 1
                max_eff_wi = max(
                    s["reduction_risque_totale"] / max(s["budget_total"], 1e6)
                    for s in valides_wi
                ) or 1

                cat_radar = ["Km renouvelés", "Réd. risque", "Tronçons", "Efficience"]
                fig_radar = go.Figure()
                for s in top4:
                    eff = s["reduction_risque_totale"] / max(s["budget_total"], 1e6)
                    vals_r = [
                        s["km_renouveles_total"] / max_km_wi,
                        s["reduction_risque_totale"] / max_rr_wi,
                        s["nb_troncons_planifies"] / max_nb_wi,
                        eff / max_eff_wi,
                    ]
                    lbl = " | ".join(f"{k}={v:.0f}" for k, v in s["parametres"].items())
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals_r + [vals_r[0]],
                        theta=cat_radar + [cat_radar[0]],
                        name=f"Sc.{s['scenario_id']} ({lbl[:25]})",
                        fill="toself", opacity=0.55,
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Comparaison radar — Top 4 scénarios (valeurs normalisées 0→1)",
                    height=420, showlegend=True,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Budget vs Risque résiduel — tous scénarios
                labels_wi = [
                    " | ".join(f"{k}={v:.0f}" for k, v in s["parametres"].items())[:30]
                    for s in valides_wi
                ]
                fig_comp = go.Figure()
                fig_comp.add_bar(
                    x=labels_wi,
                    y=[s["budget_total"] / 1e6 for s in valides_wi],
                    name=f"Budget total (M {DEVISE})",
                    marker_color="#3498db",
                    yaxis="y",
                )
                fig_comp.add_trace(go.Scatter(
                    x=labels_wi,
                    y=[s["risque_residuel_pct"] for s in valides_wi],
                    name="Risque résiduel (%)",
                    mode="lines+markers",
                    marker_color="#e74c3c",
                    yaxis="y2",
                ))
                fig_comp.update_layout(
                    yaxis=dict(title=f"Budget total (M {DEVISE})"),
                    yaxis2=dict(title="Risque résiduel (%)", overlaying="y", side="right"),
                    height=380, plot_bgcolor="white",
                    title=f"Budget engagé vs Risque résiduel — tous scénarios valides",
                    xaxis=dict(tickangle=-30),
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6 — Comparaison & Explicabilité
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🧠 Comparaison & Explicabilité":
    st.title("🧠 Comparaison des stratégies & Explicabilité")
    st.caption(
        "Comparez l'optimiseur MILP avec deux baselines (glouton et aléatoire). "
        "Découvrez pourquoi chaque tronçon a été sélectionné et visualisez la frontière Pareto."
    )

    # ── Coûts EUR/km par matériau (identiques à l'API) ────────────────────
    _COUT_KM = {
        "FT": 8_000_000, "FTG": 7_500_000, "FTVI": 8_500_000,
        "PEHD": 6_000_000, "PVC": 5_500_000, "BTM": 7_000_000,
        "POLY": 6_500_000, "AC": 9_000_000,
    }
    _URGENCE_MATS = {"FTVI", "AC"}

    # ── Helpers locaux ────────────────────────────────────────────────────
    def _p1an_loc(med, age):
        if med <= 0 or age < 0:
            return 0.0
        rho = RHO_WEIBULL
        lam = med / (np.log(2) ** (1.0 / rho))
        if lam <= 0:
            return 1.0
        def S(t): return float(np.exp(-((max(t, 0) / lam) ** rho))) if t > 0 else 1.0
        sn, ss = S(age), S(age + 1)
        return max(0.0, min(1.0, 1.0 - ss / sn)) if sn > 1e-12 else 1.0

    def _enrichir(df):
        d = df.copy()
        d["age_actuel"] = 2026 - d["DDP_year"]
        if "P_casse_1an" not in d.columns:
            d["P_casse_1an"] = d.apply(
                lambda r: _p1an_loc(r["duree_mediane_pred"], r["age_actuel"]), axis=1
            )
        d["cout_renouvellement"] = d.apply(
            lambda r: _COUT_KM.get(r["MAT_grp"], 7_000_000) * r["LNG"] / 1000, axis=1
        )
        d["urgence"] = d["MAT_grp"].isin(_URGENCE_MATS).astype(int)
        d["age_ratio"] = d["age_actuel"] / d["duree_mediane_pred"].clip(lower=1)
        d["efficience"] = d["P_casse_1an"] / (d["cout_renouvellement"].clip(lower=1e3) / 1e6)
        return d

    def _greedy(d, budget_tot, km_tot):
        """Sélection par P_casse_1an décroissant jusqu'à saturation budget/km."""
        bu, ku, sel = 0.0, 0.0, []
        for _, row in d.sort_values("P_casse_1an", ascending=False).iterrows():
            c = row["cout_renouvellement"]
            k = row["LNG"] / 1000
            if bu + c <= budget_tot and ku + k <= km_tot:
                sel.append(row)
                bu += c
                ku += k
        return pd.DataFrame(sel) if sel else pd.DataFrame(columns=d.columns), bu, ku

    def _random_avg(d, budget_tot, km_tot, n_runs=10):
        """Moyenne de n_runs sélections aléatoires."""
        stats = []
        for seed in range(n_runs):
            d2 = d.sample(frac=1, random_state=seed)
            bu, ku, sel = 0.0, 0.0, []
            for _, row in d2.iterrows():
                c = row["cout_renouvellement"]
                k = row["LNG"] / 1000
                if bu + c <= budget_tot and ku + k <= km_tot:
                    sel.append(row)
                    bu += c
                    ku += k
            sub = pd.DataFrame(sel) if sel else pd.DataFrame(columns=d.columns)
            stats.append({
                "nb":         len(sub),
                "km":         sub["LNG"].sum() / 1000 if len(sub) > 0 else 0.0,
                "budget":     bu,
                "p_casse":    sub["P_casse_1an"].sum() if len(sub) > 0 else 0.0,
                "urgence_pct": sub["urgence"].mean() * 100 if len(sub) > 0 else 0.0,
            })
        return pd.DataFrame(stats).mean()

    # ── Configuration ─────────────────────────────────────────────────────
    st.subheader("Configuration")
    col1, col2, col3, col4 = st.columns(4)
    b_max_yr   = col1.number_input("Budget annuel max (M EUR)", 50, 5000, 500, key="b6") * 1_000_000
    km_max_yr  = col2.number_input("Km max/an", 10, 500, 150, key="km6")
    top_n6     = int(col3.number_input("Top N tronçons analysés", 500, 20000, 5000, key="tn6"))
    horizon6   = col4.radio("Horizon (MILP)", [1, 3, 5], index=1, horizontal=True, key="h6")

    st.caption(
        f"Enveloppe totale : **{b_max_yr/1e6:.0f} M EUR/an × {horizon6} ans = "
        f"{b_max_yr/1e6*horizon6:.0f} M EUR** | "
        f"**{km_max_yr} km/an × {horizon6} ans = {km_max_yr*horizon6} km**  "
        "_(glouton et aléatoire utilisent ces totaux sans contrainte annuelle)_"
    )

    if st.button("▶️ Lancer la comparaison des 3 stratégies", use_container_width=True, type="primary", key="run6"):

        df_all6 = charger_scoring()
        df_enr6 = _enrichir(df_all6)
        df_top6 = df_enr6.nlargest(top_n6, "P_casse_1an").copy().reset_index(drop=True)

        budget_tot = b_max_yr * horizon6
        km_tot     = km_max_yr * horizon6

        # ── 1. Glouton ────────────────────────────────────────────────────
        with st.spinner("Stratégie glouton …"):
            df_glou6, glou_budget, _glou_km = _greedy(df_top6, budget_tot, km_tot)
        glou_nb  = len(df_glou6)
        glou_km  = df_glou6["LNG"].sum() / 1000 if glou_nb > 0 else 0.0
        glou_p   = df_glou6["P_casse_1an"].sum() if glou_nb > 0 else 0.0
        glou_urg = df_glou6["urgence"].mean() * 100 if glou_nb > 0 else 0.0

        # ── 2. Aléatoire (10 runs) ────────────────────────────────────────
        with st.spinner("Stratégie aléatoire — 10 tirages …"):
            rnd = _random_avg(df_top6, budget_tot, km_tot)

        # ── 3. MILP via API ────────────────────────────────────────────────
        with st.spinner("Optimiseur MILP (API) …"):
            milp_payload = {
                "contraintes": {
                    "budget_annuel_max":         b_max_yr,
                    "budget_annuel_min":          0,
                    "km_max_par_an":              km_max_yr,
                    "km_min_par_an":              0,
                    "taux_renouvellement_min_pct": 0,
                    "lissage_budget_pct":          0,
                    "horizon_plan":               horizon6,
                    "annee_debut":                2026,
                },
                "top_n_troncons": top_n6,
                "objectif":       "maximiser_reduction_risque",
            }
            milp_res6, err6 = api_post("/optimiser", milp_payload)

        if err6 or not milp_res6:
            milp_ok6 = False
            st.warning(f"API MILP indisponible : {err6}")
            milp_nb = milp_km = milp_budget = milp_p = milp_urg = 0.0
            plan_det6 = []
        else:
            milp_ok6 = True
            g6 = milp_res6.get("resume_global", {})
            milp_nb     = g6.get("nb_troncons_planifies", 0)
            milp_km     = g6.get("km_total_renouveles", 0.0)
            milp_budget = g6.get("budget_total_engage", 0.0)
            milp_p      = g6.get("p_casse_1an_evitee", 0.0)
            plan_det6   = milp_res6.get("plan_detaille", [])
            df_plan6    = pd.DataFrame(plan_det6) if plan_det6 else pd.DataFrame()
            milp_urg    = df_plan6["MAT_grp"].isin(_URGENCE_MATS).mean() * 100 if len(df_plan6) > 0 else 0.0

        # ══════════════════════════════════════════════════════════════════
        # A — BENCHMARK
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("A — Benchmark : MILP ⚡ vs Glouton 📋 vs Aléatoire 🎲")

        # Cartes colorées
        ca, cb, cc = st.columns(3)
        def _carte(col, titre, nb, km, bud_m, p, urg, bg):
            col.markdown(f"""
            <div style="background:{bg};border-radius:12px;padding:18px;color:white;text-align:center">
              <h4 style="margin:0 0 8px">{titre}</h4>
              <span style="font-size:1.6rem;font-weight:bold">{p:.3f}</span><br>
              <span style="font-size:0.8rem">Σ P(casse/an) évitée</span><br><br>
              <b>{nb:,.0f}</b> tronçons &nbsp;·&nbsp; <b>{km:.1f} km</b><br>
              Budget engagé : <b>{bud_m:.1f} M EUR</b><br>
              % urgences (FTVI/AC) : <b>{urg:.0f}%</b>
            </div>""", unsafe_allow_html=True)

        _carte(ca, "🎲 Aléatoire",           rnd["nb"], rnd["km"], rnd["budget"]/1e6, rnd["p_casse"], rnd["urgence_pct"], "#7f8c8d")
        _carte(cb, "📋 Glouton (tri score)",  glou_nb,   glou_km,  glou_budget/1e6,   glou_p,        glou_urg,           "#2980b9")
        if milp_ok6:
            _carte(cc, "⚡ MILP (optimal)",   milp_nb,   milp_km,  milp_budget/1e6,   milp_p,        milp_urg,           "#27ae60")
        else:
            cc.warning("MILP indisponible")

        # Bar chart comparatif
        strats   = ["Aléatoire", "Glouton"]
        p_vals   = [rnd["p_casse"], glou_p]
        clrs     = ["#95a5a6", "#2980b9"]
        bud_vals = [rnd["budget"] / 1e6, glou_budget / 1e6]
        if milp_ok6:
            strats.append("MILP")
            p_vals.append(milp_p)
            clrs.append("#27ae60")
            bud_vals.append(milp_budget / 1e6)

        fig_b = go.Figure()
        fig_b.add_bar(
            x=strats, y=p_vals, marker_color=clrs,
            text=[f"{v:.3f}" for v in p_vals], textposition="outside",
            name="P(casse/an) évitée", yaxis="y",
        )
        fig_b.add_trace(go.Scatter(
            x=strats, y=bud_vals, mode="lines+markers",
            name="Budget engagé (M EUR)", line=dict(color="#e67e22", dash="dot", width=2),
            marker=dict(size=10), yaxis="y2",
        ))
        if milp_ok6 and rnd["p_casse"] > 1e-9:
            gain_rnd  = (milp_p - rnd["p_casse"]) / rnd["p_casse"] * 100
            gain_glou = (milp_p - glou_p)         / max(glou_p, 1e-9) * 100
            fig_b.add_annotation(
                x="MILP", y=milp_p,
                text=f"<b>+{gain_rnd:.0f}%</b> vs aléatoire<br>+{gain_glou:.0f}% vs glouton",
                showarrow=True, arrowhead=2, yshift=40,
                font=dict(color="#27ae60", size=13), bgcolor="white",
                bordercolor="#27ae60", borderwidth=1,
            )
        fig_b.update_layout(
            title="Réduction de risque par stratégie (même enveloppe budget/km)",
            yaxis=dict(title="Σ P_casse_1an évitée"),
            yaxis2=dict(title=dict(text="Budget engagé (M EUR)", font=dict(color="#e67e22")),
                        overlaying="y", side="right"),
            height=400, plot_bgcolor="white", showlegend=True,
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_b, use_container_width=True)

        # Tableau synthèse
        rows_b = [
            {
                "Stratégie":              "🎲 Aléatoire (moy. 10 tirages)",
                "Tronçons":               int(rnd["nb"]),
                "Km":                     round(rnd["km"], 1),
                "Budget (M EUR)":         round(rnd["budget"] / 1e6, 1),
                "P(casse/an) évitée":     round(rnd["p_casse"], 4),
                "% Urgences":             round(rnd["urgence_pct"], 0),
                "Coût / P évitée (M EUR)": round(rnd["budget"] / 1e6 / max(rnd["p_casse"], 1e-9), 1),
            },
            {
                "Stratégie":              "📋 Glouton (tri P_casse_1an)",
                "Tronçons":               glou_nb,
                "Km":                     round(glou_km, 1),
                "Budget (M EUR)":         round(glou_budget / 1e6, 1),
                "P(casse/an) évitée":     round(glou_p, 4),
                "% Urgences":             round(glou_urg, 0),
                "Coût / P évitée (M EUR)": round(glou_budget / 1e6 / max(glou_p, 1e-9), 1),
            },
        ]
        if milp_ok6:
            rows_b.append({
                "Stratégie":              "⚡ MILP (optimiseur mixte entier)",
                "Tronçons":               milp_nb,
                "Km":                     round(milp_km, 1),
                "Budget (M EUR)":         round(milp_budget / 1e6, 1),
                "P(casse/an) évitée":     round(milp_p, 4),
                "% Urgences":             round(milp_urg, 0),
                "Coût / P évitée (M EUR)": round(milp_budget / 1e6 / max(milp_p, 1e-9), 1),
            })
        st.dataframe(
            pd.DataFrame(rows_b).style.highlight_max(
                subset=["P(casse/an) évitée"], color="#abebc6"
            ).highlight_min(
                subset=["Coût / P évitée (M EUR)"], color="#abebc6"
            ),
            use_container_width=True, hide_index=True,
        )

        # ══════════════════════════════════════════════════════════════════
        # B — KPIs ENRICHIS
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("B — KPIs enrichis du plan MILP")

        if milp_ok6 and len(df_plan6) > 0:
            # Merger plan avec données enrichies pour avoir les colonnes complètes
            df_px = df_plan6.merge(
                df_enr6[["GID", "P_casse_1an", "age_actuel", "urgence",
                          "cout_renouvellement", "efficience", "age_ratio"]],
                on="GID", how="left", suffixes=("", "_e"),
            )
            p_col = "P_casse_1an" if "P_casse_1an" in df_px.columns else "P_casse_1an_e"

            tot_ftvi = int((df_all6["MAT_grp"] == "FTVI").sum())
            tot_ac   = int((df_all6["MAT_grp"] == "AC").sum())
            plan_ftvi = int((df_plan6["MAT_grp"] == "FTVI").sum()) if "MAT_grp" in df_plan6.columns else 0
            plan_ac   = int((df_plan6["MAT_grp"] == "AC").sum())  if "MAT_grp" in df_plan6.columns else 0

            p_plan_moy    = df_px[p_col].mean() if p_col in df_px else 0.0
            p_reseau_moy  = df_enr6["P_casse_1an"].mean()
            age_plan_moy  = df_px["age_actuel"].mean() if "age_actuel" in df_px.columns else 0.0
            age_reseau_moy = df_enr6["age_actuel"].mean()
            casses_evitees = float(df_px[p_col].sum()) if p_col in df_px.columns else milp_p
            cout_casse     = milp_budget / max(casses_evitees, 1e-9) / 1e6

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Casses/an évitées",       f"{casses_evitees:.2f}",
                      help="Σ P_casse_1an des tronçons planifiés = espérance annuelle de ruptures évitées")
            k2.metric("Coût par casse évitée",   f"{cout_casse:.1f} M EUR")
            k3.metric("% FTVI planifiés",         f"{plan_ftvi/max(tot_ftvi,1)*100:.0f}%",
                      delta=f"{plan_ftvi}/{tot_ftvi}")
            k4.metric("% AC planifiés",           f"{plan_ac/max(tot_ac,1)*100:.0f}%",
                      delta=f"{plan_ac}/{tot_ac}")

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("P_casse/an moyen — PLAN",   f"{p_plan_moy:.4%}",
                      delta=f"{p_plan_moy - p_reseau_moy:+.4%} vs réseau", delta_color="off")
            k6.metric("P_casse/an moyen — RÉSEAU", f"{p_reseau_moy:.4%}")
            k7.metric("Âge moyen planifié",        f"{age_plan_moy:.0f} ans",
                      delta=f"{age_plan_moy - age_reseau_moy:+.0f} vs réseau", delta_color="off")
            k8.metric("Âge moyen réseau",          f"{age_reseau_moy:.0f} ans")

            # Comparaison répartition matériaux plan vs réseau
            st.subheader("Répartition matériaux : Plan MILP vs Réseau total")
            mat_plan6   = df_plan6["MAT_grp"].value_counts(normalize=True) * 100
            mat_reseau6 = df_all6["MAT_grp"].value_counts(normalize=True)  * 100
            mats6 = sorted(set(mat_plan6.index) | set(mat_reseau6.index))
            fig_mat = go.Figure()
            fig_mat.add_bar(x=mats6, y=[mat_reseau6.get(m, 0) for m in mats6],
                            name="Réseau total", marker_color="#3498db", opacity=0.6)
            fig_mat.add_bar(x=mats6, y=[mat_plan6.get(m, 0) for m in mats6],
                            name="Plan MILP",    marker_color="#e74c3c", opacity=0.85)
            fig_mat.update_layout(
                barmode="group", height=300, plot_bgcolor="white",
                yaxis_title="% des tronçons",
                title="Sur/sous-représentation par matériau dans le plan vs le réseau",
            )
            st.plotly_chart(fig_mat, use_container_width=True)

            # Histogramme P_casse_1an : plan vs réseau
            st.subheader("Distribution P_casse_1an : Plan vs Réseau")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=df_enr6["P_casse_1an"], nbinsx=60, name="Réseau total",
                marker_color="#3498db", opacity=0.5, histnorm="probability",
            ))
            if p_col in df_px.columns:
                fig_hist.add_trace(go.Histogram(
                    x=df_px[p_col], nbinsx=40, name="Plan MILP",
                    marker_color="#e74c3c", opacity=0.75, histnorm="probability",
                ))
            fig_hist.update_layout(
                barmode="overlay", height=300, plot_bgcolor="white",
                xaxis_title="P_casse_1an", yaxis_title="Fréquence relative",
                xaxis_tickformat=".2%",
                title="Le plan doit surreprésenter les tronçons à haute P_casse_1an",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.info("KPIs enrichis disponibles une fois l'API connectée et l'optimisation lancée.")

        # ══════════════════════════════════════════════════════════════════
        # C — COURBE PARETO
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("C — Frontière Pareto : efficience marginal du renouvellement")
        st.caption(
            "Tronçons triés par **P_casse_1an / coût** (plus efficient en premier). "
            "La courbure montre le rendement décroissant — l'étoile rouge = position du plan MILP, "
            "le losange orange = glouton."
        )

        df_par = df_top6.sort_values("efficience", ascending=False).reset_index(drop=True)
        df_par["km_cumul"]      = df_par["LNG"].cumsum() / 1000
        df_par["p_cumul"]       = df_par["P_casse_1an"].cumsum()
        df_par["budget_cumul"]  = df_par["cout_renouvellement"].cumsum() / 1e6
        df_par["idx"]           = range(1, len(df_par) + 1)

        # Sous-échantillonnage pour fluidité
        step6 = max(1, len(df_par) // 800)
        df_par_s = df_par.iloc[::step6].copy()

        fig_par = go.Figure()
        fig_par.add_trace(go.Scatter(
            x=df_par_s["km_cumul"], y=df_par_s["p_cumul"],
            mode="lines", line=dict(color="#2980b9", width=2),
            fill="tozeroy", fillcolor="rgba(41,128,185,0.08)",
            name="Frontière Pareto",
            hovertemplate="km cumulés: %{x:.1f}<br>P évitée: %{y:.3f}<extra></extra>",
        ))
        if milp_ok6:
            fig_par.add_trace(go.Scatter(
                x=[milp_km], y=[milp_p],
                mode="markers+text",
                marker=dict(color="#e74c3c", size=16, symbol="star"),
                text=["MILP"], textposition="top right",
                name="Plan MILP",
                textfont=dict(color="#e74c3c", size=13),
            ))
        fig_par.add_trace(go.Scatter(
            x=[glou_km], y=[glou_p],
            mode="markers+text",
            marker=dict(color="#f39c12", size=14, symbol="diamond"),
            text=["Glouton"], textposition="top right",
            name="Glouton",
            textfont=dict(color="#f39c12", size=13),
        ))
        fig_par.add_trace(go.Scatter(
            x=[rnd["km"]], y=[rnd["p_casse"]],
            mode="markers+text",
            marker=dict(color="#7f8c8d", size=12, symbol="circle"),
            text=["Aléatoire"], textposition="bottom right",
            name="Aléatoire",
            textfont=dict(color="#7f8c8d", size=12),
        ))
        fig_par.update_layout(
            xaxis_title="Km cumulés renouvelés",
            yaxis_title="Σ P_casse_1an évitée (casses/an)",
            height=420, plot_bgcolor="white",
            title="Frontière Pareto : efficience du renouvellement",
        )
        st.plotly_chart(fig_par, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # D — EXPLICABILITÉ PAR TRONÇON
        # ══════════════════════════════════════════════════════════════════
        st.divider()
        st.subheader("D — Explicabilité : pourquoi ce tronçon est-il dans le plan ?")
        st.caption(
            "Chaque barre = un tronçon planifié. Les segments colorés montrent la contribution "
            "de chaque facteur à la décision (scores normalisés entre 0 et 1 sur le top-N analysé)."
        )

        if milp_ok6 and len(df_plan6) > 0:
            df_exp = df_plan6.merge(
                df_enr6[["GID", "P_casse_1an", "age_actuel", "urgence",
                          "cout_renouvellement", "efficience", "age_ratio"]],
                on="GID", how="left", suffixes=("", "_e"),
            )
            p_col_e = "P_casse_1an" if "P_casse_1an" in df_exp.columns else "P_casse_1an_e"

            def _norm(s):
                lo, hi = s.min(), s.max()
                return (s - lo) / (hi - lo + 1e-12)

            df_exp["f_p_casse"]    = _norm(df_exp[p_col_e].fillna(0))
            df_exp["f_age_ratio"]  = _norm(df_exp["age_ratio"].fillna(0))
            df_exp["f_efficience"] = _norm(df_exp["efficience"].fillna(0))
            df_exp["f_urgence"]    = df_exp["urgence"].fillna(0).astype(float)

            nb_exp = st.slider("Tronçons à expliquer", 5, min(150, len(df_exp)), min(30, len(df_exp)))
            df_exp_top = df_exp.head(nb_exp).copy()

            mat_col = df_exp_top["MAT_grp"].astype(str) if "MAT_grp" in df_exp_top.columns else ""
            labels_y = (df_exp_top["GID"].astype(str)
                        + " · " + mat_col
                        + " · " + df_exp_top.get("age_actuel", pd.Series([0]*len(df_exp_top))).astype(int).astype(str) + " ans")

            fig_exp = go.Figure()
            for fname, fcol, fcolor in [
                ("P(casse/an) — urgence actuelle",   "f_p_casse",    "#e74c3c"),
                ("Âge / durée médiane",               "f_age_ratio",  "#e67e22"),
                ("Efficience (P_casse/M EUR)",        "f_efficience", "#2980b9"),
                ("Matériau urgence (FTVI/AC)",         "f_urgence",    "#8e44ad"),
            ]:
                fig_exp.add_trace(go.Bar(
                    y=labels_y,
                    x=df_exp_top[fcol],
                    name=fname,
                    orientation="h",
                    marker_color=fcolor,
                    opacity=0.85,
                ))
            fig_exp.update_layout(
                barmode="stack",
                height=max(420, nb_exp * 26),
                xaxis_title="Score de contribution normalisé [0 → 1]",
                plot_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                title="Décomposition des facteurs de décision — tronçons planifiés (triés par priorité)",
                margin=dict(l=280),
            )
            st.plotly_chart(fig_exp, use_container_width=True)

            # Tableau détaillé avec toutes les colonnes explicatives
            st.subheader("Tableau détaillé des tronçons avec scores")
            cols_exp = [c for c in [
                "GID", "MAT_grp", "DIAMETRE_imp", "LNG_km", "annee_prevue",
                "age_actuel", p_col_e, "age_ratio", "efficience", "urgence",
                "cout_estime", "raison_priorite",
            ] if c in df_exp_top.columns]
            df_show = df_exp_top[cols_exp].copy()
            rename_exp = {
                "MAT_grp":        "Matériau",
                "DIAMETRE_imp":   "Ø mm",
                "LNG_km":         "Long. km",
                "annee_prevue":   "Année",
                "age_actuel":     "Âge (ans)",
                p_col_e:          "P(casse/an)",
                "age_ratio":      "Âge/Médiane",
                "efficience":     "P/M EUR",
                "urgence":        "Urgence",
                "cout_estime":    "Coût (MAD)",
                "raison_priorite": "Raison principale",
            }
            df_show = df_show.rename(columns={k: v for k, v in rename_exp.items() if k in df_show.columns})

            fmt_exp = {}
            if "P(casse/an)" in df_show.columns:   fmt_exp["P(casse/an)"]  = "{:.4%}"
            if "Âge/Médiane" in df_show.columns:   fmt_exp["Âge/Médiane"]  = "{:.2f}"
            if "P/M EUR"     in df_show.columns:   fmt_exp["P/M EUR"]      = "{:.4f}"
            if "Coût (MAD)"  in df_show.columns:   fmt_exp["Coût (MAD)"]   = "{:,.0f}"

            grad_s = [c for c in ["P(casse/an)", "Âge/Médiane"] if c in df_show.columns]
            styled = df_show.style.format(fmt_exp)
            if grad_s:
                styled = styled.background_gradient(subset=grad_s[:1], cmap="RdYlGn_r")
            st.dataframe(styled, use_container_width=True, height=420)

            # Export
            csv_exp = df_exp_top.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Exporter le plan annoté (CSV)",
                csv_exp, "plan_annote_explicabilite.csv", "text/csv",
            )

        else:
            st.info("L'explicabilité est disponible après lancement de l'optimisation MILP (API connectée).")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 7 — Carte du réseau
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Carte du réseau":
    st.title("🗺️ Carte géographique du réseau")

    df = charger_scoring()

    LAT_CANDIDATES = ["latitude", "lat", "LAT", "COORD_Y", "Y_WGS84", "y_wgs84", "Y", "y"]
    LON_CANDIDATES = ["longitude", "lon", "LON", "COORD_X", "X_WGS84", "x_wgs84", "X", "x"]
    lat_col = next((c for c in LAT_CANDIDATES if c in df.columns), None)
    lon_col = next((c for c in LON_CANDIDATES if c in df.columns), None)

    if lat_col is None or lon_col is None:
        st.info(
            "📍 **Colonnes de coordonnées non détectées dans le CSV de scoring.**\n\n"
            "Pour activer la carte interactive, ajoutez ces colonnes à "
            "`models/scoring_troncons.csv` :\n"
            "- `latitude` (ex : 33.5731)\n"
            "- `longitude` (ex : -7.5898)\n\n"
            "En attendant, voici une vue analytique alternative."
        )

        # Scatter risque × longueur par matériau
        st.subheader("Vue analytique : Risque vs Longueur par matériau")
        df_sample = df.sample(min(3000, len(df)))
        fig_alt = px.scatter(
            df_sample,
            x="LNG", y="P_casse_1an",
            color="decile_risque",
            size="DIAMETRE_imp",
            color_continuous_scale="RdYlGn_r",
            hover_data=["GID", "age_actuel", "MAT_grp"],
            labels={"LNG": "Longueur (m)", "P_casse_1an": "P(casse/an)",
                    "decile_risque": "Décile risque"},
            title="Risque annuel selon longueur et matériau",
            height=480,
        )
        fig_alt.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_alt, use_container_width=True)

        # Top 5 risques par matériau (proxy carte)
        st.subheader("Top risques par matériau")
        mats_top = [m for m in ["FTVI", "AC", "FT", "FTG", "PEHD", "BTM"] if m in df["MAT_grp"].values]
        cols_top = st.columns(min(3, len(mats_top)))
        for i, mat in enumerate(mats_top):
            sub = df[df["MAT_grp"] == mat].nlargest(5, "P_casse_1an")[
                ["GID", "age_actuel", "LNG", "DIAMETRE_imp", "P_casse_1an", "decile_risque"]
            ]
            with cols_top[i % 3]:
                st.markdown(f"**{mat}** — top 5")
                st.dataframe(
                    sub.style.format({"P_casse_1an": "{:.3%}"}),
                    use_container_width=True, hide_index=True,
                )

    else:
        st.success(f"Coordonnées détectées : `{lat_col}` / `{lon_col}`")

        col_f1, col_f2, col_f3 = st.columns(3)
        dec_min_map = col_f1.slider("Décile min", 1, 10, 5, key="map_dec")
        mat_map = col_f2.selectbox(
            "Matériau", ["Tous"] + sorted(df["MAT_grp"].unique().tolist()), key="map_mat"
        )
        n_max_map = int(col_f3.number_input("Nb max tronçons affichés", 500, 10000, 3000, key="map_n"))

        df_map = df[df["decile_risque"] >= dec_min_map].copy()
        if mat_map != "Tous":
            df_map = df_map[df_map["MAT_grp"] == mat_map]
        df_map = df_map.dropna(subset=[lat_col, lon_col])
        if len(df_map) > n_max_map:
            df_map = df_map.sample(n_max_map)

        fig_map = px.scatter_mapbox(
            df_map,
            lat=lat_col,
            lon=lon_col,
            color="decile_risque",
            size="DIAMETRE_imp",
            color_continuous_scale="RdYlGn_r",
            hover_data=["GID", "MAT_grp", "age_actuel", "P_casse_1an", "LNG"],
            zoom=10, height=600,
            title=f"Carte des tronçons — décile ≥ {dec_min_map}",
            labels={"decile_risque": "Décile risque"},
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(f"**{len(df_map):,}** tronçons · {df_map['LNG'].sum() / 1000:.0f} km affichés")
