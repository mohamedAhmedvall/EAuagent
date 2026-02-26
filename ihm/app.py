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
API_URL = os.getenv("SOMEI_API_URL", "http://localhost:8000")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_CSV = os.path.join(BASE_DIR, "models", "scoring_troncons.csv")

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
    margin: 4px;
  }
  .metric-value { font-size: 2rem; font-weight: bold; }
  .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
  .urgent-badge {
    background: #e74c3c; color: white; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem; font-weight: bold;
  }
  .ok-badge {
    background: #27ae60; color: white; border-radius: 6px;
    padding: 2px 8px; font-size: 0.78rem;
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/43/Water_drop_001.jpg",
             width=80)
    st.title("💧 SOMEI")
    st.caption("Plan de Renouvellement — Réseau Eau Potable")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Tableau de bord", "🔍 Explorer les tronçons",
         "🎯 Scorer un tronçon", "⚙️ Optimisation du plan", "🔄 Analyse What-If"],
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
            taux_anomalie = st.number_input("Taux anomalie/an", 0.0, 1.0, 0.0, step=0.01)
        submitted = st.form_submit_button("🔬 Calculer le score", use_container_width=True)

    if submitted:
        payload = {
            "MAT_grp": mat,
            "DIAMETRE_imp": diametre,
            "LNG": lng,
            "DDP_year": annee_pose,
            "nb_anomalies": nb_anomalies,
            "nb_fuites_signalees": nb_fuites_sig,
            "nb_fuites_detectees": nb_fuites_det,
            "taux_anomalie_par_an": taux_anomalie,
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

            # Courbe de survie
            st.subheader("Probabilités de survie")
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
                xaxis_title="Horizon (années)",
                yaxis_title="P(survie)",
                yaxis_range=[0, 1.05],
                height=300, plot_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — Optimisation du plan
# ═══════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Optimisation du plan":
    st.title("⚙️ Optimisation du plan de renouvellement")

    st.info(
        "Configure les contraintes puis lance l'optimisation MILP "
        "(Programmation Linéaire en Nombres Entiers) pour obtenir un plan "
        "pluriannuel optimal."
    )

    with st.expander("💰 Contraintes financières", expanded=True):
        col1, col2 = st.columns(2)
        budget_max = col1.number_input("Budget annuel max (M MAD)", 50, 5000, 500) * 1_000_000
        budget_min = col2.number_input("Budget annuel min (M MAD)", 0, 500, 50) * 1_000_000

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

    with st.expander("📅 Horizon du plan"):
        col1, col2 = st.columns(2)
        horizon = col1.number_input("Horizon (années)", 1, 20, 5)
        annee_debut = col2.number_input("Année de début", 2025, 2035, 2026)
        objectif = st.selectbox(
            "Objectif d'optimisation",
            ["maximiser_reduction_risque", "minimiser_cout", "equilibre"],
            format_func=lambda x: {
                "maximiser_reduction_risque": "Maximiser la réduction de risque",
                "minimiser_cout": "Minimiser le coût total",
                "equilibre": "Équilibre coût / risque",
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
                "horizon_plan": horizon,
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
            col3.metric("Budget total", f"{g.get('budget_total_engage',0)/1e6:.1f} M MAD")
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
                    name="Budget (M MAD)", mode="lines+markers",
                    marker_color="#e74c3c", yaxis="y2",
                ))
                fig.update_layout(
                    yaxis=dict(title="Km renouvelés"),
                    yaxis2=dict(title="Budget (M MAD)", overlaying="y", side="right"),
                    height=350, plot_bgcolor="white", legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    df_annees.rename(columns={
                        "annee": "Année", "nb_troncons": "Tronçons",
                        "km_renouveles": "Km", "budget_engage": "Budget (M MAD)",
                        "reduction_risque_totale": "Réd. risque",
                    }),
                    use_container_width=True,
                )

            # Plan détaillé
            if result["plan_detaille"]:
                st.subheader("Plan détaillé par tronçon")
                df_plan = pd.DataFrame(result["plan_detaille"])
                df_plan["cout_estime"] = df_plan["cout_estime"] / 1e6
                st.dataframe(
                    df_plan.rename(columns={
                        "GID": "GID", "annee_prevue": "Année",
                        "MAT_grp": "Matériau", "DIAMETRE_imp": "Diamètre",
                        "LNG_km": "Longueur (km)", "cout_estime": "Coût (M MAD)",
                        "risk_score_50ans": "Score risque",
                        "decile_risque": "Décile",
                        "raison_priorite": "Raison",
                    }),
                    use_container_width=True, height=400,
                )
                csv = df_plan.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Exporter le plan (CSV)", csv,
                                   "plan_renouvellement.csv", "text/csv")


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
                       "Budget total (M MAD)": round(s["budget_total"] / 1e6, 1),
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
                    yaxis=dict(title="Risque résiduel (%)", titlefont=dict(color="#e74c3c")),
                    yaxis2=dict(title="Km renouvelés", overlaying="y", side="right",
                                titlefont=dict(color="#2980b9")),
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
