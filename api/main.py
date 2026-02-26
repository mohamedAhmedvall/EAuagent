"""
API FastAPI — SOMEI Plan de Renouvellement du Réseau d'Eau Potable
==================================================================

Endpoints :
  GET  /                         → santé / info
  GET  /stats                    → statistiques globales du réseau
  GET  /troncons                 → liste paginée des tronçons scorés
  GET  /troncons/{gid}           → détail d'un tronçon
  POST /score                    → scorer un tronçon ad hoc (Weibull AFT)
  POST /optimiser                → plan pluriannuel MILP
  POST /whatif                   → analyse what-if paramétrique
  GET  /contraintes/defaut       → valeurs par défaut des contraintes

Lancement :
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import math
import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    ScoreRequest, ScoreResponse,
    OptimisationRequest, ResultatOptimisation,
    WhatIfRequest, WhatIfResponse,
    ContraintesOptimisation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Chemins ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_CSV = os.path.join(BASE_DIR, "models", "scoring_troncons.csv")
DATA_CSV  = os.path.join(BASE_DIR, "data",   "dataset_B_simple.csv")

# ─── Chargement paresseux des données ─────────────────────────────────────
@lru_cache(maxsize=1)
def _load_scoring() -> pd.DataFrame:
    logger.info(f"Chargement scoring : {SCORE_CSV}")
    df = pd.read_csv(SCORE_CSV)
    df["age_actuel"] = 2026 - df["DDP_year"]
    return df

@lru_cache(maxsize=1)
def _load_raw() -> pd.DataFrame:
    logger.info(f"Chargement dataset raw : {DATA_CSV}")
    return pd.read_csv(DATA_CSV)

# Chargement du modèle Weibull AFT (lazy)
_waft_model = None

def _get_weibull_model():
    global _waft_model
    if _waft_model is None:
        from lifelines import WeibullAFTFitter
        df = _load_raw()
        duration_col = "duration_years"
        event_col    = "event_bin"

        mat_counts = df["MAT_grp"].value_counts()
        mats_keep  = mat_counts[mat_counts > 500].index.tolist()
        if "FT" in mats_keep:
            mats_keep.remove("FT")
        mat_dummies = pd.get_dummies(df["MAT_grp"], prefix="mat", drop_first=False)
        mat_cols = [f"mat_{m}" for m in mats_keep]

        covariates_num = [
            "DIAMETRE_imp", "LNG_log", "DDP_year",
            "nb_anomalies", "nb_fuites_signalees", "nb_fuites_detectees",
            "taux_anomalie_par_an", "DT_NB_LOGEMENT_imp", "DT_FLUX_CIRCULATION_imp",
        ]
        model_df = df[[duration_col, event_col]].copy()
        for col in covariates_num:
            model_df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in mat_cols:
            model_df[col] = mat_dummies[col].values if col in mat_dummies.columns else 0

        model_df = model_df.dropna()
        model_df = model_df[model_df[duration_col] > 0]

        logger.info("Ajustement Weibull AFT …")
        waft = WeibullAFTFitter(penalizer=0.01)
        waft.fit(model_df, duration_col=duration_col, event_col=event_col)
        _waft_model = (waft, mat_cols)
        logger.info("Weibull AFT prêt.")
    return _waft_model


# ─── Application FastAPI ──────────────────────────────────────────────────
app = FastAPI(
    title="SOMEI — API Plan de Renouvellement",
    description=(
        "API pour la prédiction de risque des tronçons du réseau d'eau potable "
        "et l'optimisation sous contraintes du plan pluriannuel de renouvellement."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Santé"])
def racine():
    df = _load_scoring()
    return {
        "service": "SOMEI API — Plan de Renouvellement",
        "version": "1.0.0",
        "troncons_charges": len(df),
        "status": "OK",
    }


@app.get("/stats", tags=["Réseau"])
def statistiques_reseau():
    """Statistiques globales du réseau scoré."""
    from api.optimizer import ajouter_p_casse_1an
    df = ajouter_p_casse_1an(_load_scoring())
    age_actuel = 2026 - df["DDP_year"]

    dist_deciles = df["decile_risque"].value_counts().sort_index().to_dict()
    dist_mat     = df["MAT_grp"].value_counts().to_dict()
    total_km     = df["LNG"].sum() / 1000

    return {
        "total_troncons": int(len(df)),
        "total_km": round(total_km, 1),
        "km_min_reglementaire_par_an": round(total_km * 1.0 / 100, 1),  # loi 1%/an
        "age_moyen_ans": round(float(age_actuel.mean()), 1),
        "age_median_ans": round(float(age_actuel.median()), 1),
        "pct_top10_risque": round(df["top10_pourcent"].mean() * 100, 1),
        "nb_fuites_actives": int((df["nb_fuites_detectees"] >= 1).sum()),
        "nb_materiaux_urgence_FTVI": int((df["MAT_grp"] == "FTVI").sum()),
        # P_casse_1an — probabilité de casse dans la prochaine année
        "p_casse_1an_moyen": round(float(df["P_casse_1an"].mean()), 6),
        "p_casse_1an_median": round(float(df["P_casse_1an"].median()), 6),
        "p_casse_1an_p90": round(float(df["P_casse_1an"].quantile(0.9)), 6),
        "nb_troncons_p_casse_1an_gt_1pct": int((df["P_casse_1an"] >= 0.01).sum()),
        "nb_troncons_p_casse_1an_gt_5pct": int((df["P_casse_1an"] >= 0.05).sum()),
        # risk_score_50ans — perspective long terme
        "distribution_deciles": {str(k): int(v) for k, v in dist_deciles.items()},
        "distribution_materiaux": {str(k): int(v) for k, v in dist_mat.items()},
        "risk_score_moyen": round(float(df["risk_score_50ans"].mean()), 4),
        "risk_score_median": round(float(df["risk_score_50ans"].median()), 4),
        "risk_score_p90": round(float(df["risk_score_50ans"].quantile(0.9)), 4),
    }


@app.get("/troncons", tags=["Tronçons"])
def lister_troncons(
    page: int = Query(1, ge=1),
    taille_page: int = Query(50, ge=1, le=500),
    decile_min: Optional[int] = Query(None, ge=1, le=10),
    materiau: Optional[str] = Query(None),
    top10_seulement: bool = Query(False),
    tri: str = Query("risk_score_50ans", description="Colonne de tri"),
    ordre: str = Query("desc", description="asc | desc"),
):
    """Liste paginée des tronçons avec filtres."""
    df = _load_scoring().copy()

    if decile_min is not None:
        df = df[df["decile_risque"] >= decile_min]
    if materiau:
        df = df[df["MAT_grp"].str.upper() == materiau.upper()]
    if top10_seulement:
        df = df[df["top10_pourcent"] == 1]

    col_tri = tri if tri in df.columns else "risk_score_50ans"
    df = df.sort_values(col_tri, ascending=(ordre == "asc"))

    total = len(df)
    debut = (page - 1) * taille_page
    fin   = debut + taille_page
    page_df = df.iloc[debut:fin]

    records = page_df.replace({float("nan"): None}).to_dict(orient="records")
    return {
        "total": total,
        "page": page,
        "taille_page": taille_page,
        "nb_pages": math.ceil(total / taille_page),
        "troncons": records,
    }


@app.get("/troncons/{gid}", tags=["Tronçons"])
def detail_troncon(gid: int):
    """Détail complet d'un tronçon par son GID."""
    df = _load_scoring()
    row = df[df["GID"] == gid]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Tronçon GID={gid} introuvable.")
    record = row.iloc[0].replace({float("nan"): None}).to_dict()
    return record


@app.post("/score", tags=["Scoring"])
def scorer_troncon(req: ScoreRequest):
    """
    Score un tronçon avec les caractéristiques fournies via le modèle Weibull AFT.
    Utile pour simuler un nouveau tronçon ou tester un changement de matériau.
    """
    waft, mat_cols = _get_weibull_model()
    import numpy as np

    lng_log = float(np.log1p(req.LNG))
    profile = {
        "DIAMETRE_imp": req.DIAMETRE_imp,
        "LNG_log": lng_log,
        "DDP_year": req.DDP_year,
        "nb_anomalies": req.nb_anomalies,
        "nb_fuites_signalees": req.nb_fuites_signalees,
        "nb_fuites_detectees": req.nb_fuites_detectees,
        "taux_anomalie_par_an": req.taux_anomalie_par_an,
        "DT_NB_LOGEMENT_imp": req.DT_NB_LOGEMENT_imp,
        "DT_FLUX_CIRCULATION_imp": req.DT_FLUX_CIRCULATION_imp,
    }
    for mc in mat_cols:
        profile[mc] = 0
    mat_col_key = f"mat_{req.MAT_grp}"
    if mat_col_key in mat_cols:
        profile[mat_col_key] = 1

    profile_df = pd.DataFrame([profile])

    median_pred = float(waft.predict_median(profile_df).iloc[0])
    horizons = [10, 20, 30, 50, 70]
    surv = {}
    for h in horizons:
        sf = waft.predict_survival_function(profile_df, times=[h])
        surv[h] = float(sf.iloc[0, 0])

    risk_score = 1 - surv[50]
    age_actuel = 2026 - req.DDP_year

    # P(casse dans la prochaine année | survie jusqu'à aujourd'hui)
    from api.optimizer import p_casse_1an as _p1an
    p1an = _p1an(median_pred, age_actuel)

    # Décile (relativement au scoring global)
    df_score = _load_scoring()
    decile = int(pd.cut(
        [risk_score],
        bins=df_score["risk_score_50ans"].quantile(np.linspace(0, 1, 11)).values,
        labels=range(1, 11),
        include_lowest=True,
    )[0])

    # Interprétation basée sur P_casse_1an (plus pertinente pour l'action immédiate)
    if p1an >= 0.05:
        interpretation = f"CRITIQUE — P(casse cette année) = {p1an:.1%} → renouvellement immédiat"
    elif p1an >= 0.01:
        interpretation = f"ÉLEVÉ — P(casse cette année) = {p1an:.1%} → à planifier sous 2 ans"
    elif p1an >= 0.001:
        interpretation = f"MODÉRÉ — P(casse cette année) = {p1an:.1%} → planification 3-5 ans"
    else:
        interpretation = f"FAIBLE — P(casse cette année) = {p1an:.2%} → maintien en service"

    return ScoreResponse(
        duree_mediane_pred=round(median_pred, 1),
        P_casse_1an=round(p1an, 6),
        risk_score_50ans=round(risk_score, 4),
        P_survie_10ans=round(surv[10], 4),
        P_survie_20ans=round(surv[20], 4),
        P_survie_30ans=round(surv[30], 4),
        P_survie_50ans=round(surv[50], 4),
        P_survie_70ans=round(surv[70], 4),
        decile_risque=decile,
        top10_pourcent=(risk_score >= df_score["risk_score_50ans"].quantile(0.9)),
        interpretation=interpretation,
    )


@app.post("/optimiser", tags=["Optimisation"])
def optimiser_plan(req: OptimisationRequest):
    """
    Génère un plan pluriannuel de renouvellement optimisé sous contraintes.

    - Objectif configurable : maximiser réduction risque / minimiser coût / équilibre
    - Contraintes : budget, km/an, urgences, réglementaire, lissage
    - Retourne un plan détaillé par tronçon et un résumé par année
    """
    from api.optimizer import optimiser_plan as _opt

    df = _load_scoring()
    result = _opt(
        df_troncons=df,
        contraintes=req.contraintes,
        objectif=req.objectif,
        top_n=req.top_n_troncons,
        time_limit_sec=90,
    )
    return result


@app.post("/whatif", tags=["What-If"])
def whatif(req: WhatIfRequest):
    """
    Analyse what-if paramétrique : varie une ou plusieurs contraintes
    et compare les résultats de chaque scénario.

    Exemple de corps :
    ```json
    {
      "contraintes_base": {},
      "parametres_variables": [
        {"nom": "budget_annuel_max", "valeurs": [300000000, 500000000, 800000000]},
        {"nom": "km_max_par_an", "valeurs": [50, 80, 120]}
      ],
      "top_n_troncons": 3000
    }
    ```
    """
    from api.optimizer import whatif_analyse

    df = _load_scoring()
    return whatif_analyse(df, req)


@app.get("/contraintes/defaut", tags=["Contraintes"])
def contraintes_defaut():
    """Retourne les valeurs par défaut de toutes les contraintes."""
    return ContraintesOptimisation().model_dump()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
