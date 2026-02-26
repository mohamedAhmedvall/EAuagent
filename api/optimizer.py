"""
Moteur d'optimisation sous contraintes — Plan de Renouvellement SOMEI
=====================================================================
Algorithme : Programmation Linéaire en Nombres Entiers (MILP) via PuLP
  - Variables de décision : x[i,t] ∈ {0,1} → tronçon i renouvelé en année t
  - Objectif : maximiser la réduction totale de risque (ou minimiser le coût)
  - Contraintes : budget, capacité km, priorisation, réglementaire

What-If : variation paramétrique sur n'importe quelle contrainte
"""

import pandas as pd
import numpy as np
import pulp
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from itertools import product as itertools_product

from api.models import (
    ContraintesOptimisation, OptimisationRequest, ResultatOptimisation,
    TronconPlanifie, AnneeResume, WhatIfRequest, WhatIfResponse,
    ScenarioWhatIf, CoutParMateriau,
)

logger = logging.getLogger(__name__)

ANNEE_REF = 2026  # année courante de référence


# ─── Utilitaires ──────────────────────────────────────────────────────────

def _cout_km(materiau: str, cout_config: CoutParMateriau) -> float:
    """Retourne le coût MAD/km pour un matériau donné."""
    mapping = {
        "FT":    cout_config.FT,
        "FTG":   cout_config.FTG,
        "FTVI":  cout_config.FTVI,
        "PEHD":  cout_config.PEHD,
        "PVC":   cout_config.PVC,
        "BTM":   cout_config.BTM,
        "POLY":  cout_config.POLY,
        "AC":    cout_config.AC,
        "AUTRE": cout_config.AUTRE,
    }
    return mapping.get(materiau, cout_config.AUTRE)


def _raison_priorite(row: pd.Series, contraintes: ContraintesOptimisation) -> str:
    """Détermine la raison principale de priorité d'un tronçon."""
    raisons = []
    if row["MAT_grp"] in contraintes.materiaux_urgence:
        raisons.append(f"matériau urgence ({row['MAT_grp']})")
    if row.get("nb_fuites_detectees", 0) >= contraintes.fuites_detectees_seuil:
        raisons.append(f"fuites détectées ({int(row['nb_fuites_detectees'])})")
    age = ANNEE_REF - row["DDP_year"]
    if age >= contraintes.age_max_sans_renouvellement:
        raisons.append(f"âge critique ({int(age)} ans)")
    if row["decile_risque"] >= contraintes.seuil_decile_prioritaire:
        raisons.append(f"décile risque {int(row['decile_risque'])}/10")
    if not raisons:
        raisons.append(f"score risque {row['risk_score_50ans']:.3f}")
    return " | ".join(raisons)


def _preparer_donnees(
    df: pd.DataFrame,
    contraintes: ContraintesOptimisation,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prépare le DataFrame des tronçons éligibles avec :
    - coût estimé de renouvellement
    - âge actuel
    - flag urgence (contraint à être renouvelé)
    - priorité ordinale pour le tri
    """
    df = df.copy()
    df["age_actuel"] = ANNEE_REF - df["DDP_year"]

    # Coût estimé par tronçon (MAD)
    df["cout_renouvellement"] = df.apply(
        lambda r: _cout_km(r["MAT_grp"], contraintes.cout_km) * r["LNG"] / 1000,
        axis=1,
    )

    # Flag urgence absolue
    df["urgence"] = (
        df["MAT_grp"].isin(contraintes.materiaux_urgence) |
        (df["nb_fuites_detectees"] >= contraintes.fuites_detectees_seuil) |
        (df["age_actuel"] >= contraintes.age_max_sans_renouvellement)
    ).astype(int)

    # Score priorité composite (pour tri initial)
    df["priorite_score"] = (
        df["risk_score_50ans"] * 0.5 +
        (df["decile_risque"] / 10) * 0.3 +
        df["urgence"] * 0.2
    )

    # Réduction de risque = risk_score_50ans (valeur à récupérer si renouvelé)
    df["reduction_risque"] = df["risk_score_50ans"]

    # Limiter aux N tronçons les plus risqués pour performance
    if top_n is not None:
        df = df.nlargest(top_n, "priorite_score").reset_index(drop=True)

    return df


# ─── Moteur MILP principal ─────────────────────────────────────────────────

def optimiser_plan(
    df_troncons: pd.DataFrame,
    contraintes: ContraintesOptimisation,
    objectif: str = "maximiser_reduction_risque",
    top_n: Optional[int] = None,
    time_limit_sec: int = 60,
) -> ResultatOptimisation:
    """
    Optimisation MILP du plan de renouvellement pluriannuel.

    Paramètres
    ----------
    df_troncons  : DataFrame issu du scoring (models/scoring_troncons.csv)
    contraintes  : ContraintesOptimisation
    objectif     : 'maximiser_reduction_risque' | 'minimiser_cout' | 'equilibre'
    top_n        : limiter aux N tronçons (perf)
    time_limit_sec : limite de temps du solveur

    Retourne
    --------
    ResultatOptimisation
    """
    t0 = time.time()

    # 1. Préparation des données
    df = _preparer_donnees(df_troncons, contraintes, top_n)
    n = len(df)
    T = contraintes.horizon_plan
    annees = list(range(contraintes.annee_debut, contraintes.annee_debut + T))

    total_km_reseau = df_troncons["LNG"].sum() / 1000
    taux_renouv_min_km = total_km_reseau * contraintes.taux_renouvellement_min_pct / 100

    logger.info(f"MILP — {n} tronçons × {T} années = {n*T} variables binaires")

    # 2. Problème PuLP
    prob = pulp.LpProblem("plan_renouvellement_somei", pulp.LpMaximize)

    # Variables de décision : x[i][t] = 1 si tronçon i renouvelé en année t
    x = pulp.LpVariable.dicts(
        "x",
        ((i, t) for i in range(n) for t in range(T)),
        cat="Binary",
    )

    # 3. Objectif
    risk_scores = df["reduction_risque"].values
    couts       = df["cout_renouvellement"].values
    lngs_km     = df["LNG"].values / 1000

    if objectif == "maximiser_reduction_risque":
        prob += pulp.lpSum(
            risk_scores[i] * x[i, t] for i in range(n) for t in range(T)
        )
    elif objectif == "minimiser_cout":
        prob += -pulp.lpSum(
            couts[i] * x[i, t] for i in range(n) for t in range(T)
        )
    else:  # equilibre
        max_risk = max(risk_scores) * n if max(risk_scores) > 0 else 1
        max_cout = sum(couts) if sum(couts) > 0 else 1
        prob += pulp.lpSum(
            (risk_scores[i] / max_risk - 0.3 * couts[i] / max_cout) * x[i, t]
            for i in range(n) for t in range(T)
        )

    # 4. Contraintes

    # C1 — Chaque tronçon renouvelé au plus une fois sur tout l'horizon
    for i in range(n):
        prob += pulp.lpSum(x[i, t] for t in range(T)) <= 1, f"unicite_{i}"

    # C2 — Urgences : tronçons avec urgence=1 DOIVENT être renouvelés
    urgences_idx = df[df["urgence"] == 1].index.tolist()
    for i in urgences_idx:
        if i < n:
            prob += pulp.lpSum(x[i, t] for t in range(T)) == 1, f"urgence_{i}"

    # C3 — Budget annuel max et min
    for t in range(T):
        budget_t = pulp.lpSum(couts[i] * x[i, t] for i in range(n))
        prob += budget_t <= contraintes.budget_annuel_max, f"budget_max_{t}"
        prob += budget_t >= contraintes.budget_annuel_min, f"budget_min_{t}"

    # C4 — Km annuel max et min
    for t in range(T):
        km_t = pulp.lpSum(lngs_km[i] * x[i, t] for i in range(n))
        prob += km_t <= contraintes.km_max_par_an,  f"km_max_{t}"
        prob += km_t >= max(taux_renouv_min_km, contraintes.km_min_par_an), f"km_min_{t}"

    # C5 — Lissage budget (variation max entre années consécutives)
    alpha = contraintes.lissage_budget_pct
    for t in range(T - 1):
        b_t  = pulp.lpSum(couts[i] * x[i, t]     for i in range(n))
        b_t1 = pulp.lpSum(couts[i] * x[i, t + 1] for i in range(n))
        prob += b_t1 <= b_t * (1 + alpha) + contraintes.budget_annuel_max * 0.01, f"lissage_up_{t}"
        prob += b_t1 >= b_t * (1 - alpha) - contraintes.budget_annuel_max * 0.01, f"lissage_dn_{t}"

    # C6 — Chantiers simultanés max (proxy : nb tronçons/an)
    for t in range(T):
        prob += (
            pulp.lpSum(x[i, t] for i in range(n)) <= contraintes.chantiers_simultanes_max * 500,
            f"chantiers_max_{t}",
        )

    # 5. Résolution
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_sec, gapRel=0.02)
    prob.solve(solver)

    statut = pulp.LpStatus[prob.status]
    elapsed = time.time() - t0
    logger.info(f"Solveur terminé en {elapsed:.1f}s — statut : {statut}")

    if statut not in ("Optimal", "Not Solved"):
        # Retour dégradé si infaisable
        return ResultatOptimisation(
            statut=statut,
            message=f"Optimisation {statut} en {elapsed:.1f}s. Vérifier les contraintes (budget_min trop élevé ?).",
            objectif=objectif,
            contraintes_appliquees=contraintes.model_dump(),
            resume_global={},
            resume_par_annee=[],
            plan_detaille=[],
            troncons_non_planifies=n,
            risque_residuel_pct=100.0,
        )

    # 6. Extraction des résultats
    plan: List[TronconPlanifie] = []
    for i in range(n):
        for t in range(T):
            val = pulp.value(x[i, t])
            if val is not None and val > 0.5:
                row = df.iloc[i]
                plan.append(TronconPlanifie(
                    GID=int(row["GID"]),
                    annee_prevue=annees[t],
                    MAT_grp=row["MAT_grp"],
                    DIAMETRE_imp=float(row["DIAMETRE_imp"]),
                    LNG_km=float(row["LNG"]) / 1000,
                    cout_estime=float(couts[i]),
                    risk_score_50ans=float(row["risk_score_50ans"]),
                    decile_risque=int(row["decile_risque"]),
                    age_au_renouvellement=float(row["age_actuel"]) + t,
                    raison_priorite=_raison_priorite(row, contraintes),
                ))

    # 7. Résumé par année
    resume_par_annee: List[AnneeResume] = []
    for t, annee in enumerate(annees):
        plan_t = [p for p in plan if p.annee_prevue == annee]
        resume_par_annee.append(AnneeResume(
            annee=annee,
            nb_troncons=len(plan_t),
            km_renouveles=sum(p.LNG_km for p in plan_t),
            budget_engage=sum(p.cout_estime for p in plan_t),
            reduction_risque_totale=sum(p.risk_score_50ans for p in plan_t),
        ))

    # 8. Résumé global
    gids_planifies = {p.GID for p in plan}
    n_non_planifies = len(df) - len(plan)
    risque_total = df["risk_score_50ans"].sum()
    risque_reduit = sum(p.risk_score_50ans for p in plan)
    risque_residuel_pct = (
        (risque_total - risque_reduit) / risque_total * 100
        if risque_total > 0 else 0.0
    )

    resume_global = {
        "nb_troncons_planifies": len(plan),
        "nb_troncons_total_eligible": n,
        "km_total_renouveles": round(sum(p.LNG_km for p in plan), 2),
        "budget_total_engage": round(sum(p.cout_estime for p in plan), 0),
        "reduction_risque_totale": round(risque_reduit, 2),
        "risque_residuel_pct": round(risque_residuel_pct, 2),
        "pct_urgences_traites": round(
            len([p for p in plan if df.loc[df["GID"] == p.GID, "urgence"].values[0] == 1
                 if len(df.loc[df["GID"] == p.GID]) > 0]) /
            max(len(urgences_idx), 1) * 100, 1
        ) if urgences_idx else 100.0,
        "temps_resolution_sec": round(elapsed, 1),
        "statut_solveur": statut,
    }

    return ResultatOptimisation(
        statut="OK",
        message=f"Plan optimisé en {elapsed:.1f}s — {len(plan)} tronçons sur {T} ans.",
        objectif=objectif,
        contraintes_appliquees=contraintes.model_dump(),
        resume_global=resume_global,
        resume_par_annee=resume_par_annee,
        plan_detaille=plan,
        troncons_non_planifies=n_non_planifies,
        risque_residuel_pct=round(risque_residuel_pct, 2),
    )


# ─── Moteur What-If ────────────────────────────────────────────────────────

def _scenario_rapide(
    df: pd.DataFrame,
    contraintes: ContraintesOptimisation,
    top_n: int = 5000,
) -> Dict[str, Any]:
    """
    Version allégée de l'optimisation pour les scénarios what-if (heuristique greedy
    + LP relaxé) afin de rester interactif.
    """
    df_prep = _preparer_donnees(df, contraintes, top_n)
    n = len(df_prep)
    T = contraintes.horizon_plan
    annees = list(range(contraintes.annee_debut, contraintes.annee_debut + T))

    total_km_reseau = df["LNG"].sum() / 1000
    taux_min_km = total_km_reseau * contraintes.taux_renouvellement_min_pct / 100

    # Heuristique greedy : trier par score de priorité décroissant
    df_prep = df_prep.sort_values("priorite_score", ascending=False).reset_index(drop=True)

    plan = []
    budgets = [0.0] * T
    kms    = [0.0] * T
    risk_reduit = 0.0

    for _, row in df_prep.iterrows():
        cout = row["cout_renouvellement"]
        lng_km = row["LNG"] / 1000
        risk = row["risk_score_50ans"]
        placed = False

        for t in range(T):
            if (budgets[t] + cout <= contraintes.budget_annuel_max and
                    kms[t] + lng_km <= contraintes.km_max_par_an):
                # Vérif lissage budget
                if t > 0 and budgets[t - 1] > 0:
                    variation = abs(budgets[t] + cout - budgets[t - 1]) / budgets[t - 1]
                    if variation > contraintes.lissage_budget_pct + 0.5:
                        continue
                budgets[t] += cout
                kms[t] += lng_km
                risk_reduit += risk
                plan.append({"annee": annees[t], "risk": risk, "cout": cout, "lng_km": lng_km})
                placed = True
                break

    risque_total = df_prep["risk_score_50ans"].sum()
    risque_residuel_pct = (risque_total - risk_reduit) / max(risque_total, 1e-9) * 100

    return {
        "statut": "OK",
        "km_renouveles_total": round(sum(p["lng_km"] for p in plan), 2),
        "budget_total": round(sum(p["cout"] for p in plan), 0),
        "nb_troncons_planifies": len(plan),
        "reduction_risque_totale": round(risk_reduit, 4),
        "risque_residuel_pct": round(risque_residuel_pct, 2),
    }


def whatif_analyse(
    df_troncons: pd.DataFrame,
    request: WhatIfRequest,
) -> WhatIfResponse:
    """
    Analyse what-if : varie les paramètres de contrainte et compare les résultats.
    """
    # Construire la grille de scénarios
    param_noms = [p.nom for p in request.parametres_variables]
    param_valeurs = [p.valeurs for p in request.parametres_variables]

    scenarios_grid = list(itertools_product(*param_valeurs))
    scenarios: List[ScenarioWhatIf] = []

    top_n = request.top_n_troncons or 5000

    for sid, combo in enumerate(scenarios_grid):
        params_dict = dict(zip(param_noms, combo))

        # Copier les contraintes de base et appliquer les valeurs
        c_dict = request.contraintes_base.model_dump()
        for nom, val in params_dict.items():
            # Support des chemins imbriqués (ex. "cout_km.FT")
            parts = nom.split(".")
            obj = c_dict
            for part in parts[:-1]:
                obj = obj[part]
            obj[parts[-1]] = val

        try:
            contraintes_scenario = ContraintesOptimisation(**c_dict)
            res = _scenario_rapide(df_troncons, contraintes_scenario, top_n)
            scenario = ScenarioWhatIf(
                scenario_id=sid,
                parametres=params_dict,
                statut=res["statut"],
                km_renouveles_total=res["km_renouveles_total"],
                budget_total=res["budget_total"],
                nb_troncons_planifies=res["nb_troncons_planifies"],
                reduction_risque_totale=res["reduction_risque_totale"],
                risque_residuel_pct=res["risque_residuel_pct"],
            )
        except Exception as e:
            scenario = ScenarioWhatIf(
                scenario_id=sid,
                parametres=params_dict,
                statut=f"ERREUR: {str(e)[:80]}",
                km_renouveles_total=0,
                budget_total=0,
                nb_troncons_planifies=0,
                reduction_risque_totale=0,
                risque_residuel_pct=100.0,
            )
        scenarios.append(scenario)

    # Meilleur scénario = minimum risque résiduel
    valides = [s for s in scenarios if s.statut == "OK"]
    meilleur = min(valides, key=lambda s: s.risque_residuel_pct) if valides else scenarios[0]

    # Recommandation automatique
    if valides:
        best_params = ", ".join(f"{k}={v}" for k, v in meilleur.parametres.items())
        recommandation = (
            f"Le scénario optimal ({best_params}) réduit le risque résiduel à "
            f"{meilleur.risque_residuel_pct:.1f}% en renouvelant "
            f"{meilleur.km_renouveles_total:.1f} km pour "
            f"{meilleur.budget_total/1e6:.1f} M MAD."
        )
    else:
        recommandation = "Aucun scénario valide — assouplir les contraintes budget/km."

    return WhatIfResponse(
        nb_scenarios=len(scenarios),
        parametres_testes=param_noms,
        scenarios=scenarios,
        meilleur_scenario=meilleur,
        recommandation=recommandation,
    )
