"""
Schémas Pydantic pour l'API SOMEI — Plan de Renouvellement
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ─── Énumérations ──────────────────────────────────────────────────────────

class Materiau(str, Enum):
    FT   = "FT"
    FTG  = "FTG"
    FTVI = "FTVI"
    PEHD = "PEHD"
    PVC  = "PVC"
    BTM  = "BTM"
    POLY = "POLY"
    AC   = "AC"
    AUTRE = "AUTRE"


class StatutTroncon(str, Enum):
    EN_SERVICE = "EN SERVICE"
    ABANDONNE  = "ABANDONNE"
    EN_TRAVAUX = "EN TRAVAUX"


# ─── Tronçon ───────────────────────────────────────────────────────────────

class TronconBase(BaseModel):
    GID: int
    MAT_grp: str
    DIAMETRE_imp: float
    LNG: float                     # longueur en mètres
    DDP_year: int                  # année de pose
    nb_anomalies: int = 0
    nb_fuites_signalees: int = 0
    nb_fuites_detectees: int = 0

class TronconScored(TronconBase):
    duree_mediane_pred: float
    risk_score_50ans: float
    P_survie_10ans: float
    P_survie_20ans: float
    P_survie_30ans: float
    P_survie_50ans: float
    P_survie_70ans: float
    decile_risque: int
    top10_pourcent: int
    age_actuel: Optional[float] = None
    cout_renouvellement_estime: Optional[float] = None  # MAD

class TronconDetail(TronconScored):
    STATUT_OBJET: str = "EN SERVICE"
    abandon_type: Optional[str] = None
    event_bin: int = 0
    decade_pose: Optional[int] = None


# ─── Score à la demande ────────────────────────────────────────────────────

class ScoreRequest(BaseModel):
    MAT_grp: str = Field(..., description="Matériau du tronçon")
    DIAMETRE_imp: float = Field(..., ge=25, le=1000, description="Diamètre (mm)")
    LNG: float = Field(..., gt=0, description="Longueur (m)")
    DDP_year: int = Field(..., ge=1900, le=2025, description="Année de pose")
    nb_anomalies: int = Field(0, ge=0)
    nb_fuites_signalees: int = Field(0, ge=0)
    nb_fuites_detectees: int = Field(0, ge=0)
    taux_anomalie_par_an: float = Field(0.0, ge=0)
    DT_NB_LOGEMENT_imp: float = Field(40.0, ge=0)
    DT_FLUX_CIRCULATION_imp: float = Field(3.0, ge=0)

class ScoreResponse(BaseModel):
    duree_mediane_pred: float
    risk_score_50ans: float
    P_survie_10ans: float
    P_survie_20ans: float
    P_survie_30ans: float
    P_survie_50ans: float
    P_survie_70ans: float
    decile_risque: int
    top10_pourcent: bool
    interpretation: str


# ─── Contraintes d'optimisation ───────────────────────────────────────────

class CoutParMateriau(BaseModel):
    FT:    float = 8_000_000
    FTG:   float = 7_500_000
    FTVI:  float = 8_500_000
    PEHD:  float = 6_000_000
    PVC:   float = 5_500_000
    BTM:   float = 7_000_000
    POLY:  float = 6_500_000
    AC:    float = 9_000_000
    AUTRE: float = 7_000_000

class ContraintesOptimisation(BaseModel):
    # Financier
    budget_annuel_max: float = Field(500_000_000, description="Budget max annuel (MAD)")
    budget_annuel_min: float = Field(50_000_000,  description="Budget min annuel (MAD)")
    cout_km: CoutParMateriau = Field(default_factory=CoutParMateriau, description="Coût MAD/km par matériau")

    # Capacité
    km_max_par_an: float = Field(80.0,  description="Max km renouvelables/an")
    km_min_par_an: float = Field(10.0,  description="Min km renouvelables/an (taux cible)")
    chantiers_simultanes_max: int = Field(10, description="Chantiers simultanés max")

    # Priorisation
    seuil_decile_prioritaire: int = Field(7, ge=1, le=10, description="Décile min prioritaire")
    age_max_sans_renouvellement: int = Field(60, description="Age max avant renouvellement obligatoire (ans)")
    fuites_detectees_seuil: int = Field(1, description="Nb fuites détectées déclenchant urgence")
    materiaux_urgence: List[str] = Field(default=["FTVI", "AC"], description="Matériaux à renouveler en priorité absolue")

    # Réglementaire
    taux_renouvellement_min_pct: float = Field(1.5, description="Taux min annuel imposé par régulateur (%)")

    # Planning
    horizon_plan: int = Field(5, ge=1, le=20, description="Horizon du plan (années)")
    annee_debut: int = Field(2026, description="Année de début du plan")

    # Lissage budget
    lissage_budget_pct: float = Field(0.30, description="Variation max budget d'une année sur l'autre")


# ─── Requête d'optimisation ───────────────────────────────────────────────

class OptimisationRequest(BaseModel):
    contraintes: ContraintesOptimisation = Field(default_factory=ContraintesOptimisation)
    top_n_troncons: Optional[int] = Field(None, description="Limiter aux N tronçons les plus risqués (pour performance)")
    objectif: str = Field("maximiser_reduction_risque",
        description="maximiser_reduction_risque | minimiser_cout | equilibre")

class TronconPlanifie(BaseModel):
    GID: int
    annee_prevue: int
    MAT_grp: str
    DIAMETRE_imp: float
    LNG_km: float
    cout_estime: float
    risk_score_50ans: float
    decile_risque: int
    age_au_renouvellement: float
    raison_priorite: str

class AnneeResume(BaseModel):
    annee: int
    nb_troncons: int
    km_renouveles: float
    budget_engage: float
    reduction_risque_totale: float

class ResultatOptimisation(BaseModel):
    statut: str
    message: str
    objectif: str
    contraintes_appliquees: Dict[str, Any]
    resume_global: Dict[str, Any]
    resume_par_annee: List[AnneeResume]
    plan_detaille: List[TronconPlanifie]
    troncons_non_planifies: int
    risque_residuel_pct: float


# ─── Requête What-If ──────────────────────────────────────────────────────

class ParametreWhatIf(BaseModel):
    nom: str = Field(..., description="Nom du paramètre à faire varier")
    valeurs: List[float] = Field(..., description="Liste de valeurs à tester")

class WhatIfRequest(BaseModel):
    contraintes_base: ContraintesOptimisation = Field(default_factory=ContraintesOptimisation)
    parametres_variables: List[ParametreWhatIf]
    top_n_troncons: Optional[int] = Field(5000, description="Nb tronçons (pour performance)")

class ScenarioWhatIf(BaseModel):
    scenario_id: int
    parametres: Dict[str, float]
    statut: str
    km_renouveles_total: float
    budget_total: float
    nb_troncons_planifies: int
    reduction_risque_totale: float
    risque_residuel_pct: float

class WhatIfResponse(BaseModel):
    nb_scenarios: int
    parametres_testes: List[str]
    scenarios: List[ScenarioWhatIf]
    meilleur_scenario: ScenarioWhatIf
    recommandation: str


# ─── Statistiques globales ────────────────────────────────────────────────

class StatsReseau(BaseModel):
    total_troncons: int
    total_km: float
    age_moyen: float
    pct_top10_risque: float
    nb_fuites_actives: int
    nb_materiaux_urgence: int
    distribution_deciles: Dict[str, int]
    distribution_materiaux: Dict[str, int]
    risk_score_moyen: float
    risk_score_median: float
