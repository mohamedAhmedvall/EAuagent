
# RAPPORT DE SYNTHÈSE — ANALYSE DE SURVIE DU RÉSEAU D'EAU POTABLE

## 1. DONNÉES
- **194 754 tronçons** analysés
- **31 152 abandons** (16.0%) dont 27 653 préventifs et 3 499 correctifs
- **163 602 tronçons encore en service** (censurés)
- Covariables : matériau (10 types), diamètre, longueur, année de pose, anomalies, fuites, environnement urbain

## 2. MODÈLES AJUSTÉS

### 2.1 Cox PH (Dataset B — tous abandons)
- **C-index = 0.586** (discrimination modeste)
- **Proportionnalité violée** pour quasi toutes les variables (attendu avec n=194K)
- Facteurs de risque majeurs : BTM (HR=12.7), PEHD (HR=12.0), LNG_log (HR=3.0), FTG (HR=2.2)
- Facteurs protecteurs : POLY, taux_anomalie_par_an, DT_FLUX_CIRCULATION_imp

### 2.2 Weibull AFT (Dataset B — MODÈLE RETENU)
- **C-index = 0.750** (bonne discrimination)
- **AIC = 370 974** (meilleur parmi les paramétriques)
- **ρ = 2.78** → risque croissant avec l'âge (vieillissement confirmé)
- Durées médianes prédites par matériau :
  - FTVI : 27 ans | PEHD : 40 ans | FT : 54 ans
  - POLY : 74 ans | PVC : 75 ans | FTG : 87 ans | BTM : 95 ans

### 2.3 Risques compétitifs (Dataset A — cause-specific Cox)
- **Préventif** (C-index=0.789) : fort pouvoir discriminant
  - Les tronçons avec anomalies/fuites sont MOINS abandonnés préventivement (biais de surveillance)
  - DDP_year significatif (HR=1.049) : les récents abandonnés plus vite
- **Correctif** : les anomalies/fuites sont des prédicteurs forts
  - nb_fuites_detectees : HR=258 (signal d'alerte majeur)
  - Effets inversés par rapport au préventif pour FTG, LNG_log

## 3. SCORING DES TRONÇONS
- Score de risque basé sur P(abandon avant 50 ans) — Weibull AFT
- **19 475 tronçons top 10%** identifiés comme prioritaires
- Profil type top 10% : FTVI (60%), FT (31%), posés après 2010
- Le gradient décile 1→10 est net et cohérent

## 4. RECOMMANDATIONS OPÉRATIONNELLES

### Priorité haute
1. **FTVI** : 95% en décile 10, durée médiane 27 ans → surveillance renforcée immédiate
2. **PEHD** : concentration déciles 7-9, durée médiane 40 ans → planifier le renouvellement
3. **FT ancien** (pose <1960) : risque croissant avec l'âge, ρ=2.78

### Matériaux résilients (maintien en l'état)
- **BTM** : durée médiane 95 ans, 45% en décile 1
- **FTG** : durée médiane 87 ans (mais attention au risque correctif)
- **PVC/POLY** : durées médianes 74-75 ans

### Indicateurs d'alerte
- **Fuites détectées** : multiplicateur de risque correctif ×258
- **Taux d'anomalie/an** : facteur d'accélération le plus significatif
- **Longueur du tronçon** : HR=3.0 dans le Cox (tronçons longs = plus vulnérables)

### Stratégie
- Utiliser le scoring individuel (models/scoring_troncons.csv) pour prioriser les renouvellements
- Cibler en priorité les 19 475 tronçons du décile 10
- Adapter le plan pluriannuel selon les durées médianes par matériau

## 5. FICHIERS PRODUITS
- `models/cox_ph_summary_B.csv` — Résultats Cox PH
- `models/weibull_aft_summary_B.csv` — Résultats Weibull AFT
- `models/comparaison_modeles.csv` — AIC/BIC/C-index
- `models/cox_cause_specific_preventif.csv` — Cox cause préventive
- `models/cox_cause_specific_correctif.csv` — Cox cause corrective
- `models/comparaison_causes_HR.csv` — HR comparés
- `models/scoring_troncons.csv` — Score de risque par tronçon (194 745 lignes)
- `figures/etape5_*` — Figures Cox PH
- `figures/etape6_*` — Figures Weibull AFT
- `figures/etape7_*` — Figures risques compétitifs
- `figures/etape8_*` — Figures scoring
- `figures/etape9_*` — Planches de synthèse
