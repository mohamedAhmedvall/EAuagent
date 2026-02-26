# SOMEI — Système d'Aide à la Décision pour le Plan de Renouvellement du Réseau d'Eau Potable

> Analyse de survie · Scoring probabiliste · Optimisation MILP · API FastAPI · IHM Streamlit

---

## Sommaire

1. [Contexte](#1-contexte)
2. [Architecture générale](#2-architecture-générale)
3. [Modèles de survie](#3-modèles-de-survie)
4. [Score P_casse_1an — la métrique clé](#4-score-p_casse_1an--la-métrique-clé)
5. [Moteur d'optimisation MILP](#5-moteur-doptimisation-milp)
6. [API FastAPI](#6-api-fastapi)
7. [IHM Streamlit — les 6 pages](#7-ihm-streamlit--les-6-pages)
8. [Contraintes modélisées](#8-contraintes-modélisées)
9. [Démarrage rapide](#9-démarrage-rapide)
10. [Structure des fichiers](#10-structure-des-fichiers)

---

## 1. Contexte

**SOMEI** exploite un réseau d'eau potable de **~7 920 km** en Mauritanie, composé de **194 754 tronçons** de canalisation. La question centrale est : *quels tronçons renouveler, dans quel ordre, et sur quel horizon budgétaire ?*

Ce projet fournit une chaîne complète :

```
Données historiques            Modèle de survie           Décision
(194 754 tronçons,     →       Weibull AFT           →    Plan de
 31 152 abandons)              P_casse_1an                renouvellement MILP
                                                           + IHM Streamlit
```

---

## 2. Architecture générale

```
┌─────────────────────────────────────────────────────────────────┐
│                        IHM Streamlit                            │
│               ihm/app.py   (port 8501)                          │
│  Page 1: Dashboard   Page 2: Explorer   Page 3: Scorer          │
│  Page 4: Optimiser   Page 5: What-If    Page 6: Comparaison     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP REST
┌────────────────────────────▼────────────────────────────────────┐
│                        API FastAPI                              │
│               api/main.py  (port 8000)                          │
│  /stats  /troncons  /score  /optimiser  /whatif                 │
└──────────┬──────────────────────────────────────┬───────────────┘
           │                                      │
┌──────────▼──────────────┐          ┌────────────▼───────────────┐
│   Weibull AFT (lifelines)│          │   Moteur MILP (PuLP)       │
│   api/main.py /_score    │          │   api/optimizer.py          │
│   ρ = 2.78               │          │   Variables x[i,t] ∈ {0,1} │
└──────────┬──────────────┘          └────────────────────────────┘
           │
┌──────────▼──────────────┐
│   models/scoring_       │
│   troncons.csv          │
│   (194 745 lignes)      │
└─────────────────────────┘
```

---

## 3. Modèles de survie

### Données

| Indicateur | Valeur |
|---|---|
| Tronçons analysés | 194 754 |
| Abandons observés | 31 152 (16,0 %) |
| — dont préventifs | 27 653 |
| — dont correctifs | 3 499 |
| Tronçons encore en service | 163 602 (censurés à droite) |

Covariables : matériau (10 types), diamètre, longueur, année de pose, nb anomalies, nb fuites signalées/détectées, taux anomalie/an, logements desservis, flux de circulation.

---

### 3.1 Cox PH (modèle de référence, Dataset B)

- C-index = **0,586** (discrimination modeste, proportionnalité violée)
- Facteurs de risque : BTM (HR=12,7), PEHD (HR=12,0), longueur (HR=3,0), FTG (HR=2,2)
- Non retenu pour le scoring individuel en raison de la violation de l'hypothèse PH

---

### 3.2 Weibull AFT — **modèle retenu**

- C-index = **0,750** · AIC = 370 974 (meilleur parmi les modèles paramétriques)
- **ρ = 2,78** → risque croissant avec l'âge (vieillissement confirmé sur le réseau)

**Interprétation du paramètre de forme ρ :**

| ρ | Profil de risque |
|---|---|
| ρ < 1 | Risque décroissant avec l'âge (mortalité infantile) |
| ρ = 1 | Risque constant (processus de Poisson) |
| **ρ > 1** | **Risque croissant avec l'âge (vieillissement)** ← notre cas |

**Durées médianes prédites par matériau :**

| Matériau | Durée médiane | Urgence |
|---|---|---|
| FTVI (Fonte Ductile Vieille Italienne) | **27 ans** | Critique |
| PEHD (Polyéthylène HD) | 40 ans | Élevée |
| FT (Fonte) | 54 ans | Modérée |
| POLY (Polypropylène) | 74 ans | Faible |
| PVC | 75 ans | Faible |
| FTG (Fonte Galvanisée) | 87 ans | Faible |
| BTM (Béton) | 95 ans | Très faible |

---

### 3.3 Risques compétitifs — Fine-Gray / Cause-Specific Cox (Dataset A)

Distinction entre abandons **préventifs** et **correctifs** :

- **Préventif** (C-index=0,789) : les tronçons avec fuites/anomalies sont paradoxalement *moins* abandonnés préventivement → biais de surveillance (les équipes réparent au lieu de remplacer)
- **Correctif** : `nb_fuites_detectees` HR=**258** (signal d'alerte critique), effets de matériau inversés

---

### 3.4 Scoring des 194 745 tronçons

Fichier : `models/scoring_troncons.csv`

| Colonne | Description |
|---|---|
| `GID` | Identifiant unique du tronçon |
| `MAT_grp` | Matériau (FT, FTG, FTVI, PEHD, PVC, BTM, POLY, AC…) |
| `DIAMETRE_imp` | Diamètre (mm) |
| `LNG` | Longueur (mètres) |
| `DDP_year` | Année de pose |
| `duree_mediane_pred` | Durée médiane de survie prédite (ans) — sortie Weibull AFT |
| `risk_score_50ans` | P(abandon avant 50 ans) — scoré sur [0,1] |
| `P_survie_10ans` | P(encore en service à 10 ans) |
| `P_survie_20ans` | … |
| `P_survie_50ans` | … |
| `decile_risque` | Décile 1 (faible) → 10 (critique) |
| `top10_pourcent` | 1 si dans le top 10% des tronçons les plus risqués |

---

## 4. Score P_casse_1an — la métrique clé

### Définition

`P_casse_1an` est la **probabilité conditionnelle de casse dans la prochaine année**, sachant que le tronçon a survécu jusqu'à aujourd'hui.

C'est le **hazard discret sur 1 an** du modèle Weibull AFT.

### Formule

```
Weibull AFT :   S(t) = exp( -(t / λ)^ρ )

avec :
  λ = durée_médiane_pred / ln(2)^(1/ρ)    [paramètre d'échelle]
  ρ = 2,78                                  [paramètre de forme — ajusté sur les données]

P_casse_1an(âge) = 1 - S(âge + 1) / S(âge)
```

### Propriétés

- **Conditionnelle** : tient compte du fait que le tronçon a déjà survécu `âge` années
- **Dynamique** : augmente avec l'âge (car ρ = 2,78 > 1)
- **Interprétable** : une valeur de 3% signifie "ce tronçon a 3 chances sur 100 de casser dans la prochaine année"
- **Score ≠ risque_score_50ans** : risk_score_50ans est calculé une fois à la pose, P_casse_1an évolue chaque année

### Seuils d'interprétation

| P_casse_1an | Niveau | Action |
|---|---|---|
| ≥ 5 % | Critique | Renouvellement immédiat |
| ≥ 1 % | Élevé | Planifier dans l'année |
| ≥ 0,1 % | Modéré | Surveiller — inclure dans le plan à 5 ans |
| < 0,1 % | Faible | Maintien en l'état |

### Exemple numérique

Tronçon FTVI posé en 1990 (âge = 36 ans), durée médiane = 27 ans :

```
λ = 27 / ln(2)^(1/2.78) ≈ 27 / 0,776 ≈ 34,8

S(36) = exp(-(36/34.8)^2.78) = exp(-1.034^2.78) ≈ exp(-1.096) ≈ 0,334
S(37) = exp(-(37/34.8)^2.78) ≈ 0,302

P_casse_1an = 1 - 0,302 / 0,334 ≈ 9,6%  → CRITIQUE
```

### Utilisations du score P_casse_1an

1. **Dashboard** : KPIs réseau (nb tronçons P≥1%, P≥5%)
2. **Explorer** : tri et filtre des tronçons par urgence annuelle
3. **Scorer** : évaluation ad hoc + projection dans le temps
4. **Optimisation** : objectif du MILP = maximiser Σ P_casse_1an évitée

---

## 5. Moteur d'optimisation MILP

### Problème posé

Sélectionner quels tronçons renouveler et en quelle année, de façon à **maximiser le risque évité** tout en respectant les contraintes opérationnelles et réglementaires.

### Formulation mathématique

**Variables de décision :**

```
x[i,t] ∈ {0, 1}    pour i ∈ {0,…,n-1}, t ∈ {0,…,T-1}

x[i,t] = 1  →  le tronçon i est renouvelé durant l'année t
```

**Fonction objectif (bénéfice cumulatif) :**

```
maximiser  Σᵢ Σₜ  x[i,t] · benefit_cum[i,t]

avec :
  benefit_cum[i,t] = Σ_{s=t}^{T-1} P_casse_1an(âge_i + s)
```

Le bénéfice est **cumulatif** : renouveler le tronçon i en année t évite toutes les casses de t à T-1. Ainsi `benefit_cum[i,0] > benefit_cum[i,1] > …`, ce qui crée une **incitation naturelle au renouvellement précoce**.

**Contraintes :**

| # | Contrainte | Formule |
|---|---|---|
| C1 | Unicité | `Σₜ x[i,t] ≤ 1` pour tout i — renouveler au plus une fois |
| C2 | Urgences | `Σₜ x[i,t] = 1` pour les urgences (FTVI, AC, fuites, vieux) |
| C3 | Budget max | `Σᵢ coût[i] · x[i,t] ≤ budget_max` pour tout t |
| C4 | Budget min | `Σᵢ coût[i] · x[i,t] ≥ budget_min_eff` pour tout t |
| C5 | Km max | `Σᵢ lng[i] · x[i,t] ≤ km_max` pour tout t |
| C6 | Km min (loi 1%) | `Σᵢ lng[i] · x[i,t] ≥ km_min_eff` pour tout t |
| C7 | Lissage budget | `budget[t+1] ≤ budget[t] · (1 + lissage_pct)` |

**Robustesse des contraintes min :**

Les contraintes minimum sont auto-adaptées au sous-ensemble sélectionné pour éviter l'infaisabilité systématique :

```python
# km_min ne peut pas exiger plus que ce qui est disponible / T années
km_min_effectif = min(km_min_cible, km_max * 0.9, km_disponible_total / T * 0.9)

# budget_min adapté à la taille du sous-ensemble
budget_min_effectif = min(budget_annuel_min, budget_disponible_total / T * 0.8)
```

**Résolution :** PuLP (interface Python MILP) avec CBC (solveur open-source). Limite : 60 secondes par défaut.

### Score de priorité composite

Pour sélectionner le sous-ensemble top-N avant le MILP :

```
priorite_score = 0.5 × P_casse_1an
               + 0.3 × (decile_risque / 10)
               + 0.2 × urgence_flag
```

### What-If

Exploration paramétrique sans MILP (algorithme glouton rapide) : fait varier 1 ou 2 paramètres (budget, km, horizon…) et calcule le résultat pour chaque combinaison. Recommande le scénario optimal.

---

## 6. API FastAPI

### Démarrage

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Santé de l'API + version |
| `GET` | `/stats` | Statistiques globales du réseau (194k tronçons) |
| `GET` | `/troncons` | Liste paginée des tronçons scorés |
| `GET` | `/troncons/{gid}` | Détail d'un tronçon par GID |
| `POST` | `/score` | Scorer un tronçon ad hoc |
| `POST` | `/optimiser` | Plan de renouvellement MILP |
| `POST` | `/whatif` | Analyse de sensibilité paramétrique |
| `GET` | `/contraintes/defaut` | Valeurs par défaut des contraintes |

---

### `POST /score` — Scorer un tronçon

**Entrée :**
```json
{
  "MAT_grp": "FTVI",
  "DIAMETRE_imp": 100,
  "LNG": 120.5,
  "DDP_year": 1992,
  "nb_anomalies": 2,
  "nb_fuites_signalees": 1,
  "nb_fuites_detectees": 1,
  "taux_anomalie_par_an": 0.05
}
```

**Sortie :**
```json
{
  "duree_mediane_pred": 27.3,
  "P_casse_1an": 0.0862,
  "risk_score_50ans": 0.978,
  "P_survie_10ans": 0.621,
  "P_survie_20ans": 0.312,
  "P_survie_30ans": 0.089,
  "P_survie_50ans": 0.002,
  "P_survie_70ans": 0.0,
  "decile_risque": 10,
  "top10_pourcent": true,
  "interpretation": "CRITIQUE — P(casse cette année) = 8.62% — Renouvellement immédiat recommandé"
}
```

---

### `POST /optimiser` — Plan MILP

**Entrée :**
```json
{
  "contraintes": {
    "budget_annuel_max": 500000000,
    "budget_annuel_min": 50000000,
    "km_max_par_an": 80,
    "km_min_par_an": 10,
    "taux_renouvellement_min_pct": 1.0,
    "horizon_plan": 5,
    "annee_debut": 2026,
    "materiaux_urgence": ["FTVI", "AC"],
    "lissage_budget_pct": 0.30
  },
  "top_n_troncons": 5000,
  "objectif": "maximiser_reduction_risque"
}
```

**Sortie :**
```json
{
  "statut": "OK",
  "message": "Plan optimal trouvé — 3 847 tronçons planifiés sur 5 ans",
  "resume_global": {
    "nb_troncons_planifies": 3847,
    "km_total_renouveles": 284.2,
    "budget_total_engage": 2143000000,
    "p_casse_1an_evitee": 98.43,
    "risque_residuel_pct": 31.2
  },
  "resume_par_annee": [
    {"annee": 2026, "nb_troncons": 1204, "km_renouveles": 71.3, "budget_engage": 502000000},
    {"annee": 2027, "nb_troncons": 856,  "km_renouveles": 58.1, "budget_engage": 437000000},
    ...
  ],
  "plan_detaille": [
    {"GID": 12345, "annee_prevue": 2026, "MAT_grp": "FTVI",
     "LNG_km": 0.12, "cout_estime": 1020000,
     "raison_priorite": "matériau urgence (FTVI) | décile risque 10/10"}
  ]
}
```

---

### `POST /whatif` — Analyse de sensibilité

**Entrée :**
```json
{
  "parametres_variables": [
    {"nom": "budget_annuel_max", "valeurs": [200000000, 350000000, 500000000, 700000000]}
  ],
  "top_n_troncons": 3000
}
```

**Sortie :** liste de scénarios avec km renouvelés, risque résiduel, recommandation du meilleur scénario.

---

## 7. IHM Streamlit — les 6 pages

### Démarrage

```bash
streamlit run ihm/app.py
```

Interface : [http://localhost:8501](http://localhost:8501)

---

### Page 1 — 📊 Tableau de bord réseau

Vue d'ensemble du réseau :
- KPIs : nb tronçons, km total, km min légal (1%/an), top 10% risque, fuites actives
- Section P_casse_1an : P moyen/médian, nb tronçons P≥1%, nb P≥5%
- Distribution des déciles de risque (bar chart)
- Répartition par matériau (donut)
- Score par matériau (tableau)
- Distribution des âges (histogramme)
- Heatmap Matériau × Décile de risque

---

### Page 2 — 🔍 Explorer les tronçons

Navigation filtrable dans les 194 745 tronçons :
- Filtres : décile minimum, matériau, top 10%, fuites actives
- Tri automatique par P_casse_1an (urgence annuelle)
- Tableau avec dégradé couleur sur P(casse/an)
- Scatter P_casse_1an vs âge, coloré par matériau
- Export CSV de la sélection

---

### Page 3 — 🎯 Scorer un tronçon

Évaluation ad hoc d'un tronçon via formulaire :
- Entrées : matériau, diamètre, longueur, année de pose, anomalies, fuites
- Appel `POST /score` → retour instantané
- P(casse cette année) affiché en métrique principale avec badge couleur
- Courbe de survie S(t) sur 70 ans
- Graphique P_casse_1an projetée aux âges futurs (si non renouvelé, la probabilité monte)

---

### Page 4 — ⚙️ Optimisation du plan

Génération du plan pluriannuel MILP :

- **Sélecteur d'horizon** proéminent : 1 / 3 / 5 / 10 ans
- Contraintes configurables : budget max/min, km max/min, lissage, décile prioritaire, matériaux urgence, taux réglementaire
- Objectif : maximiser réduction de risque / minimiser coût / équilibre
- Résultats :
  - KPIs globaux : tronçons, km, budget, P_casse évitée, risque résiduel
  - Graphique km + budget par année
  - Tableau plan annuel
  - Tableau détaillé par tronçon avec `raison_priorite`
  - Export CSV

---

### Page 5 — 🔄 Analyse What-If

Exploration paramétrique :
- **Scénarios prédéfinis** : Impact budget, Budget vs Km, Urgences FTVI, Horizon 3/5/10 ans
- **Paramétrage manuel** : varier 1 à 2 paramètres avec des listes de valeurs
- Tableau comparatif multi-scénarios (surlignage du meilleur)
- Courbe de sensibilité (1 paramètre) ou heatmap (2 paramètres)
- Recommandation automatique du scénario optimal

---

### Page 6 — 🧠 Comparaison & Explicabilité

**A. Benchmark 3 stratégies** (même enveloppe budget/km totale) :

| Stratégie | Description |
|---|---|
| 🎲 Aléatoire | Moyenne de 10 tirages aléatoires respectant le budget/km |
| 📋 Glouton | Tri par P_casse_1an décroissant, sélection séquentielle |
| ⚡ MILP | Optimiseur mathématique (solution optimale sous contraintes annuelles) |

→ Cartes colorées + bar chart avec annotation du gain MILP vs baselines
→ Tableau synthèse : tronçons, km, budget, P évitée, **coût par casse évitée**

**B. KPIs enrichis du plan MILP :**
- Casses/an évitées (= Σ P_casse_1an, interprétable comme un nombre de ruptures)
- Coût par casse évitée (M MAD)
- % FTVI et % AC planifiés vs total réseau
- P_casse_1an moyen plan vs réseau
- Âge moyen planifié vs réseau
- Répartition matériaux plan vs réseau (grouped bar)
- Histogramme P_casse_1an plan vs réseau (overlay — le plan doit décaler vers la droite)

**C. Frontière Pareto :**
Tronçons triés par `P_casse_1an / coût` (efficience marginale), courbe cumulative km → P_évitée montrant le rendement décroissant. Positions MILP (⭐), Glouton (◆) et Aléatoire (●) sur la courbe.

**D. Explicabilité par tronçon :**
Pour chaque tronçon planifié, stacked bar horizontal montrant la contribution de 4 facteurs (normalisés 0–1) :

| Facteur | Ce qu'il mesure |
|---|---|
| P(casse/an) — urgence actuelle | Probabilité de casse cette année |
| Âge / durée médiane | Rapport âge actuel / espérance de vie (usure relative) |
| Efficience (P_casse / M MAD) | Rendement du renouvellement — risque évité par euro dépensé |
| Matériau urgence (FTVI/AC) | Flag matériau critique |

→ Tableau détaillé avec scores + `raison_priorite`
→ Export CSV annoté

---

## 8. Contraintes modélisées

Voir le fichier `CONTRAINTES_SOMEI.md` pour le détail complet.

### Synthèse des 7 catégories

| Catégorie | Contraintes clés | Type |
|---|---|---|
| Financière | Budget annuel max/min, lissage ±30% | Dure / Souple |
| Capacité opérationnelle | Km max/an, chantiers simultanés | Dure |
| Priorisation | Décile ≥7, âge max 60 ans, fuites | Souple / Dure |
| Réglementaire | **1%/an obligatoire** (~79 km/an) | Dure |
| Continuité de service | Zones sans coupure, hôpitaux | Dure |
| Coordination urbaine | Voirie, SOMELEC, assainissement | Souple |
| Matériaux | FTVI/AC urgence absolue | Dure |

### Coûts de renouvellement par matériau

| Matériau | Coût (MAD/km) |
|---|---|
| AC | 9 000 000 |
| FTVI | 8 500 000 |
| FT | 8 000 000 |
| BTM | 7 000 000 |
| FTG | 7 500 000 |
| POLY | 6 500 000 |
| PEHD | 6 000 000 |
| PVC | 5 500 000 |

---

## 9. Démarrage rapide

### Prérequis

```bash
pip install -r requirements.txt
```

Dépendances principales : `fastapi`, `uvicorn`, `streamlit`, `lifelines`, `pulp`, `pandas`, `numpy`, `plotly`.

### Lancement complet

**Terminal 1 — API :**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — IHM :**
```bash
streamlit run ihm/app.py
```

**IHM** → [http://localhost:8501](http://localhost:8501)
**Docs API** → [http://localhost:8000/docs](http://localhost:8000/docs)

### Exemple d'appel direct à l'API

```bash
# Score d'un tronçon FTVI posé en 1992
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "MAT_grp": "FTVI",
    "DIAMETRE_imp": 100,
    "LNG": 120,
    "DDP_year": 1992,
    "nb_fuites_detectees": 1
  }'

# Plan optimal 5 ans, top 5000 tronçons
curl -X POST http://localhost:8000/optimiser \
  -H "Content-Type: application/json" \
  -d '{
    "contraintes": {"horizon_plan": 5, "budget_annuel_max": 500000000},
    "top_n_troncons": 5000,
    "objectif": "maximiser_reduction_risque"
  }'
```

---

## 10. Structure des fichiers

```
EAuagent/
│
├── README.md                       ← ce fichier
├── CONTRAINTES_SOMEI.md            ← catalogue des 7 catégories de contraintes
├── resultats.md                    ← rapport de synthèse des modèles de survie
├── requirements.txt                ← dépendances Python
├── .gitignore
│
├── api/                            ← Backend FastAPI
│   ├── __init__.py
│   ├── main.py                     ← endpoints REST
│   ├── models.py                   ← schémas Pydantic (entrées/sorties)
│   └── optimizer.py                ← moteur MILP (PuLP) + what-if
│
├── ihm/                            ← IHM Streamlit
│   ├── __init__.py
│   └── app.py                      ← 6 pages (1300 lignes)
│
├── models/                         ← Données et résultats des modèles
│   ├── scoring_troncons.csv        ← 194 745 tronçons scorés (Weibull AFT)
│   ├── weibull_aft_summary_B.csv   ← coefficients du modèle retenu
│   ├── cox_ph_summary_B.csv        ← coefficients Cox PH
│   ├── cox_cause_specific_*.csv    ← modèles cause-specific
│   ├── comparaison_modeles.csv     ← AIC/BIC/C-index comparés
│   └── comparaison_causes_HR.csv   ← HR préventif vs correctif
│
├── data/                           ← Données brutes (non versionnées)
│   └── dataset_B_simple.csv
│
├── figures/                        ← Figures générées par les étapes
│   └── etape{5..9}_*               ← Cox PH, Weibull, Scoring, Synthèse
│
├── etape5_cox.py                   ← Ajustement Cox PH
├── etape6_weibull.py               ← Ajustement Weibull AFT (modèle retenu)
├── etape7_finegray.py              ← Risques compétitifs Fine-Gray
├── etape8_scoring.py               ← Scoring des 194k tronçons
├── etape9_rapport.py               ← Rapport et planches de synthèse
└── audit_metriques.py              ← Validation croisée, calibration
```

---

## Performances connues du système

| Indicateur | Valeur |
|---|---|
| Modèle Weibull AFT — C-index | **0,750** |
| Paramètre de forme ρ | 2,78 (risque croissant) |
| Taille du réseau modélisé | 194 745 tronçons · 7 920 km |
| Temps MILP (top-1000, horizon 5 ans) | ~30 secondes |
| Temps MILP (top-5000, horizon 5 ans) | ~60 secondes |
| Gain MILP vs glouton (P_casse évitée) | +5–20% selon le sous-ensemble |
| Gain MILP vs aléatoire | +40–80% |

---

*Développé pour la SOMEI — Plan de Renouvellement Réseau Eau Potable Mauritanie — 2026*
