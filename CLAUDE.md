# EAuagent — Prédiction de casse de canalisations d'eau

## Contexte projet

Projet de prédiction de casse (abandon correctif) de canalisations pour le réseau d'eau SOMEI.
L'objectif final est de combiner modèle prédictif + optimisation OR-Tools pour planifier le renouvellement.

## État d'avancement

### TERMINÉ
- **Étape 1 — Nettoyage des données** : fait, dataset nettoyé sauvegardé dans `data/dataset_clean.csv`
- **Étape 2 — Variables cibles** : `failure_1y`, `failure_3y`, `failure_5y` créées dans `data/dataset_model.csv`

### À FAIRE
- **Étape 3 — Entraînement des modèles** : script prêt dans `train_models.py`, pas encore exécuté
  - 6 modèles : LogReg, RandomForest, XGBoost, MLP (chacun balanced + SMOTE pour LR/XGB)
  - 3 horizons : 1 an, 3 ans, 5 ans
  - lifelines (Cox PH, Random Survival Forest) n'a pas pu s'installer → trouver alternative ou fixer
- **Étape 4 — Évaluation** : métriques codées dans le script (ROC-AUC, PR-AUC, F1, Brier, matrice confusion)
- **Étape 5 (future)** — Optimisation OR-Tools : contrainte budget, choix tronçons, abandon préventif opportuniste
- **Étape 6 (future)** — IHM Streamlit avec onglet paramétrage (coûts, seuils, etc.)

## Fichiers importants

| Fichier | Description |
|---|---|
| `data/dataset_A_competitif.csv` | Dataset original A (multi-classe : préventif vs correctif) |
| `data/dataset_B_simple.csv` | Dataset original B (binaire simple) |
| `data/dataset_C_correctif.csv` | Dataset original C (correctif = casse uniquement) — **base de travail** |
| `data/dataset_clean.csv` | Dataset nettoyé (157 192 lignes, 11 colonnes, 0 nulls) |
| `data/dataset_model.csv` | Dataset prêt pour modélisation (134 459 lignes, features + 3 targets) |
| `train_models.py` | Script d'entraînement (6 modèles × 3 horizons) — **à exécuter** |

## Nettoyage appliqué

1. Colonnes DT supprimées (`DT_NB_LOGEMENT_imp`, `DT_NB_ABONNE_imp`, `DT_FLUX_CIRCULATION_imp`, `dt_missing_flag`)
2. Colonnes leakage supprimées (`STATUT_OBJET`, `abandon_type`)
3. Colonnes redondantes supprimées (`duration_days`, `LNG_log`, `MAT`, `diam_imputed_flag`, `decade_pose`, `a_anomalie`, `event`)
4. `jours_depuis_derniere_anomalie` supprimée (93.9% NaN)
5. `duration_years <= 0` supprimées (7 lignes)
6. `duration_years > 120` supprimées (3 lignes)
7. `DDP_year == 1949` supprimées (16 264 — date par défaut suspecte)
8. `DIAMETRE_imp > 1500mm` supprimées (29 — réseau transport)
9. `DIAMETRE_imp NaN` supprimées (2)
10. **Abandons préventifs supprimés** (21 257 — non prédictibles, décisions opportunistes)

## Variables cibles (Étape 2)

- Split temporel : CUTOFF = 2020, seuls les tronçons posés avant 2020 et encore en service au 01/01/2020
- `failure_1y` : casse dans les 12 mois → 184 positifs (0.14%)
- `failure_3y` : casse dans les 3 ans → 455 positifs (0.34%)
- `failure_5y` : casse dans les 5 ans → 681 positifs (0.51%)
- Très déséquilibré → class_weight='balanced' et SMOTE testés

## Dockerisation (REQUIS pour exécution locale)

Pour exécuter les scripts Python localement, il faut dockeriser le projet.

### Dockerfile recommandé

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    pandas numpy scikit-learn xgboost imbalanced-learn \
    lifelines scikit-survival \
    streamlit fastapi uvicorn ortools

EXPOSE 8501 8000

CMD ["bash"]
```

### Lancer

```bash
docker build -t eauagent .
docker run -it -v $(pwd):/app -p 8501:8501 -p 8000:8000 eauagent
# Puis dans le conteneur :
python train_models.py
```

## Décisions de conception

- **Cible = abandon correctif uniquement** (casse réelle). Les abandons préventifs sont des décisions opportunistes (chantier voisin, moins cher de remplacer) et ne sont pas prédictibles.
- **Split temporel** (pas aléatoire) pour éviter le data leakage. Train = tronçons posés avant 2000, Test = posés 2000+.
- `event_corr` du dataset C est la cible de référence.
- Les features anomalies (`nb_anomalies`, `nb_fuites_*`, `taux_anomalie_par_an`) sont conservées mais attention : elles couvrent la période complète et peuvent contenir des infos post-cutoff. Limitation acceptée faute de dates individuelles.
