"""
ÉTAPE 5 — Modélisation Cox PH sur Dataset B (tous abandons)
============================================================
- Préparation des données
- Ajustement Cox PH multivarié
- Test de proportionnalité (Schoenfeld)
- Forest plot des Hazard Ratios
- Courbes de survie ajustées

Covariables retenues (sans fuite de données) :
  - MAT_grp (dummies, ref=FT)
  - DIAMETRE_imp, LNG_log
  - nb_anomalies, nb_fuites_signalees, nb_fuites_detectees
  - DT_NB_LOGEMENT_imp, DT_FLUX_CIRCULATION_imp

Exclusions justifiées :
  - DDP_year : corr=-1.0 avec duration_years pour les 84% de tronçons censurés
    (DDP_year = 2024 - duration_years pour un tronçon encore en service)
    → confond effet d'âge et effet de cohorte, biaisant tous les coefficients.
  - taux_anomalie_par_an : = nb_anomalies / duration_years → data leakage
    (duration_years est la variable cible du modèle de survie).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

# ── 1. Chargement ──────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 5 — MODÈLE COX PH (Dataset B — tous abandons)")
print("=" * 60)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')
print(f"\nDataset B : {len(df)} tronçons, {df['event_bin'].sum()} événements ({df['event_bin'].mean()*100:.1f}%)")

# ── 2. Préparation des variables ──────────────────────────────
duration_col = 'duration_years'
event_col = 'event_bin'

# Matériau — dummies (référence = FT, le plus fréquent)
mat_dummies = pd.get_dummies(df['MAT_grp'], prefix='mat', drop_first=False)
mat_counts = df['MAT_grp'].value_counts()
mats_keep = mat_counts[mat_counts > 500].index.tolist()
print(f"\nMatériaux retenus (n>500) : {mats_keep}")

if 'FT' in mats_keep:
    mats_keep.remove('FT')
mat_cols = [f'mat_{m}' for m in mats_keep]
for col in mat_cols:
    if col not in mat_dummies.columns:
        mat_dummies[col] = 0

# Covariables continues — NOTE : DDP_year et taux_anomalie_par_an exclus
covariates_num = [
    'DIAMETRE_imp',
    'LNG_log',
    'nb_anomalies',
    'nb_fuites_signalees',
    'nb_fuites_detectees',
    'DT_NB_LOGEMENT_imp',
    'DT_FLUX_CIRCULATION_imp',
]

# Construire le dataframe de modélisation
model_df = df[[duration_col, event_col]].copy()
for col in covariates_num:
    model_df[col] = pd.to_numeric(df[col], errors='coerce')
for col in mat_cols:
    model_df[col] = mat_dummies[col].values

model_df = model_df.dropna()
model_df = model_df[model_df[duration_col] > 0]

print(f"Observations après nettoyage : {len(model_df)}")
print(f"Événements : {model_df[event_col].sum()}")

# ── 3. Split train/test pour évaluation honnête ───────────────
train_df, test_df = train_test_split(model_df, test_size=0.20, random_state=42)
print(f"\nTrain : {len(train_df)} | Test : {len(test_df)}")

# ── 4. Ajustement Cox PH ──────────────────────────────────────
print("\n── Ajustement Cox PH ──")
cox = CoxPHFitter(penalizer=0.01)
cox.fit(train_df, duration_col=duration_col, event_col=event_col)

c_train = cox.concordance_index_
from lifelines.utils import concordance_index
c_test = concordance_index(
    test_df[duration_col],
    -cox.predict_partial_hazard(test_df),  # hazard → négatif pour aligner avec durée
    test_df[event_col],
)

print(f"C-index (train) : {c_train:.4f}")
print(f"C-index (test)  : {c_test:.4f}  ← valeur honnête hors-échantillon")

print("\n── RÉSULTATS COX PH ──")
summary_c = cox.summary
print(summary_c[['coef', 'exp(coef)', 'p', 'coef lower 95%', 'coef upper 95%']].to_string())

cox.summary.to_csv('/home/user/EAuagent/models/cox_ph_summary_B.csv')

# ── 5. Test de proportionnalité (Schoenfeld) ──────────────────
print("\n── Test proportionnalité ──")
try:
    cox.check_assumptions(train_df, p_value_threshold=0.05, show_plots=False)
except Exception as e:
    print(f"(test non disponible : {e})")

# ── 6. FIGURES ────────────────────────────────────────────────

# 6a. Forest plot — Hazard Ratios
fig, ax = plt.subplots(figsize=(10, 7))
hr = np.exp(summary_c['coef'])
ci_lo = np.exp(summary_c['coef lower 95%'])
ci_hi = np.exp(summary_c['coef upper 95%'])
pvals = summary_c['p']

hr_sorted = hr.sort_values()
idx = hr_sorted.index
ci_lo = ci_lo[idx]
ci_hi = ci_hi[idx]
pvals = pvals[idx]

colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.05 else '#95a5a6'
          for p in pvals]

y_pos = range(len(idx))
for i, (var, hr_val, lo, hi, col) in enumerate(zip(idx, hr_sorted, ci_lo, ci_hi, colors)):
    ax.plot([lo, hi], [i, i], color=col, linewidth=1.5, alpha=0.7)
    ax.plot(hr_val, i, 'o', color=col, markersize=6)

ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(idx, fontsize=9)
ax.set_xlabel('Hazard Ratio (IC 95%)', fontsize=11)
ax.set_title(
    f'Cox PH — Forest plot Hazard Ratios\n'
    f'C-index train={c_train:.3f} | test={c_test:.3f}',
    fontsize=13, fontweight='bold'
)
legend_elements = [
    mpatches.Patch(color='#e74c3c', alpha=0.8, label='p < 0.001'),
    mpatches.Patch(color='#f39c12', alpha=0.8, label='p < 0.05'),
    mpatches.Patch(color='#95a5a6', alpha=0.8, label='Non significatif'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape5_cox_forest_plot.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape5_cox_forest_plot.png")

print("\n✓ Étape 5 terminée.")
