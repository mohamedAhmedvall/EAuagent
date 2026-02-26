"""
ÉTAPE 5 — Modélisation Cox PH sur Dataset B (tous abandons)
============================================================
- Préparation des données
- Ajustement Cox PH multivarié
- Test de proportionnalité (Schoenfeld)
- Forest plot des Hazard Ratios
- Courbes de survie ajustées
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

# ── 1. Chargement ──────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 5 — MODÈLE COX PH (Dataset B — tous abandons)")
print("=" * 60)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')
print(f"\nDataset B : {len(df)} tronçons, {df['event_bin'].sum()} événements ({df['event_bin'].mean()*100:.1f}%)")

# ── 2. Préparation des variables ──────────────────────────────
# Variable cible
duration_col = 'duration_years'
event_col = 'event_bin'

# Matériau — dummies (référence = FT, le plus fréquent)
mat_dummies = pd.get_dummies(df['MAT_grp'], prefix='mat', drop_first=False)
# Garder les matériaux avec effectif suffisant (>500)
mat_counts = df['MAT_grp'].value_counts()
mats_keep = mat_counts[mat_counts > 500].index.tolist()
print(f"\nMatériaux retenus (n>500) : {mats_keep}")

# Référence = FT
if 'FT' in mats_keep:
    mats_keep.remove('FT')
mat_cols = [f'mat_{m}' for m in mats_keep]
for col in mat_cols:
    if col not in mat_dummies.columns:
        mat_dummies[col] = 0

# Covariables continues
covariates_num = [
    'DIAMETRE_imp',
    'LNG_log',
    'DDP_year',
    'nb_anomalies',
    'nb_fuites_signalees',
    'nb_fuites_detectees',
    'taux_anomalie_par_an',
    'DT_NB_LOGEMENT_imp',
    'DT_FLUX_CIRCULATION_imp',
]

# Construire le dataframe de modélisation
model_df = df[[duration_col, event_col]].copy()

# Ajouter les covariables numériques
for col in covariates_num:
    model_df[col] = pd.to_numeric(df[col], errors='coerce')

# Ajouter les dummies matériau
for col in mat_cols:
    model_df[col] = mat_dummies[col].values

# Supprimer les lignes avec NA ou durée <= 0
model_df = model_df.dropna()
model_df = model_df[model_df[duration_col] > 0]

print(f"Observations après nettoyage : {len(model_df)}")
print(f"Événements : {model_df[event_col].sum()}")

# ── 3. Ajustement Cox PH ──────────────────────────────────────
print("\n── Ajustement du modèle Cox PH ──")
cph = CoxPHFitter(penalizer=0.01)
cph.fit(
    model_df,
    duration_col=duration_col,
    event_col=event_col,
    show_progress=False
)

# Résumé
print("\n── RÉSULTATS COX PH ──")
summary = cph.summary
print(summary.to_string())

# Concordance
print(f"\nIndice de concordance (C-index) : {cph.concordance_index_:.4f}")

# ── 4. Test de proportionnalité (Schoenfeld) ──────────────────
print("\n── Test de proportionnalité des risques (Schoenfeld) ──")
try:
    ph_test = cph.check_assumptions(model_df, p_value_threshold=0.05, show_plots=False)
    print("Test de proportionnalité terminé.")
except Exception as e:
    print(f"Note : {e}")

# ── 5. Sauvegarde des résultats ───────────────────────────────
summary.to_csv('/home/user/EAuagent/models/cox_ph_summary_B.csv')
print("\nRésumé sauvegardé dans models/cox_ph_summary_B.csv")

# ── 6. FIGURES ────────────────────────────────────────────────

# 6a. Forest plot des Hazard Ratios
fig, ax = plt.subplots(figsize=(10, 8))

# Trier par HR
hr = summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
hr.columns = ['HR', 'HR_lo', 'HR_hi', 'p']
hr = hr.sort_values('HR')

# Couleurs selon significativité
colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.05 else '#95a5a6' for p in hr['p']]

y_pos = range(len(hr))
ax.barh(y_pos, hr['HR'] - 1, left=1, color=colors, alpha=0.7, height=0.6)
ax.errorbar(hr['HR'], y_pos, xerr=[hr['HR'] - hr['HR_lo'], hr['HR_hi'] - hr['HR']],
            fmt='o', color='black', markersize=5, capsize=3)
ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(hr.index, fontsize=9)
ax.set_xlabel('Hazard Ratio (HR)', fontsize=11)
ax.set_title('Cox PH — Forest Plot des Hazard Ratios\n(Dataset B — tous abandons, réf. = FT)',
             fontsize=13, fontweight='bold')

# Légende
legend_elements = [
    mpatches.Patch(color='#e74c3c', alpha=0.7, label='p < 0.001'),
    mpatches.Patch(color='#f39c12', alpha=0.7, label='p < 0.05'),
    mpatches.Patch(color='#95a5a6', alpha=0.7, label='Non significatif'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape5_forest_plot_cox.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape5_forest_plot_cox.png")

# 6b. Courbes de survie ajustées par matériau (profils types)
fig, ax = plt.subplots(figsize=(10, 6))

# Créer des profils types pour chaque matériau principal
materials_plot = ['FT'] + mats_keep[:6]
median_vals = model_df[covariates_num].median()

colors_mat = plt.cm.Set1(np.linspace(0, 1, len(materials_plot)))

for i, mat in enumerate(materials_plot):
    # Profil type : valeurs médianes pour tout, sauf le matériau
    profile = pd.DataFrame([median_vals], columns=covariates_num)
    for mc in mat_cols:
        profile[mc] = 0
    if mat != 'FT' and f'mat_{mat}' in mat_cols:
        profile[f'mat_{mat}'] = 1

    cph.plot_partial_effects_on_outcome(
        covariates=mat_cols[0] if mat_cols else covariates_num[0],
        values=[0],
        ax=ax,
        plot_baseline=False
    )

# Approche alternative : courbes prédites
plt.close()
fig, ax = plt.subplots(figsize=(10, 6))

for i, mat in enumerate(materials_plot):
    # Filtrer les données pour ce matériau
    if mat == 'FT':
        mask = True
        for mc in mat_cols:
            mask = mask & (model_df[mc] == 0)
        # Vérifier que c'est bien FT (aucun dummy matériau à 1)
    else:
        if f'mat_{mat}' in model_df.columns:
            mask = model_df[f'mat_{mat}'] == 1
        else:
            continue

    subset = model_df[mask]
    if len(subset) < 100:
        continue

    sf = cph.predict_survival_function(subset.sample(min(500, len(subset)), random_state=42))
    sf_mean = sf.mean(axis=1)
    ax.plot(sf_mean.index, sf_mean.values, label=f'{mat} (n={len(subset)})',
            color=colors_mat[i], linewidth=2)

ax.set_xlabel('Durée (années)', fontsize=11)
ax.set_ylabel('S(t) — Probabilité de survie', fontsize=11)
ax.set_title('Cox PH — Courbes de survie ajustées par matériau\n(profil médian, Dataset B)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape5_cox_survie_materiau.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape5_cox_survie_materiau.png")

# 6c. Tableau récapitulatif des HR significatifs
sig = hr[hr['p'] < 0.05].copy()
sig['stars'] = sig['p'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*')
sig['HR_fmt'] = sig.apply(lambda r: f"{r['HR']:.3f} [{r['HR_lo']:.3f}-{r['HR_hi']:.3f}]", axis=1)
print("\n── HAZARD RATIOS SIGNIFICATIFS ──")
print(sig[['HR_fmt', 'p', 'stars']].to_string())

print("\n✓ Étape 5 terminée.")
