"""
ÉTAPE 7 — Modèle Fine-Gray (risques compétitifs) sur Dataset A
===============================================================
- Cause 1 = abandon préventif, Cause 2 = abandon correctif
- Incidence cumulée par covariable
- Modélisation sous-distribution de risque
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
print("ÉTAPE 7 — FINE-GRAY / RISQUES COMPÉTITIFS (Dataset A)")
print("=" * 60)

df = pd.read_csv('/home/user/EAuagent/data/dataset_A_competitif.csv')
print(f"Dataset A : {len(df)} tronçons")
print(f"  event_code=0 (censuré)  : {(df['event_code']==0).sum()}")
print(f"  event_code=1 (préventif): {(df['event_code']==1).sum()}")
print(f"  event_code=2 (correctif): {(df['event_code']==2).sum()}")

# ── 2. Approche Fine-Gray via cause-specific Cox ──────────────
# lifelines n'a pas de Fine-Gray natif, on utilise l'approche
# cause-specific : un Cox PH par cause, censurant l'autre cause

duration_col = 'duration_years'

mat_counts = df['MAT_grp'].value_counts()
mats_keep = mat_counts[mat_counts > 500].index.tolist()
if 'FT' in mats_keep:
    mats_keep.remove('FT')
mat_dummies = pd.get_dummies(df['MAT_grp'], prefix='mat', drop_first=False)
mat_cols = [f'mat_{m}' for m in mats_keep]

covariates_num = [
    'DIAMETRE_imp', 'LNG_log', 'DDP_year',
    'nb_anomalies', 'nb_fuites_signalees', 'nb_fuites_detectees',
    'taux_anomalie_par_an',
    'DT_NB_LOGEMENT_imp', 'DT_FLUX_CIRCULATION_imp',
]

# Construire le df de modélisation
model_df = df[[duration_col, 'event_code']].copy()
for col in covariates_num:
    model_df[col] = pd.to_numeric(df[col], errors='coerce')
for col in mat_cols:
    if col in mat_dummies.columns:
        model_df[col] = mat_dummies[col].values
    else:
        model_df[col] = 0

model_df = model_df.dropna()
model_df = model_df[model_df[duration_col] > 0]

# ── 3. Modèle cause-specific : PRÉVENTIF (cause 1) ───────────
print("\n" + "=" * 55)
print("MODÈLE CAUSE-SPECIFIC : ABANDON PRÉVENTIF (cause 1)")
print("=" * 55)

df_prev = model_df.copy()
# Événement = 1 si préventif, 0 sinon (censuré ou correctif)
df_prev['event'] = (df_prev['event_code'] == 1).astype(int)
df_prev = df_prev.drop(columns=['event_code'])

cph_prev = CoxPHFitter(penalizer=0.01)
cph_prev.fit(df_prev, duration_col=duration_col, event_col='event', show_progress=False)

print(f"C-index préventif : {cph_prev.concordance_index_:.4f}")
print("\nHazard Ratios significatifs (p<0.05) :")
sig_prev = cph_prev.summary[cph_prev.summary['p'] < 0.05][['exp(coef)', 'p']]
sig_prev.columns = ['HR', 'p']
sig_prev = sig_prev.sort_values('HR', ascending=False)
print(sig_prev.to_string())

cph_prev.summary.to_csv('/home/user/EAuagent/models/cox_cause_specific_preventif.csv')

# ── 4. Modèle cause-specific : CORRECTIF (cause 2) ───────────
print("\n" + "=" * 55)
print("MODÈLE CAUSE-SPECIFIC : ABANDON CORRECTIF (cause 2)")
print("=" * 55)

df_corr = model_df.copy()
# Événement = 1 si correctif, 0 sinon
df_corr['event'] = (df_corr['event_code'] == 2).astype(int)
df_corr = df_corr.drop(columns=['event_code'])

cph_corr = CoxPHFitter(penalizer=0.01)
cph_corr.fit(df_corr, duration_col=duration_col, event_col='event', show_progress=False)

print(f"C-index correctif : {cph_corr.concordance_index_:.4f}")
print("\nHazard Ratios significatifs (p<0.05) :")
sig_corr = cph_corr.summary[cph_corr.summary['p'] < 0.05][['exp(coef)', 'p']]
sig_corr.columns = ['HR', 'p']
sig_corr = sig_corr.sort_values('HR', ascending=False)
print(sig_corr.to_string())

cph_corr.summary.to_csv('/home/user/EAuagent/models/cox_cause_specific_correctif.csv')

# ── 5. FIGURES ────────────────────────────────────────────────

# 5a. Forest plot comparatif : Préventif vs Correctif
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

for ax, (title, cph_model, color) in zip(axes, [
    ('Abandon PRÉVENTIF\n(cause 1)', cph_prev, '#f39c12'),
    ('Abandon CORRECTIF\n(cause 2)', cph_corr, '#e74c3c'),
]):
    hr = cph_model.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
    hr.columns = ['HR', 'HR_lo', 'HR_hi', 'p']
    hr = hr.sort_values('HR')

    colors_bar = [color if p < 0.05 else '#95a5a6' for p in hr['p']]
    y_pos = range(len(hr))

    ax.barh(y_pos, hr['HR'] - 1, left=1, color=colors_bar, alpha=0.7, height=0.6)
    ax.errorbar(hr['HR'], y_pos, xerr=[hr['HR'] - hr['HR_lo'], hr['HR_hi'] - hr['HR']],
                fmt='o', color='black', markersize=4, capsize=2)
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(hr.index, fontsize=8)
    ax.set_xlabel('Hazard Ratio', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('Risques compétitifs — Comparaison HR Préventif vs Correctif\n(cause-specific Cox, réf. = FT)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape7_forest_competitif.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure sauvegardée : figures/etape7_forest_competitif.png")

# 5b. Comparaison des HR entre les deux causes
fig, ax = plt.subplots(figsize=(10, 7))

# Extraire HR pour les deux modèles
hr_prev_all = cph_prev.summary['exp(coef)'].rename('HR_preventif')
hr_corr_all = cph_corr.summary['exp(coef)'].rename('HR_correctif')
hr_compare = pd.concat([hr_prev_all, hr_corr_all], axis=1)

# Scatter plot
ax.scatter(np.log2(hr_compare['HR_preventif']),
           np.log2(hr_compare['HR_correctif']),
           s=80, color='#3498db', alpha=0.8, edgecolors='black', linewidth=0.5)

for idx in hr_compare.index:
    ax.annotate(idx,
                (np.log2(hr_compare.loc[idx, 'HR_preventif']),
                 np.log2(hr_compare.loc[idx, 'HR_correctif'])),
                fontsize=7, textcoords="offset points", xytext=(5, 5))

# Diagonale
lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
ax.axhline(y=0, color='grey', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='grey', linestyle='-', alpha=0.3)

ax.set_xlabel('log₂(HR) — Abandon préventif', fontsize=11)
ax.set_ylabel('log₂(HR) — Abandon correctif', fontsize=11)
ax.set_title('Comparaison des effets : Préventif vs Correctif\n(cause-specific Cox, log₂ HR)',
             fontsize=13, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape7_scatter_hr_causes.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape7_scatter_hr_causes.png")

# 5c. Incidence cumulée par matériau et par cause
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

materials_main = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM']
colors_mat = plt.cm.Set1(np.linspace(0, 1, len(materials_main)))

for ax, (cause_code, cause_name) in zip(axes, [(1, 'Préventif'), (2, 'Correctif')]):
    for i, mat in enumerate(materials_main):
        mask = df['MAT_grp'] == mat
        subset = df[mask]
        if len(subset) < 100:
            continue

        # Calculer l'incidence cumulée brute pour cette cause
        times = np.sort(subset[duration_col].unique())
        n_total = len(subset)

        # Événements de cette cause
        events_this_cause = subset[subset['event_code'] == cause_code]
        event_times = events_this_cause[duration_col].value_counts().sort_index()

        cum_inc = []
        n_risk = n_total
        cum_hazard = 0
        for t in times:
            n_events = event_times.get(t, 0)
            n_any_event = len(subset[subset[duration_col] == t]) - len(subset[(subset[duration_col] == t) & (subset['event_code'] == 0)])
            if n_risk > 0:
                cum_hazard += n_events / n_risk
            n_risk -= len(subset[subset[duration_col] == t])
            cum_inc.append(cum_hazard)

        ax.plot(times, cum_inc, label=f'{mat} (n={len(subset)})',
                color=colors_mat[i], linewidth=2)

    ax.set_xlabel('Durée (années)', fontsize=10)
    ax.set_ylabel('Incidence cumulée', fontsize=10)
    ax.set_title(f'Incidence cumulée — {cause_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 130)

plt.suptitle('Risques compétitifs — Incidence cumulée par cause et matériau\n(Dataset A)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape7_incidence_cumulee_causes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : figures/etape7_incidence_cumulee_causes.png")

# 5d. Tableau récapitulatif des différences entre causes
print("\n" + "=" * 55)
print("COMPARAISON DES FACTEURS DE RISQUE PAR CAUSE")
print("=" * 55)

compare_df = pd.DataFrame({
    'Variable': hr_compare.index,
    'HR_preventif': hr_compare['HR_preventif'].values,
    'HR_correctif': hr_compare['HR_correctif'].values,
    'p_prev': cph_prev.summary['p'].values,
    'p_corr': cph_corr.summary['p'].values,
})
compare_df['ratio_HR'] = compare_df['HR_correctif'] / compare_df['HR_preventif']
compare_df['sig_prev'] = compare_df['p_prev'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
compare_df['sig_corr'] = compare_df['p_corr'].apply(lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns')
compare_df = compare_df.sort_values('ratio_HR', ascending=False)

print(compare_df[['Variable', 'HR_preventif', 'sig_prev', 'HR_correctif', 'sig_corr', 'ratio_HR']].to_string(index=False))
compare_df.to_csv('/home/user/EAuagent/models/comparaison_causes_HR.csv', index=False)

print("\n✓ Étape 7 terminée.")
