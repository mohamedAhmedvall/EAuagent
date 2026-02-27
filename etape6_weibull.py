"""
ÉTAPE 6 — Modélisation Weibull AFT sur Dataset B (tous abandons)
================================================================
- Ajustement Weibull AFT multivarié
- Comparaison AIC/BIC avec Cox
- Paramètres de forme (rho) et d'échelle (lambda) par matériau
- Courbes de survie paramétriques
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

# ── 1. Chargement ──────────────────────────────────────────────
print("=" * 60)
print("ÉTAPE 6 — MODÈLE WEIBULL AFT (Dataset B — tous abandons)")
print("=" * 60)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')

duration_col = 'duration_years'
event_col = 'event_bin'

# ── 2. Préparation ────────────────────────────────────────────
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

model_df = df[[duration_col, event_col]].copy()
for col in covariates_num:
    model_df[col] = pd.to_numeric(df[col], errors='coerce')
for col in mat_cols:
    if col in mat_dummies.columns:
        model_df[col] = mat_dummies[col].values
    else:
        model_df[col] = 0

model_df = model_df.dropna()
model_df = model_df[model_df[duration_col] > 0]
print(f"Observations : {len(model_df)}, Événements : {model_df[event_col].sum()}")

# ── 3. Ajustement Weibull AFT ─────────────────────────────────
print("\n── Ajustement Weibull AFT ──")
waft = WeibullAFTFitter(penalizer=0.01)
waft.fit(model_df, duration_col=duration_col, event_col=event_col)

print("\n── RÉSULTATS WEIBULL AFT ──")
summary_w = waft.summary
print(summary_w.to_string())
print(f"\nAIC Weibull : {waft.AIC_:.1f}")
print(f"BIC Weibull : {waft.BIC_:.1f}")
print(f"C-index Weibull : {waft.concordance_index_:.4f}")

# ── 4. Log-Normal AFT pour comparaison ────────────────────────
print("\n── Ajustement Log-Normal AFT ──")
lnaft = LogNormalAFTFitter(penalizer=0.01)
lnaft.fit(model_df, duration_col=duration_col, event_col=event_col)
print(f"AIC Log-Normal : {lnaft.AIC_:.1f}")
print(f"BIC Log-Normal : {lnaft.BIC_:.1f}")
print(f"C-index Log-Normal : {lnaft.concordance_index_:.4f}")

# ── 5. Log-Logistic AFT pour comparaison ──────────────────────
print("\n── Ajustement Log-Logistique AFT ──")
llaft = LogLogisticAFTFitter(penalizer=0.01)
llaft.fit(model_df, duration_col=duration_col, event_col=event_col)
print(f"AIC Log-Logistique : {llaft.AIC_:.1f}")
print(f"BIC Log-Logistique : {llaft.BIC_:.1f}")
print(f"C-index Log-Logistique : {llaft.concordance_index_:.4f}")

# ── 6. Tableau comparatif ─────────────────────────────────────
print("\n" + "=" * 55)
print("COMPARAISON DES MODÈLES")
print("=" * 55)
comparison = pd.DataFrame({
    'Modèle': ['Cox PH', 'Weibull AFT', 'Log-Normal AFT', 'Log-Logistique AFT'],
    'AIC': ['-', f'{waft.AIC_:.1f}', f'{lnaft.AIC_:.1f}', f'{llaft.AIC_:.1f}'],
    'BIC': ['-', f'{waft.BIC_:.1f}', f'{lnaft.BIC_:.1f}', f'{llaft.BIC_:.1f}'],
    'C-index': ['0.5862', f'{waft.concordance_index_:.4f}', f'{lnaft.concordance_index_:.4f}', f'{llaft.concordance_index_:.4f}'],
})
print(comparison.to_string(index=False))
comparison.to_csv('/home/user/EAuagent/models/comparaison_modeles.csv', index=False)

# ── 7. Sauvegarde résumé Weibull ──────────────────────────────
summary_w.to_csv('/home/user/EAuagent/models/weibull_aft_summary_B.csv')

# ── 8. Paramètre de forme Weibull ─────────────────────────────
rho = np.exp(waft.summary.loc[('rho_', 'Intercept'), 'coef'])
print(f"\nParamètre de forme Weibull (rho) : {rho:.4f}")
if rho > 1:
    print("  → rho > 1 : risque croissant avec l'âge (vieillissement)")
elif rho < 1:
    print("  → rho < 1 : risque décroissant (mortalité infantile)")
else:
    print("  → rho ≈ 1 : risque constant (exponentiel)")

# ── 9. FIGURES ────────────────────────────────────────────────

# 9a. Comparaison AIC des modèles paramétriques
fig, ax = plt.subplots(figsize=(8, 5))
models_aic = {
    'Weibull AFT': waft.AIC_,
    'Log-Normal AFT': lnaft.AIC_,
    'Log-Logistique AFT': llaft.AIC_,
}
bars = ax.barh(list(models_aic.keys()), list(models_aic.values()),
               color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
ax.set_xlabel('AIC (plus bas = meilleur)', fontsize=11)
ax.set_title('Comparaison des modèles paramétriques — AIC\n(Dataset B — tous abandons)',
             fontsize=13, fontweight='bold')
for bar, val in zip(bars, models_aic.values()):
    ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape6_comparaison_aic.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape6_comparaison_aic.png")

# 9b. Courbes de survie Weibull par matériau
fig, ax = plt.subplots(figsize=(10, 6))

materials_plot = ['FT'] + mats_keep[:6]
colors_mat = plt.cm.Set1(np.linspace(0, 1, len(materials_plot)))
median_vals = model_df[covariates_num].median()

for i, mat in enumerate(materials_plot):
    # Filtrer
    if mat == 'FT':
        mask = pd.Series(True, index=model_df.index)
        for mc in mat_cols:
            mask = mask & (model_df[mc] == 0)
    else:
        if f'mat_{mat}' in model_df.columns:
            mask = model_df[f'mat_{mat}'] == 1
        else:
            continue

    subset = model_df[mask]
    if len(subset) < 100:
        continue

    sample = subset.sample(min(500, len(subset)), random_state=42)
    sf = waft.predict_survival_function(sample)
    sf_mean = sf.mean(axis=1)
    ax.plot(sf_mean.index, sf_mean.values, label=f'{mat} (n={len(subset)})',
            color=colors_mat[i], linewidth=2)

ax.set_xlabel('Durée (années)', fontsize=11)
ax.set_ylabel('S(t) — Probabilité de survie', fontsize=11)
ax.set_title('Weibull AFT — Courbes de survie par matériau\n(Dataset B — tous abandons)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 130)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape6_weibull_survie_materiau.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape6_weibull_survie_materiau.png")

# 9c. Coefficients Weibull (lambda_ = partie accélération)
fig, ax = plt.subplots(figsize=(10, 7))

# Extraire les coefficients lambda (accélération de vie)
lambda_coefs = summary_w.loc['lambda_']
lambda_coefs = lambda_coefs[lambda_coefs.index != 'Intercept']
coefs = lambda_coefs['coef'].sort_values()

colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.05 else '#95a5a6'
          for p in lambda_coefs.loc[coefs.index, 'p']]

ax.barh(range(len(coefs)), coefs.values, color=colors, alpha=0.8)
ax.set_yticks(range(len(coefs)))
ax.set_yticklabels(coefs.index, fontsize=9)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Coefficient Weibull AFT (>0 = durée de vie plus longue)', fontsize=11)
ax.set_title('Weibull AFT — Coefficients d\'accélération (λ)\n(Dataset B, réf. = FT)',
             fontsize=13, fontweight='bold')

import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='#e74c3c', alpha=0.8, label='p < 0.001'),
    mpatches.Patch(color='#f39c12', alpha=0.8, label='p < 0.05'),
    mpatches.Patch(color='#95a5a6', alpha=0.8, label='Non significatif'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape6_weibull_coefficients.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape6_weibull_coefficients.png")

# 9d. Durées médianes prédites par matériau
fig, ax = plt.subplots(figsize=(9, 5))

medians = {}
for mat in materials_plot:
    if mat == 'FT':
        mask = pd.Series(True, index=model_df.index)
        for mc in mat_cols:
            mask = mask & (model_df[mc] == 0)
    else:
        if f'mat_{mat}' in model_df.columns:
            mask = model_df[f'mat_{mat}'] == 1
        else:
            continue

    subset = model_df[mask]
    if len(subset) < 100:
        continue

    med = waft.predict_median(subset).median()
    medians[mat] = med

mats_sorted = sorted(medians.keys(), key=lambda x: medians[x])
vals = [medians[m] for m in mats_sorted]
colors_bar = ['#e74c3c' if v < 50 else '#f39c12' if v < 70 else '#2ecc71' for v in vals]

ax.barh(mats_sorted, vals, color=colors_bar, alpha=0.8)
for j, (mat, val) in enumerate(zip(mats_sorted, vals)):
    label = f'{val:.0f} ans' if not np.isinf(val) else '> 120 ans'
    ax.text(min(val, 120) + 1, j, label, va='center', fontsize=10)

ax.set_xlabel('Durée médiane prédite (années)', fontsize=11)
ax.set_title('Weibull AFT — Durée médiane de survie par matériau\n(Dataset B)',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 140)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape6_weibull_mediane_materiau.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape6_weibull_mediane_materiau.png")

print("\n✓ Étape 6 terminée.")
