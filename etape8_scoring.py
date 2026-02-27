"""
ÉTAPE 8 — Scoring & prédiction par tronçon
============================================
- Score de risque individuel (Weibull AFT)
- Courbes de survie prédites pour différents profils
- Identification des tronçons prioritaires (top 10%)
- Export du scoring complet
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from lifelines import WeibullAFTFitter

# ── 1. Chargement et réajustement Weibull ─────────────────────
print("=" * 60)
print("ÉTAPE 8 — SCORING & PRÉDICTION PAR TRONÇON")
print("=" * 60)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')

duration_col = 'duration_years'
event_col = 'event_bin'

# Préparation identique à l'étape 6
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

# Réajuster Weibull AFT
waft = WeibullAFTFitter(penalizer=0.01)
waft.fit(model_df, duration_col=duration_col, event_col=event_col)

# ── 2. Prédictions ────────────────────────────────────────────
print("\n── Calcul des scores de risque ──")

# Durée médiane prédite = indicateur principal
median_pred = waft.predict_median(model_df)

# Probabilité de survie à différents horizons
horizons = [10, 20, 30, 50, 70]
surv_probs = {}
for h in horizons:
    sf = waft.predict_survival_function(model_df, times=[h])
    surv_probs[f'P_survie_{h}ans'] = sf.iloc[0].values

# Score de risque = 1 - P(survie à 50 ans)
risk_score = 1 - surv_probs['P_survie_50ans']

# ── 3. Construire le tableau de scoring ───────────────────────
scoring = df.loc[model_df.index].copy()
scoring['duree_mediane_pred'] = median_pred.values
scoring['risk_score_50ans'] = risk_score
for h in horizons:
    scoring[f'P_survie_{h}ans'] = surv_probs[f'P_survie_{h}ans']

# Déciles de risque
scoring['decile_risque'] = pd.qcut(scoring['risk_score_50ans'], 10, labels=False, duplicates='drop') + 1
scoring['top10_pourcent'] = (scoring['decile_risque'] == 10).astype(int)

print(f"\nTronçons scorés : {len(scoring)}")
print(f"Top 10% à risque : {scoring['top10_pourcent'].sum()} tronçons")

# ── 4. Statistiques par décile ────────────────────────────────
print("\n── STATISTIQUES PAR DÉCILE DE RISQUE ──")
decile_stats = scoring.groupby('decile_risque').agg(
    n_troncons=('GID', 'count'),
    taux_abandon_reel=(event_col, 'mean'),
    risk_score_moyen=('risk_score_50ans', 'mean'),
    duree_mediane_moy=('duree_mediane_pred', 'mean'),
    pct_FTG=('MAT_grp', lambda x: (x == 'FTG').mean()),
    pct_FT=('MAT_grp', lambda x: (x == 'FT').mean()),
    diametre_moyen=('DIAMETRE_imp', 'mean'),
).round(3)
print(decile_stats.to_string())

# ── 5. Profil des top 10% ────────────────────────────────────
top10 = scoring[scoring['top10_pourcent'] == 1]
print(f"\n── PROFIL DES TOP 10% (n={len(top10)}) ──")
print(f"Taux d'abandon réel : {top10[event_col].mean()*100:.1f}%")
print(f"Score de risque moyen : {top10['risk_score_50ans'].mean():.3f}")
print(f"Durée médiane prédite moyenne : {top10['duree_mediane_pred'].mean():.1f} ans")
print(f"\nDistribution matériau :")
print(top10['MAT_grp'].value_counts(normalize=True).head(5).to_string())
print(f"\nDécennie de pose :")
print(top10['decade_pose'].value_counts(normalize=True).sort_index().to_string())

# ── 6. Export du scoring ──────────────────────────────────────
export_cols = ['GID', 'MAT_grp', 'DIAMETRE_imp', 'LNG', 'DDP_year', 'decade_pose',
               'nb_anomalies', 'nb_fuites_signalees', 'nb_fuites_detectees',
               'STATUT_OBJET', 'abandon_type', event_col,
               'duree_mediane_pred', 'risk_score_50ans',
               'P_survie_10ans', 'P_survie_20ans', 'P_survie_30ans',
               'P_survie_50ans', 'P_survie_70ans',
               'decile_risque', 'top10_pourcent']
scoring_export = scoring[[c for c in export_cols if c in scoring.columns]]
scoring_export.to_csv('/home/user/EAuagent/models/scoring_troncons.csv', index=False)
print(f"\nScoring exporté : models/scoring_troncons.csv ({len(scoring_export)} lignes)")

# ── 7. FIGURES ────────────────────────────────────────────────

# 7a. Distribution du score de risque
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(scoring['risk_score_50ans'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
ax.axvline(scoring['risk_score_50ans'].quantile(0.9), color='black', linestyle='--',
           linewidth=2, label='Seuil top 10%')
ax.set_xlabel('Score de risque à 50 ans', fontsize=11)
ax.set_ylabel('Nombre de tronçons', fontsize=11)
ax.set_title('Distribution du score de risque', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(scoring['duree_mediane_pred'], bins=50, color='#3498db', alpha=0.7, edgecolor='white')
ax.axvline(scoring['duree_mediane_pred'].median(), color='black', linestyle='--',
           linewidth=2, label=f'Médiane = {scoring["duree_mediane_pred"].median():.0f} ans')
ax.set_xlabel('Durée médiane prédite (années)', fontsize=11)
ax.set_ylabel('Nombre de tronçons', fontsize=11)
ax.set_title('Distribution des durées médianes prédites', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle('Scoring Weibull AFT — Dataset B', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape8_distribution_scores.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape8_distribution_scores.png")

# 7b. Taux d'abandon réel par décile (validation)
fig, ax = plt.subplots(figsize=(9, 5))
decile_abandon = scoring.groupby('decile_risque')[event_col].mean() * 100

colors_decile = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 10))
bars = ax.bar(decile_abandon.index, decile_abandon.values, color=colors_decile, edgecolor='white')
for bar, val in zip(bars, decile_abandon.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val:.1f}%', ha='center', fontsize=9)

ax.set_xlabel('Décile de risque (1=faible, 10=élevé)', fontsize=11)
ax.set_ylabel('Taux d\'abandon réel (%)', fontsize=11)
ax.set_title('Validation — Taux d\'abandon réel par décile de risque\n(Weibull AFT, Dataset B)',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(1, 11))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape8_validation_deciles.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape8_validation_deciles.png")

# 7c. Courbes de survie prédites pour profils types
fig, ax = plt.subplots(figsize=(10, 6))

profiles = {
    'FT ancien (1960), diam 100, anomalies': {
        'DIAMETRE_imp': 100, 'LNG_log': 3.5, 'DDP_year': 1960,
        'nb_anomalies': 3, 'nb_fuites_signalees': 2, 'nb_fuites_detectees': 1,
        'taux_anomalie_par_an': 0.05, 'DT_NB_LOGEMENT_imp': 40,
        'DT_FLUX_CIRCULATION_imp': 3,
    },
    'FTG récent (2000), diam 150, sans anomalie': {
        'DIAMETRE_imp': 150, 'LNG_log': 4.0, 'DDP_year': 2000,
        'nb_anomalies': 0, 'nb_fuites_signalees': 0, 'nb_fuites_detectees': 0,
        'taux_anomalie_par_an': 0, 'DT_NB_LOGEMENT_imp': 50,
        'DT_FLUX_CIRCULATION_imp': 2,
    },
    'PEHD récent (2005), diam 110': {
        'DIAMETRE_imp': 110, 'LNG_log': 3.0, 'DDP_year': 2005,
        'nb_anomalies': 0, 'nb_fuites_signalees': 0, 'nb_fuites_detectees': 0,
        'taux_anomalie_par_an': 0, 'DT_NB_LOGEMENT_imp': 30,
        'DT_FLUX_CIRCULATION_imp': 3,
    },
    'FT ancien (1950), diam 80, anomalies multiples': {
        'DIAMETRE_imp': 80, 'LNG_log': 3.0, 'DDP_year': 1950,
        'nb_anomalies': 5, 'nb_fuites_signalees': 3, 'nb_fuites_detectees': 2,
        'taux_anomalie_par_an': 0.1, 'DT_NB_LOGEMENT_imp': 60,
        'DT_FLUX_CIRCULATION_imp': 4,
    },
}

mat_assignments = {
    'FT ancien (1960), diam 100, anomalies': 'FT',
    'FTG récent (2000), diam 150, sans anomalie': 'FTG',
    'PEHD récent (2005), diam 110': 'PEHD',
    'FT ancien (1950), diam 80, anomalies multiples': 'FT',
}

colors_profiles = ['#e74c3c', '#2ecc71', '#3498db', '#e67e22']

for i, (name, profile) in enumerate(profiles.items()):
    profile_df = pd.DataFrame([profile])
    for mc in mat_cols:
        profile_df[mc] = 0

    mat = mat_assignments[name]
    if mat != 'FT' and f'mat_{mat}' in mat_cols:
        profile_df[f'mat_{mat}'] = 1

    times = np.linspace(0.1, 120, 200)
    sf = waft.predict_survival_function(profile_df, times=times)
    ax.plot(times, sf.iloc[:, 0].values, label=name, color=colors_profiles[i], linewidth=2)

ax.set_xlabel('Durée (années)', fontsize=11)
ax.set_ylabel('S(t) — Probabilité de survie', fontsize=11)
ax.set_title('Courbes de survie prédites — Profils types\n(Weibull AFT, Dataset B)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 120)
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape8_profils_survie.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape8_profils_survie.png")

# 7d. Carte matériau × décile
fig, ax = plt.subplots(figsize=(10, 6))
mats_main = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM', 'FTVI']
heatmap_data = []
for mat in mats_main:
    row = []
    for d in range(1, 11):
        mask = (scoring['MAT_grp'] == mat) & (scoring['decile_risque'] == d)
        row.append(mask.sum())
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=mats_main, columns=range(1, 11))
# Normaliser par matériau (%)
heatmap_pct = heatmap_df.div(heatmap_df.sum(axis=1), axis=0) * 100

im = ax.imshow(heatmap_pct.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(10))
ax.set_xticklabels(range(1, 11))
ax.set_yticks(range(len(mats_main)))
ax.set_yticklabels(mats_main)
ax.set_xlabel('Décile de risque (1=faible, 10=élevé)', fontsize=11)
ax.set_ylabel('Matériau', fontsize=11)
ax.set_title('Répartition des tronçons par matériau et décile de risque (%)\n(Weibull AFT)',
             fontsize=12, fontweight='bold')

# Annoter
for i in range(len(mats_main)):
    for j in range(10):
        val = heatmap_pct.values[i, j]
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=8,
                color='white' if val > 25 else 'black')

plt.colorbar(im, ax=ax, label='% du matériau')
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape8_heatmap_materiau_decile.png', dpi=150)
plt.close()
print("Figure sauvegardée : figures/etape8_heatmap_materiau_decile.png")

print("\n✓ Étape 8 terminée.")
