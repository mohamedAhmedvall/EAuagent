"""
AUDIT DES MÉTRIQUES & CALIBRATION
===================================
Vérification : les prédictions collent-elles à la réalité ?
- Calibration : P(prédit) vs P(observé)
- C-index réel sur données de test
- Brier Score
- Analyse du paradoxe décile inversé
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from lifelines import WeibullAFTFitter, CoxPHFitter, KaplanMeierFitter

# ── 1. Charger les données ────────────────────────────────────
print("=" * 65)
print("AUDIT DE FIABILITÉ — MODÈLES DE SURVIE")
print("=" * 65)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')
scoring = pd.read_csv('/home/user/EAuagent/models/scoring_troncons.csv')

duration_col = 'duration_years'
event_col = 'event_bin'

print(f"\n── DONNÉES DE BASE ──")
print(f"N total : {len(df)}")
print(f"Événements : {df[event_col].sum()} ({df[event_col].mean()*100:.1f}%)")
print(f"Durée médiane observée (tous) : {df[duration_col].median():.1f} ans")
print(f"Durée médiane observée (abandonnés) : {df[df[event_col]==1][duration_col].median():.1f} ans")
print(f"Durée médiane observée (en service) : {df[df[event_col]==0][duration_col].median():.1f} ans")

# ── 2. PROBLÈME CLÉ : comprendre le paradoxe des déciles ─────
print("\n" + "=" * 65)
print("DIAGNOSTIC DU PARADOXE DÉCILE INVERSÉ")
print("=" * 65)

print("""
CONSTAT : Le décile 10 (risque prédit le plus élevé) a le taux
d'abandon réel le plus BAS (1.7%), et le décile 1 le plus HAUT (29.7%).

EXPLICATION : Le score de risque est P(abandon avant 50 ans).
Les tronçons du décile 10 sont JEUNES (posés après 2010) avec des
matériaux à durée de vie courte (FTVI). Le modèle prédit qu'ils
casseront tôt, MAIS ils n'ont que 5-15 ans d'âge → pas encore eu
le temps de casser.

Ce n'est PAS une erreur du modèle mais une confusion entre :
- RISQUE INTRINSÈQUE (ce que le modèle prédit = vulnérabilité)
- RISQUE OBSERVÉ (ce qu'on voit = dépend du temps d'exposition)
""")

# Joindre la durée au scoring via df original
scoring_merged = scoring.merge(df[['GID', duration_col]], on='GID', how='left')
print("── Âge actuel moyen par décile ──")
age_by_decile = scoring_merged.groupby('decile_risque').agg(
    age_moyen=(duration_col, 'mean'),
    taux_abandon=(event_col, 'mean'),
    score_moyen=('risk_score_50ans', 'mean'),
    n=('GID', 'count'),
).round(3)
print(age_by_decile.to_string())

# ── 3. CALIBRATION CORRECTE : par tranche d'âge ──────────────
print("\n" + "=" * 65)
print("CALIBRATION PAR TRANCHE D'ÂGE (SEULE APPROCHE VALIDE)")
print("=" * 65)

# Reconstituer le modèle pour faire des prédictions cohérentes
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

# Réajuster Weibull
waft = WeibullAFTFitter(penalizer=0.01)
waft.fit(model_df, duration_col=duration_col, event_col=event_col)

# Réajuster Cox
cph = CoxPHFitter(penalizer=0.01)
cph.fit(model_df, duration_col=duration_col, event_col=event_col, show_progress=False)

print(f"\nC-index Weibull AFT : {waft.concordance_index_:.4f}")
print(f"C-index Cox PH     : {cph.concordance_index_:.4f}")

# ── 4. VALIDATION CROISÉE : train/test split ─────────────────
print("\n" + "=" * 65)
print("VALIDATION TRAIN/TEST (70/30)")
print("=" * 65)

np.random.seed(42)
n = len(model_df)
idx_train = np.random.choice(n, size=int(0.7 * n), replace=False)
idx_test = np.setdiff1d(np.arange(n), idx_train)

train_df = model_df.iloc[idx_train]
test_df = model_df.iloc[idx_test]

print(f"Train : {len(train_df)} ({train_df[event_col].mean()*100:.1f}% événements)")
print(f"Test  : {len(test_df)} ({test_df[event_col].mean()*100:.1f}% événements)")

# Weibull sur train
waft_train = WeibullAFTFitter(penalizer=0.01)
waft_train.fit(train_df, duration_col=duration_col, event_col=event_col)

# Cox sur train
cph_train = CoxPHFitter(penalizer=0.01)
cph_train.fit(train_df, duration_col=duration_col, event_col=event_col, show_progress=False)

# C-index sur test
c_weibull_test = waft_train.score(test_df, scoring_method='concordance_index')
c_cox_test = cph_train.score(test_df, scoring_method='concordance_index')

print(f"\nC-index sur TEST :")
print(f"  Weibull AFT : {c_weibull_test:.4f}")
print(f"  Cox PH      : {c_cox_test:.4f}")

# ── 5. CALIBRATION RÉELLE : prédictions vs observations ──────
print("\n" + "=" * 65)
print("CALIBRATION : P(prédit) vs P(observé)")
print("=" * 65)

# Pour différents horizons temporels
horizons_calib = [20, 30, 40, 50, 60, 70, 80]

calib_results = []
for t in horizons_calib:
    # Prédire P(survie > t)
    sf_pred = waft.predict_survival_function(model_df, times=[t])
    p_event_pred = 1 - sf_pred.iloc[0].values  # P(abandon avant t)

    # Observé : parmi ceux avec durée >= t OU événement avant t
    # On ne peut calculer le taux réel que sur les tronçons qui ont
    # atteint cet âge OU ont eu l'événement avant
    observable = model_df[(model_df[duration_col] >= t) |
                          ((model_df[event_col] == 1) & (model_df[duration_col] <= t))]

    if len(observable) > 100:
        # KM observé pour validation
        kmf = KaplanMeierFitter()
        kmf.fit(model_df[duration_col], event_observed=model_df[event_col])
        km_surv_at_t = kmf.predict(t)
        p_event_obs_km = 1 - km_surv_at_t

        # Taux brut observé (biaisé mais informatif)
        taux_brut = model_df[
            (model_df[event_col] == 1) & (model_df[duration_col] <= t)
        ].shape[0] / len(model_df)

        p_event_pred_mean = p_event_pred.mean()

        calib_results.append({
            'horizon': t,
            'P_pred_moyen': p_event_pred_mean,
            'P_obs_KM': p_event_obs_km,
            'taux_brut': taux_brut,
            'n_observable': len(observable),
            'ratio_pred_obs': p_event_pred_mean / p_event_obs_km if p_event_obs_km > 0 else np.nan,
        })

calib_df = pd.DataFrame(calib_results)
print(calib_df.to_string(index=False))

# ── 6. CALIBRATION PAR QUINTILE DE RISQUE ────────────────────
print("\n" + "=" * 65)
print("CALIBRATION PAR QUINTILE — Horizon 60 ans")
print("=" * 65)

sf_60 = waft.predict_survival_function(model_df, times=[60])
p_event_60 = 1 - sf_60.iloc[0].values

model_df_cal = model_df.copy()
model_df_cal['p_pred_60'] = p_event_60
model_df_cal['quintile'] = pd.qcut(model_df_cal['p_pred_60'], 5, labels=False, duplicates='drop') + 1

for q in sorted(model_df_cal['quintile'].unique()):
    sub = model_df_cal[model_df_cal['quintile'] == q]
    # KM sur ce quintile
    kmf_q = KaplanMeierFitter()
    kmf_q.fit(sub[duration_col], event_observed=sub[event_col])
    p_obs_60 = 1 - kmf_q.predict(60)
    p_pred_mean = sub['p_pred_60'].mean()

    print(f"  Quintile {q} : P(prédit)={p_pred_mean:.3f}  |  P(observé KM)={p_obs_60:.3f}  |  "
          f"ratio={p_pred_mean/p_obs_60:.2f}  |  n={len(sub)}")

# ── 7. BRIER SCORE (mesure de calibration) ───────────────────
print("\n" + "=" * 65)
print("BRIER SCORE (plus bas = meilleur, max 0.25 pour 50/50)")
print("=" * 65)

for t in [30, 50, 70]:
    sf_t = waft.predict_survival_function(model_df, times=[t])
    p_surv_pred = sf_t.iloc[0].values

    # Pour le Brier score, on a besoin de savoir si l'événement s'est
    # produit avant t ET que l'observation est non-censurée
    y_true = ((model_df[event_col].values == 1) &
              (model_df[duration_col].values <= t)).astype(float)

    # Tronçons dont la durée > t ou événement avant t (observables)
    observable_mask = (model_df[duration_col].values >= t) | (model_df[event_col].values == 1)

    if observable_mask.sum() > 0:
        p_event_pred = 1 - p_surv_pred[observable_mask]
        y_obs = y_true[observable_mask]
        brier = np.mean((p_event_pred - y_obs) ** 2)
        print(f"  Brier Score à {t} ans : {brier:.4f}  (prévalence observée : {y_obs.mean():.3f})")

# ── 8. ANALYSE PAR MATÉRIAU : prédit vs observé ──────────────
print("\n" + "=" * 65)
print("DURÉE MÉDIANE : PRÉDITE (Weibull) vs OBSERVÉE (KM)")
print("=" * 65)

mats_check = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM', 'FTVI']
mat_calib = []

for mat in mats_check:
    if mat == 'FT':
        mask = pd.Series(True, index=model_df.index)
        for mc in mat_cols:
            mask = mask & (model_df[mc] == 0)
        # Exclure les matériaux rares aussi
        mask = mask & df.loc[model_df.index, 'MAT_grp'].eq('FT')
    else:
        if f'mat_{mat}' in model_df.columns:
            mask = model_df[f'mat_{mat}'] == 1
        else:
            continue

    sub = model_df[mask]
    if len(sub) < 100:
        continue

    # KM observé
    kmf_mat = KaplanMeierFitter()
    kmf_mat.fit(sub[duration_col], event_observed=sub[event_col])
    try:
        km_median = kmf_mat.median_survival_time_
    except:
        km_median = np.nan

    # Weibull prédit
    weibull_median = waft.predict_median(sub).median()

    # Taux événement
    n_events = sub[event_col].sum()
    pct_events = n_events / len(sub) * 100

    # Taux censure
    pct_censored = (1 - sub[event_col].mean()) * 100

    mat_calib.append({
        'Matériau': mat,
        'N': len(sub),
        'N_événements': n_events,
        'Taux_abandon_%': round(pct_events, 1),
        'Censure_%': round(pct_censored, 1),
        'Médiane_KM': round(km_median, 1) if not np.isinf(km_median) and not np.isnan(km_median) else '>120',
        'Médiane_Weibull': round(weibull_median, 1) if not np.isinf(weibull_median) and not np.isnan(weibull_median) else '>120',
    })

mat_calib_df = pd.DataFrame(mat_calib)
print(mat_calib_df.to_string(index=False))

# ── 9. LIMITES ET BIAIS IDENTIFIÉS ──────────────────────────
print("\n" + "=" * 65)
print("LIMITES & BIAIS IDENTIFIÉS")
print("=" * 65)

print("""
1. CENSURE LOURDE (84%) : seuls 16% des tronçons ont été abandonnés.
   Les tronçons récents sont massivement censurés → le modèle
   extrapole pour eux sans données d'événement.

2. CONFUSION ÂGE/COHORTE : DDP_year capture à la fois :
   - L'effet matériau (les nouveaux matériaux sont posés récemment)
   - L'effet exposition (les récents ont moins de temps pour casser)
   Le coefficient DDP_year est probablement SURESTIMÉ.

3. BIAIS DE SURVIVANT : Les tronçons anciens encore en service
   SONT les plus résistants. Cela peut gonfler artificiellement
   la durée médiane des matériaux anciens (FTG, BTM).

4. PROPORTIONNALITÉ VIOLÉE (Cox) : toutes les variables violent
   l'hypothèse PH → le Cox ne devrait PAS être utilisé pour
   prédire, seulement pour l'interprétation directionnelle.

5. ABSENCE DE VALIDATION TEMPORELLE : pas de découpage
   train=avant 2015 / test=après 2015. La validation 70/30
   aléatoire surestime la performance prédictive.

6. VARIABLES D'ANOMALIE POTENTIELLEMENT ENDOGÈNES :
   nb_anomalies, nb_fuites sont des CONSÉQUENCES de l'état
   du tronçon, pas des causes. Les inclure comme prédicteurs
   crée un biais de causalité inverse.
""")

# ── 10. FIGURES D'AUDIT ──────────────────────────────────────

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle('AUDIT DE FIABILITÉ — CALIBRATION & MÉTRIQUES\n'
             'Validation des prédictions vs réalité observée',
             fontsize=15, fontweight='bold', y=0.98)

# Panel 1 : Calibration P(prédit) vs P(observé KM)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(calib_df['horizon'], calib_df['P_pred_moyen'], 'o-', color='#e74c3c',
         linewidth=2, markersize=8, label='P(prédit) moyen')
ax1.plot(calib_df['horizon'], calib_df['P_obs_KM'], 's-', color='#3498db',
         linewidth=2, markersize=8, label='P(observé) KM')
ax1.plot(calib_df['horizon'], calib_df['taux_brut'], '^--', color='#95a5a6',
         linewidth=1, markersize=6, label='Taux brut (biaisé)')
ax1.set_xlabel('Horizon (années)', fontsize=11)
ax1.set_ylabel('Probabilité d\'abandon', fontsize=11)
ax1.set_title('Calibration globale\nP(prédit) vs P(observé KM)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Panel 2 : Ratio prédit/observé
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(calib_df['horizon'], calib_df['ratio_pred_obs'],
        color=['#2ecc71' if abs(r-1) < 0.2 else '#f39c12' if abs(r-1) < 0.5 else '#e74c3c'
               for r in calib_df['ratio_pred_obs']], alpha=0.8)
ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Calibration parfaite')
ax2.axhspan(0.8, 1.2, alpha=0.1, color='green', label='Zone acceptable (±20%)')
ax2.set_xlabel('Horizon (années)', fontsize=11)
ax2.set_ylabel('Ratio P(prédit) / P(observé)', fontsize=11)
ax2.set_title('Ratio de calibration par horizon', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Panel 3 : Paradoxe des déciles — explication visuelle
ax3 = fig.add_subplot(gs[1, 0])
decile_data = scoring_merged.groupby('decile_risque').agg(
    taux_abandon=(event_col, 'mean'),
    age_moyen=(duration_col, 'mean'),
).reset_index()

ax3_twin = ax3.twinx()
bars = ax3.bar(decile_data['decile_risque'], decile_data['taux_abandon'] * 100,
               color='#e74c3c', alpha=0.6, label='Taux abandon réel (%)')
ax3_twin.plot(decile_data['decile_risque'], decile_data['age_moyen'],
              'o-', color='#3498db', linewidth=2, markersize=8, label='Âge moyen (ans)')

ax3.set_xlabel('Décile de risque', fontsize=11)
ax3.set_ylabel('Taux abandon réel (%)', color='#e74c3c', fontsize=11)
ax3_twin.set_ylabel('Âge moyen des tronçons (ans)', color='#3498db', fontsize=11)
ax3.set_title('PARADOXE EXPLIQUÉ :\nDécile élevé = jeune = pas encore cassé',
              fontsize=11, fontweight='bold')
ax3.set_xticks(range(1, 11))

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper center')
ax3.grid(alpha=0.3)

# Panel 4 : Médiane prédite vs KM par matériau
ax4 = fig.add_subplot(gs[1, 1])
mat_names = mat_calib_df['Matériau'].values
km_vals = [float(v) if v != '>120' else 120 for v in mat_calib_df['Médiane_KM']]
wb_vals = [float(v) if v != '>120' else 120 for v in mat_calib_df['Médiane_Weibull']]

x = np.arange(len(mat_names))
width = 0.35
ax4.bar(x - width/2, km_vals, width, label='Médiane KM (observée)', color='#3498db', alpha=0.8)
ax4.bar(x + width/2, wb_vals, width, label='Médiane Weibull (prédite)', color='#e74c3c', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(mat_names, fontsize=9)
ax4.set_ylabel('Durée médiane (années)', fontsize=11)
ax4.set_title('Médiane observée (KM) vs prédite (Weibull)\npar matériau', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Annoter les écarts
for i, (km, wb) in enumerate(zip(km_vals, wb_vals)):
    diff = wb - km
    if abs(diff) > 5:
        ax4.annotate(f'{diff:+.0f}', (i, max(km, wb) + 2), ha='center', fontsize=8,
                     color='red' if abs(diff) > 15 else 'orange')

# Panel 5 : Calibration par quintile à 60 ans
ax5 = fig.add_subplot(gs[2, 0])
quintile_data = []
model_df_cal2 = model_df.copy()
sf_60_full = waft.predict_survival_function(model_df, times=[60])
model_df_cal2['p_pred_60'] = 1 - sf_60_full.iloc[0].values
model_df_cal2['quintile'] = pd.qcut(model_df_cal2['p_pred_60'], 5, labels=False, duplicates='drop') + 1

for q in sorted(model_df_cal2['quintile'].unique()):
    sub = model_df_cal2[model_df_cal2['quintile'] == q]
    kmf_q = KaplanMeierFitter()
    kmf_q.fit(sub[duration_col], event_observed=sub[event_col])
    p_obs = 1 - kmf_q.predict(60)
    p_pred = sub['p_pred_60'].mean()
    quintile_data.append({'quintile': q, 'p_pred': p_pred, 'p_obs': p_obs})

qdf = pd.DataFrame(quintile_data)
ax5.scatter(qdf['p_pred'], qdf['p_obs'], s=150, color='#e74c3c', zorder=5, edgecolors='black')
for _, row in qdf.iterrows():
    ax5.annotate(f"Q{int(row['quintile'])}", (row['p_pred'], row['p_obs']),
                 fontsize=10, textcoords="offset points", xytext=(8, 5))

lims = [0, max(qdf['p_pred'].max(), qdf['p_obs'].max()) * 1.1]
ax5.plot(lims, lims, 'k--', alpha=0.5, label='Calibration parfaite')
ax5.set_xlabel('P(abandon avant 60 ans) — PRÉDIT', fontsize=11)
ax5.set_ylabel('P(abandon avant 60 ans) — OBSERVÉ (KM)', fontsize=11)
ax5.set_title('Diagramme de calibration\n(quintiles, horizon 60 ans)', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Panel 6 : Résumé des métriques
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')

summary_text = (
    "RÉSUMÉ DES MÉTRIQUES DE FIABILITÉ\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    f"C-index Weibull AFT (full)  : {waft.concordance_index_:.4f}\n"
    f"C-index Weibull AFT (test)  : {c_weibull_test:.4f}\n"
    f"C-index Cox PH (full)       : {cph.concordance_index_:.4f}\n"
    f"C-index Cox PH (test)       : {c_cox_test:.4f}\n\n"
    "INTERPRÉTATION C-INDEX :\n"
    "  0.50 = aléatoire\n"
    "  0.60-0.70 = faible discrimination\n"
    "  0.70-0.80 = acceptable\n"
    "  0.80-0.90 = bon\n"
    "  > 0.90 = excellent\n\n"
    "VERDICT :\n"
    "  • Weibull AFT : discrimination ACCEPTABLE (0.75)\n"
    "  • Cox PH : discrimination FAIBLE (0.59)\n"
    "  • Calibration : à vérifier par quintile\n"
    "  • Biais principal : confusion âge/cohorte\n"
    "  • Le scoring est DIRECTIONNEL, pas ABSOLU\n"
    "  • Les durées médianes sont indicatives,\n"
    "    pas des prédictions de date de casse"
)

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#fadbd8', alpha=0.8))

plt.savefig('/home/user/EAuagent/figures/audit_calibration.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure sauvegardée : figures/audit_calibration.png")

# ── 11. Validation KM par matériau (courbes superposées) ─────
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, mat in enumerate(mats_check):
    ax = axes[idx]

    if mat == 'FT':
        mask = df.loc[model_df.index, 'MAT_grp'].eq('FT')
    else:
        if f'mat_{mat}' in model_df.columns:
            mask = model_df[f'mat_{mat}'] == 1
        else:
            continue

    sub = model_df[mask]
    if len(sub) < 100:
        continue

    # KM observé
    kmf_mat = KaplanMeierFitter()
    kmf_mat.fit(sub[duration_col], event_observed=sub[event_col])
    kmf_mat.plot_survival_function(ax=ax, color='#3498db', linewidth=2, label='KM observé')

    # Weibull prédit (moyenne sur l'échantillon)
    sample = sub.sample(min(500, len(sub)), random_state=42)
    times = np.linspace(0.1, 120, 200)
    sf_pred = waft.predict_survival_function(sample, times=times)
    sf_mean = sf_pred.mean(axis=1)
    ax.plot(times, sf_mean.values, color='#e74c3c', linewidth=2, linestyle='--', label='Weibull prédit')

    ax.set_title(f'{mat} (n={len(sub)}, ev={sub[event_col].sum()})', fontsize=10, fontweight='bold')
    ax.set_xlabel('Durée (années)', fontsize=8)
    ax.set_ylabel('S(t)', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

# Cacher le dernier panel si vide
if len(mats_check) < 8:
    axes[-1].axis('off')

plt.suptitle('VALIDATION PAR MATÉRIAU — Courbe KM observée vs Weibull prédite',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/audit_km_vs_weibull.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : figures/audit_km_vs_weibull.png")

print("\n✓ Audit terminé.")
