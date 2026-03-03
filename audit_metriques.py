"""
AUDIT DE FIABILITÉ — MODÈLE WEIBULL AFT (version corrigée)
===========================================================
Métriques couvertes :
  1. C-index train / test (discrimination)
  2. Brier Score intégré (calibration globale)
  3. Calibration P(prédit) vs P(observé KM) par quintile et par horizon
  4. SHAP — explicabilité des contributions par covariable
  5. Matrice de confusion (sur sous-ensemble avec recul suffisant ≥ 20 ans)
  6. Analyse des biais d'erreur (résidus par matériau, tranche d'âge, diamètre)
  7. Courbes KM observées vs Weibull prédites par matériau

Covariables : identiques à etape6/8 (sans DDP_year, sans taux_anomalie_par_an)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

from lifelines import WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)
from scipy.stats import spearmanr
import shap

# ══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PRÉPARATION
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("AUDIT DE FIABILITÉ — MODÈLE WEIBULL AFT (sans DDP_year ni leakage)")
print("=" * 70)

df = pd.read_csv('/home/user/EAuagent/data/dataset_B_simple.csv')
scoring = pd.read_csv('/home/user/EAuagent/models/scoring_troncons.csv')

duration_col = 'duration_years'
event_col    = 'event_bin'

print(f"\nDataset : {len(df)} tronçons | {df[event_col].sum()} événements ({df[event_col].mean()*100:.1f}%)")
print(f"Durée observée — médiane tous  : {df[duration_col].median():.1f} ans")
print(f"Durée observée — médiane events: {df[df[event_col]==1][duration_col].median():.1f} ans")

# ── Préparation features ───────────────────────────────────────────────────
mat_counts  = df['MAT_grp'].value_counts()
mats_keep   = mat_counts[mat_counts > 500].index.tolist()
if 'FT' in mats_keep:
    mats_keep.remove('FT')
mat_dummies = pd.get_dummies(df['MAT_grp'], prefix='mat', drop_first=False)
mat_cols    = [f'mat_{m}' for m in mats_keep]

COVARIATES_NUM = [
    'DIAMETRE_imp', 'LNG_log',
    'nb_anomalies', 'nb_fuites_signalees', 'nb_fuites_detectees',
    'DT_NB_LOGEMENT_imp', 'DT_FLUX_CIRCULATION_imp',
]
ALL_FEATURES = COVARIATES_NUM + mat_cols

model_df = df[[duration_col, event_col]].copy()
for col in COVARIATES_NUM:
    model_df[col] = pd.to_numeric(df[col], errors='coerce')
for col in mat_cols:
    model_df[col] = mat_dummies[col].values if col in mat_dummies.columns else 0

model_df = model_df.dropna()
model_df = model_df[model_df[duration_col] > 0]

# ── Split train/test ───────────────────────────────────────────────────────
train_df, test_df = train_test_split(model_df, test_size=0.20, random_state=42)
print(f"Train : {len(train_df)} | Test : {len(test_df)}")

# ── Ajustement ────────────────────────────────────────────────────────────
waft = WeibullAFTFitter(penalizer=0.01)
waft.fit(train_df, duration_col=duration_col, event_col=event_col)
rho = float(np.exp(waft.summary.loc[('rho_', 'Intercept'), 'coef']))

# ══════════════════════════════════════════════════════════════════════════
# 2. C-INDEX (DISCRIMINATION)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. C-INDEX (DISCRIMINATION)")
print("=" * 70)

c_train = waft.concordance_index_
c_test  = concordance_index(test_df[duration_col],
                             waft.predict_median(test_df),
                             test_df[event_col])

print(f"  C-index TRAIN : {c_train:.4f}")
print(f"  C-index TEST  : {c_test:.4f}   ← hors-échantillon")
print(f"  Écart train-test : {c_train - c_test:.4f}  (< 0.02 = pas d'overfitting)")
print(f"  Paramètre de forme rho : {rho:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. BRIER SCORE (CALIBRATION GLOBALE)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. BRIER SCORE — Calibration à différents horizons")
print("=" * 70)
print("  (0.00 = parfait, 0.25 = aléatoire pour 50/50, plus bas = mieux)")

horizons_brier = [10, 20, 30, 40, 50, 60, 70, 80]
brier_results  = []

for t in horizons_brier:
    sf_t   = waft.predict_survival_function(test_df, times=[t])
    p_surv = sf_t.iloc[0].values   # P(survie > t)

    # Événements connus au temps t (non-censurés avant t, ou encore en service à t)
    observable = ((test_df[event_col].values == 1) | (test_df[duration_col].values >= t))
    if observable.sum() < 50:
        continue

    y_true = ((test_df[event_col].values == 1) &
              (test_df[duration_col].values <= t)).astype(float)

    # Brier score IPCW simplifié (sans pondération censure pour la lisibilité)
    p_event = 1 - p_surv
    brier   = np.mean((p_event[observable] - y_true[observable]) ** 2)
    prev    = y_true[observable].mean()
    null_bs = prev * (1 - prev)   # Brier null model (prédire la prévalence)
    skill   = 1 - brier / null_bs if null_bs > 0 else np.nan   # Brier Skill Score

    brier_results.append({
        'horizon': t,
        'brier_score': round(brier, 5),
        'prevalence': round(prev, 3),
        'brier_null': round(null_bs, 5),
        'brier_skill': round(skill, 4),
        'n_obs': int(observable.sum()),
    })

brier_df = pd.DataFrame(brier_results)
print(brier_df.to_string(index=False))
print("  Interprétation Brier Skill > 0 → modèle meilleur que null (prévalence)")

# ══════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION P(prédit) vs P(observé KM)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CALIBRATION PAR HORIZON — P(prédit) moyen vs P(KM) global")
print("=" * 70)

kmf_global = KaplanMeierFitter()
kmf_global.fit(model_df[duration_col], event_observed=model_df[event_col])

calib_global = []
for t in [20, 30, 40, 50, 60, 70, 80]:
    sf_t      = waft.predict_survival_function(model_df, times=[t])
    p_pred    = (1 - sf_t.iloc[0].values).mean()
    p_km      = float(1 - kmf_global.predict(t))
    ratio     = p_pred / p_km if p_km > 0 else np.nan
    calib_global.append({'horizon': t, 'P_predit': round(p_pred, 4),
                         'P_KM': round(p_km, 4), 'ratio': round(ratio, 3)})

print(pd.DataFrame(calib_global).to_string(index=False))
print("  Ratio idéal ≈ 1.0  |  >1 = surestimation  |  <1 = sous-estimation")

# ── Calibration par quintile à 50 ans ─────────────────────────────────────
print("\n── Calibration par quintile (horizon 50 ans) ──")
sf_50   = waft.predict_survival_function(model_df, times=[50])
p50_all = 1 - sf_50.iloc[0].values

model_df_cal = model_df.copy()
model_df_cal['p_pred_50'] = p50_all
model_df_cal['quintile']  = pd.qcut(model_df_cal['p_pred_50'], 5,
                                     labels=False, duplicates='drop') + 1
calib_q = []
for q in sorted(model_df_cal['quintile'].unique()):
    sub = model_df_cal[model_df_cal['quintile'] == q]
    kmf_q = KaplanMeierFitter()
    kmf_q.fit(sub[duration_col], event_observed=sub[event_col])
    p_obs  = float(1 - kmf_q.predict(50))
    p_pred = sub['p_pred_50'].mean()
    calib_q.append({'quintile': q, 'P_predit': round(p_pred, 4),
                    'P_obs_KM': round(p_obs, 4),
                    'ratio': round(p_pred / p_obs, 3) if p_obs > 0 else np.nan,
                    'n': len(sub)})

print(pd.DataFrame(calib_q).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 5. SHAP — EXPLICABILITÉ
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. SHAP — CONTRIBUTION DES FEATURES À LOG(MÉDIANE)")
print("=" * 70)
print("  (Weibull AFT est linéaire en log(lambda) → SHAP exact via LinearExplainer)")

# Extraire les coefficients lambda du modèle ajusté sur train
lambda_summary = waft.summary.loc['lambda_']
coef_map       = lambda_summary['coef'].to_dict()

# Construire la matrice X d'entrée (features uniquement, sans duration/event)
X_full = model_df[ALL_FEATURES].copy()
X_mean = X_full.mean()

# SHAP analytique pour le Weibull AFT (linéaire en log(median))
# log(median) = intercept + sum(beta_i * x_i) + (1/rho)*log(log(2))
# SHAP(i) = beta_i * (x_i - E[x_i])
shap_contribs = pd.DataFrame({
    feat: coef_map.get(feat, 0.0) * (X_full[feat] - X_mean[feat])
    for feat in ALL_FEATURES
})

# Importance moyenne absolue
mean_abs_shap = shap_contribs.abs().mean().sort_values(ascending=False)
print("\n  Importance SHAP (|contribution| moyenne sur log(médiane)) :")
for feat, val in mean_abs_shap.items():
    direction = "↑ durée" if coef_map.get(feat, 0) > 0 else "↓ durée"
    print(f"    {feat:35s}  {val:.5f}  ({direction})")

# ── SHAP analytique (Weibull AFT = modèle linéaire en log(median)) ──────
print("\n  Calcul SHAP analytique (exact pour modèle linéaire) sur 1 000 obs …")

sample_shap = X_full.sample(1000, random_state=42).reset_index(drop=True)
coef_vec    = np.array([coef_map.get(f, 0.0) for f in ALL_FEATURES])
X_mean_arr  = X_full.mean().values

# SHAP(i, obs) = coef_i * (x_i(obs) - mean(x_i))  — exact pour AFT linéaire
shap_matrix = (sample_shap.values - X_mean_arr) * coef_vec   # shape (n_obs, n_feat)

# Créer un objet Explanation compatible shap.plots
shap_values = shap.Explanation(
    values=shap_matrix.astype(float),
    base_values=float(lambda_summary.loc['Intercept', 'coef']),
    data=sample_shap.values.astype(float),
    feature_names=ALL_FEATURES,
)
print("  SHAP calculé sur 1 000 tronçons échantillonnés.")

# ══════════════════════════════════════════════════════════════════════════
# 6. MATRICE DE CONFUSION (sous-ensemble recul ≥ 20 ans)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("6. MATRICE DE CONFUSION (tronçons avec recul ≥ 20 ans)")
print("=" * 70)
print("  Seuils testés sur risk_score_50ans pour classifier 'à risque élevé'")

# Sous-ensemble avec recul suffisant (observés ≥ 20 ans ou ayant eu l'événement)
recul_mask = (model_df[duration_col] >= 20) | (model_df[event_col] == 1)
model_recul = model_df[recul_mask].copy()
print(f"  Sous-ensemble recul ≥ 20 ans : {len(model_recul)} tronçons "
      f"({model_recul[event_col].mean()*100:.1f}% événements)")

# Prédire risk_score sur ce sous-ensemble
sf_50_recul  = waft.predict_survival_function(model_recul, times=[50])
p50_recul    = (1 - sf_50_recul.iloc[0].values)
y_true_recul = model_recul[event_col].values

# Chercher le meilleur seuil (maximise F1)
thresholds  = np.percentile(p50_recul, np.arange(50, 95, 5))
best_thresh = 0.5
best_f1     = 0.0
thresh_results = []
for thr in thresholds:
    y_pred = (p50_recul >= thr).astype(int)
    tp = ((y_pred == 1) & (y_true_recul == 1)).sum()
    fp = ((y_pred == 1) & (y_true_recul == 0)).sum()
    fn = ((y_pred == 0) & (y_true_recul == 1)).sum()
    tn = ((y_pred == 0) & (y_true_recul == 0)).sum()
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    thresh_results.append({'seuil': round(thr, 4), 'precision': round(prec, 3),
                            'rappel': round(rec, 3), 'f1': round(f1, 3),
                            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn})
    if f1 > best_f1:
        best_f1    = f1
        best_thresh = thr

print(pd.DataFrame(thresh_results).to_string(index=False))

# Matrice au meilleur seuil
y_pred_best = (p50_recul >= best_thresh).astype(int)
cm = confusion_matrix(y_true_recul, y_pred_best)
print(f"\n  Matrice de confusion au seuil optimal ({best_thresh:.4f}) :")
print(f"                     Prédit NON-RISQUE  Prédit À-RISQUE")
print(f"  Réel non-abandon  :      {cm[0,0]:6d}             {cm[0,1]:6d}")
print(f"  Réel abandon      :      {cm[1,0]:6d}             {cm[1,1]:6d}")
print()
print(classification_report(y_true_recul, y_pred_best,
                             target_names=['non-abandon', 'abandon']))

# ROC-AUC
fpr, tpr, _ = roc_curve(y_true_recul, p50_recul)
roc_auc_val = auc(fpr, tpr)
print(f"  ROC-AUC : {roc_auc_val:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. ANALYSE DES BIAIS D'ERREUR
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("7. ANALYSE DES BIAIS D'ERREUR")
print("=" * 70)

# Erreur individuelle = (p_pred_50 - event_bin) pour les observables
obs_idx    = recul_mask
p50_obs    = p50_recul
y_obs      = y_true_recul
residu_raw = p50_obs - y_obs   # positif = sur-estimation du risque

# Ajouter au dataframe
model_recul = model_recul.copy()
model_recul['residu']   = residu_raw
model_recul['p50_pred'] = p50_obs
model_recul['MAT_grp']  = df.loc[model_recul.index, 'MAT_grp'].values
model_recul['age']      = 2026 - df.loc[model_recul.index, 'DDP_year'].values

# Biais par matériau
print("\n  Biais moyen par matériau (résidu = p_prédit_50 − event_bin) :")
print("  Positif = le modèle SURESTIME le risque ; Négatif = SOUS-ESTIME")
mat_bias = model_recul.groupby('MAT_grp').agg(
    n=('residu', 'count'),
    biais_moyen=('residu', 'mean'),
    biais_std=('residu', 'std'),
    p50_moyen=('p50_pred', 'mean'),
    event_rate=(event_col, 'mean'),
).round(4)
print(mat_bias.to_string())

# Biais par tranche d'âge
print("\n  Biais moyen par tranche d'âge actuel :")
model_recul['tranche_age'] = pd.cut(
    model_recul['age'], bins=[0, 20, 40, 60, 80, 130],
    labels=['0-20', '20-40', '40-60', '60-80', '80+']
)
age_bias = model_recul.groupby('tranche_age', observed=True).agg(
    n=('residu', 'count'),
    biais_moyen=('residu', 'mean'),
    p50_moyen=('p50_pred', 'mean'),
    event_rate=(event_col, 'mean'),
).round(4)
print(age_bias.to_string())

# Biais par tranche de diamètre
print("\n  Biais moyen par diamètre :")
model_recul['tranche_diam'] = pd.cut(
    model_recul['DIAMETRE_imp'], bins=[0, 63, 110, 200, 1000],
    labels=['≤63mm', '63-110mm', '110-200mm', '>200mm']
)
diam_bias = model_recul.groupby('tranche_diam', observed=True).agg(
    n=('residu', 'count'),
    biais_moyen=('residu', 'mean'),
    event_rate=(event_col, 'mean'),
).round(4)
print(diam_bias.to_string())

# ══════════════════════════════════════════════════════════════════════════
# 8. MÉDIANE PRÉDITE vs KM PAR MATÉRIAU
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("8. MÉDIANE PRÉDITE vs KAPLAN-MEIER OBSERVÉE PAR MATÉRIAU")
print("=" * 70)

mats_check = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM', 'FTVI']
mat_comp   = []
for mat in mats_check:
    if mat == 'FT':
        mask_mat = df.loc[model_df.index, 'MAT_grp'].eq('FT')
    elif f'mat_{mat}' in model_df.columns:
        mask_mat = model_df[f'mat_{mat}'] == 1
    else:
        continue

    sub = model_df[mask_mat]
    if len(sub) < 100:
        continue

    kmf_m = KaplanMeierFitter()
    kmf_m.fit(sub[duration_col], event_observed=sub[event_col])
    km_med = kmf_m.median_survival_time_
    km_med_str = f"{km_med:.1f}" if not np.isinf(km_med) and not np.isnan(km_med) else ">120"

    wb_med = waft.predict_median(sub).median()
    wb_med_str = f"{wb_med:.1f}" if not np.isinf(wb_med) and not np.isnan(wb_med) else ">120"

    try:
        diff = wb_med - km_med
        diff_str = f"{diff:+.1f}"
    except Exception:
        diff_str = "N/A"

    mat_comp.append({'Matériau': mat, 'N': len(sub),
                     'Taux_abandon_%': f"{sub[event_col].mean()*100:.1f}",
                     'Médiane_KM': km_med_str, 'Médiane_Weibull': wb_med_str,
                     'Écart': diff_str})

print(pd.DataFrame(mat_comp).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# 9. FIGURES
# ══════════════════════════════════════════════════════════════════════════
print("\n── Génération des figures d'audit ──")

# ── Figure 1 : tableau de bord 6 panneaux ────────────────────────────────
fig = plt.figure(figsize=(22, 18))
gs  = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)
fig.suptitle(
    f'AUDIT DE FIABILITÉ — Weibull AFT (sans DDP_year / sans leakage)\n'
    f'C-index test = {c_test:.4f} | rho = {rho:.3f}',
    fontsize=14, fontweight='bold', y=0.99
)

# Panel A : Calibration globale
axA = fig.add_subplot(gs[0, 0])
calib_gdf = pd.DataFrame(calib_global)
axA.plot(calib_gdf['horizon'], calib_gdf['P_predit'], 'o-',
         color='#e74c3c', lw=2, ms=7, label='P prédit moyen')
axA.plot(calib_gdf['horizon'], calib_gdf['P_KM'], 's--',
         color='#3498db', lw=2, ms=7, label='P observé (KM)')
axA.set_xlabel('Horizon (années)', fontsize=10)
axA.set_ylabel('P(abandon)', fontsize=10)
axA.set_title('Calibration globale\nP(prédit) vs KM', fontsize=11, fontweight='bold')
axA.legend(fontsize=9)
axA.grid(alpha=0.3)

# Panel B : Calibration par quintile
axB = fig.add_subplot(gs[0, 1])
qdf = pd.DataFrame(calib_q)
axB.scatter(qdf['P_predit'], qdf['P_obs_KM'], s=200, c=np.arange(len(qdf)),
            cmap='RdYlGn_r', zorder=5, edgecolors='black', linewidths=0.8)
for _, row in qdf.iterrows():
    axB.annotate(f"Q{int(row['quintile'])}\n(n={row['n']:,})",
                 (row['P_predit'], row['P_obs_KM']),
                 fontsize=8, textcoords='offset points', xytext=(8, 4))
lim_max = max(qdf['P_predit'].max(), qdf['P_obs_KM'].max()) * 1.15
axB.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.5, label='Calibration parfaite')
axB.fill_between([0, lim_max], [0, lim_max * 0.8], [0, lim_max * 1.2],
                 alpha=0.08, color='green', label='±20%')
axB.set_xlabel('P(prédit) moyen', fontsize=10)
axB.set_ylabel('P(observé KM)', fontsize=10)
axB.set_title('Calibration par quintile\n(horizon 50 ans)', fontsize=11, fontweight='bold')
axB.legend(fontsize=8)
axB.grid(alpha=0.3)

# Panel C : Brier Skill Score par horizon
axC = fig.add_subplot(gs[0, 2])
brier_plot = brier_df.dropna(subset=['brier_skill'])
colors_bss = ['#2ecc71' if s > 0.1 else '#f39c12' if s > 0 else '#e74c3c'
              for s in brier_plot['brier_skill']]
axC.bar(brier_plot['horizon'], brier_plot['brier_skill'], color=colors_bss, alpha=0.85)
axC.axhline(y=0, color='black', lw=1.2, linestyle='--', label='Null model')
axC.axhline(y=0.1, color='green', lw=1, linestyle=':', alpha=0.7, label='Skill > 0.1')
axC.set_xlabel('Horizon (années)', fontsize=10)
axC.set_ylabel('Brier Skill Score', fontsize=10)
axC.set_title('Brier Skill Score par horizon\n(>0 = meilleur que la prévalence)', fontsize=11, fontweight='bold')
axC.legend(fontsize=8)
axC.grid(axis='y', alpha=0.3)

# Panel D : SHAP beeswarm / bar
axD = fig.add_subplot(gs[1, :2])
top_n   = 12
top_feat = mean_abs_shap.head(top_n).index.tolist()
top_vals = mean_abs_shap.head(top_n).values
colors_shap = ['#2ecc71' if coef_map.get(f, 0) > 0 else '#e74c3c' for f in top_feat]
axD.barh(range(top_n), top_vals[::-1], color=colors_shap[::-1], alpha=0.85)
axD.set_yticks(range(top_n))
axD.set_yticklabels(top_feat[::-1], fontsize=9)
axD.set_xlabel('|SHAP| moyen (contribution log-médiane)', fontsize=10)
axD.set_title('SHAP — Importance des features (vert = ↑ durée vie, rouge = ↓ durée vie)',
              fontsize=11, fontweight='bold')
axD.grid(axis='x', alpha=0.3)
import matplotlib.patches as mpatches
legend_shap = [
    mpatches.Patch(color='#2ecc71', alpha=0.85, label='↑ durée de vie (protecteur)'),
    mpatches.Patch(color='#e74c3c', alpha=0.85, label='↓ durée de vie (facteur de risque)'),
]
axD.legend(handles=legend_shap, fontsize=9)

# Panel E : ROC curve
axE = fig.add_subplot(gs[1, 2])
axE.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
axE.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Aléatoire')
axE.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axE.set_xlabel('Taux Faux Positifs', fontsize=10)
axE.set_ylabel('Taux Vrais Positifs', fontsize=10)
axE.set_title(f'Courbe ROC\n(sous-ensemble recul ≥ 20 ans, n={len(model_recul):,})',
              fontsize=11, fontweight='bold')
axE.legend(fontsize=9)
axE.grid(alpha=0.3)

# Panel F : Biais par matériau
axF = fig.add_subplot(gs[2, 0])
mat_b = mat_bias.sort_values('biais_moyen')
colors_b = ['#e74c3c' if v > 0.02 else '#3498db' if v < -0.02 else '#2ecc71'
            for v in mat_b['biais_moyen']]
axF.barh(mat_b.index, mat_b['biais_moyen'], color=colors_b, alpha=0.85)
axF.axvline(0, color='black', lw=1, linestyle='--')
axF.axvspan(-0.02, 0.02, alpha=0.1, color='green', label='Biais acceptable (±0.02)')
axF.set_xlabel('Biais moyen (p_prédit_50 − event_bin)', fontsize=10)
axF.set_title('Biais d\'erreur par matériau\n(rouge = surestimation, bleu = sous-estimation)',
              fontsize=11, fontweight='bold')
axF.legend(fontsize=8)
axF.grid(axis='x', alpha=0.3)

# Panel G : Biais par tranche d'âge
axG = fig.add_subplot(gs[2, 1])
age_b = age_bias.dropna()
colors_ag = ['#e74c3c' if v > 0.02 else '#3498db' if v < -0.02 else '#2ecc71'
             for v in age_b['biais_moyen']]
axG.bar(range(len(age_b)), age_b['biais_moyen'], color=colors_ag, alpha=0.85)
axG.axhline(0, color='black', lw=1, linestyle='--')
axG.axhspan(-0.02, 0.02, alpha=0.1, color='green')
axG.set_xticks(range(len(age_b)))
axG.set_xticklabels(age_b.index, fontsize=9)
axG.set_xlabel('Tranche d\'âge', fontsize=10)
axG.set_ylabel('Biais moyen', fontsize=10)
axG.set_title('Biais par tranche d\'âge\n(recul ≥ 20 ans)', fontsize=11, fontweight='bold')
axG.grid(axis='y', alpha=0.3)

# Panel H : Matrice de confusion
axH = fig.add_subplot(gs[2, 2])
im = axH.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar(im, ax=axH, fraction=0.046)
axH.set_xticks([0, 1])
axH.set_xticklabels(['Prédit\nNon-risque', 'Prédit\nÀ-risque'], fontsize=9)
axH.set_yticks([0, 1])
axH.set_yticklabels(['Réel\nNon-abandon', 'Réel\nAbandon'], fontsize=9)
axH.set_title(f'Matrice de confusion\n(seuil={best_thresh:.3f}, recul≥20 ans)',
              fontsize=11, fontweight='bold')
for i in range(2):
    for j in range(2):
        axH.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

plt.savefig('/home/user/EAuagent/figures/audit_fiabilite_complet.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure sauvegardée : figures/audit_fiabilite_complet.png")

# ── Figure 2 : SHAP Beeswarm détaillé ────────────────────────────────────
fig2, ax_shap = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title(f'SHAP Beeswarm — Contributions individuelles (1 000 tronçons)\n'
          f'Feature → impact sur log(durée de vie prédite)',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/audit_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure sauvegardée : figures/audit_shap_beeswarm.png")

# ── Figure 3 : KM observée vs Weibull par matériau ────────────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
axes3 = axes3.flatten()

for idx_m, mat in enumerate(mats_check):
    ax_m = axes3[idx_m]
    if mat == 'FT':
        mask_m = df.loc[model_df.index, 'MAT_grp'].eq('FT')
    elif f'mat_{mat}' in model_df.columns:
        mask_m = model_df[f'mat_{mat}'] == 1
    else:
        continue
    sub = model_df[mask_m]
    if len(sub) < 100:
        continue

    kmf_m = KaplanMeierFitter()
    kmf_m.fit(sub[duration_col], event_observed=sub[event_col])
    kmf_m.plot_survival_function(ax=ax_m, color='#3498db', lw=2, label='KM observé',
                                  ci_show=True, ci_alpha=0.15)

    sample_m = sub.sample(min(500, len(sub)), random_state=42)
    times_m  = np.linspace(0.1, 120, 200)
    sf_pred  = waft.predict_survival_function(sample_m, times=times_m)
    sf_mean  = sf_pred.mean(axis=1)
    ax_m.plot(times_m, sf_mean.values, color='#e74c3c', lw=2.5,
              linestyle='--', label='Weibull prédit')

    km_med_m  = kmf_m.median_survival_time_
    wb_med_m  = waft.predict_median(sub).median()
    title_med = (f"KM={km_med_m:.0f}a / Wb={wb_med_m:.0f}a"
                 if not np.isinf(km_med_m) and not np.isnan(km_med_m)
                 else f"KM=>120 / Wb={wb_med_m:.0f}a")

    ax_m.set_title(f'{mat} (n={len(sub)}, ev={sub[event_col].sum()})\n{title_med}',
                   fontsize=9, fontweight='bold')
    ax_m.set_xlabel('Durée (années)', fontsize=8)
    ax_m.set_ylabel('S(t)', fontsize=8)
    ax_m.set_ylim(0, 1.05)
    ax_m.set_xlim(0, 120)
    ax_m.legend(fontsize=7)
    ax_m.grid(alpha=0.3)

if len(mats_check) < 8:
    axes3[-1].axis('off')

plt.suptitle('Validation KM observée vs Weibull prédite par matériau\n'
             '(IC 95% KM en bleu clair | pointillés rouges = modèle)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/audit_km_vs_weibull.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Figure sauvegardée : figures/audit_km_vs_weibull.png")

# ══════════════════════════════════════════════════════════════════════════
# 10. RÉSUMÉ SYNTHÉTIQUE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RÉSUMÉ SYNTHÉTIQUE DE FIABILITÉ")
print("=" * 70)
print(f"""
  Discrimination (C-index)
  ─────────────────────────
    Train  : {c_train:.4f}
    Test   : {c_test:.4f}  ← honnête (hors-échantillon)
    Écart  : {c_train-c_test:.4f}  (< 0.02 → pas d'overfitting)
    Verdict: {'ACCEPTABLE (0.65-0.70)' if 0.65 <= c_test <= 0.70
              else 'BON (>0.70)' if c_test > 0.70 else 'FAIBLE (<0.65)'}

  Calibration (Brier Skill Score)
  ────────────────────────────────
    Horizon 30 ans : {brier_df[brier_df['horizon']==30]['brier_skill'].values[0]:.4f}
    Horizon 50 ans : {brier_df[brier_df['horizon']==50]['brier_skill'].values[0]:.4f}
    Horizon 70 ans : {brier_df[brier_df['horizon']==70]['brier_skill'].values[0]:.4f}
    Verdict: BSS > 0 = modèle meilleur que de prédire la prévalence brute

  Discrimination binaire (recul ≥ 20 ans)
  ─────────────────────────────────────────
    ROC-AUC : {roc_auc_val:.4f}
    Verdict: {'ACCEPTABLE (0.65-0.70)' if 0.65 <= roc_auc_val <= 0.70
              else 'BON (>0.70)' if roc_auc_val > 0.70 else 'FAIBLE (<0.65)'}

  Biais d'erreur
  ───────────────
    Matériau avec plus grand biais : {mat_bias['biais_moyen'].abs().idxmax()}
      (biais = {mat_bias.loc[mat_bias['biais_moyen'].abs().idxmax(), 'biais_moyen']:+.4f})
    Tranche d'âge avec plus grand biais : {age_bias['biais_moyen'].abs().idxmax()}
      (biais = {age_bias.loc[age_bias['biais_moyen'].abs().idxmax(), 'biais_moyen']:+.4f})

  Limites principales
  ─────────────────────
    1. Censure lourde : 84% des observations censurées → extrapolation
       au-delà de la durée d'observation (surtout tronçons récents)
    2. Biais de survivant : tronçons anciens encore en service sont les
       plus résistants → médiane KM peut surestimer la durée réelle
    3. Variables d'anomalie endogènes : nb_fuites_* reflètent l'état
       actuel mais pas forcément la cause de défaillance future
    4. Absence de validation temporelle (train avant 2015 / test après)
""")

print("✓ Audit terminé.")
