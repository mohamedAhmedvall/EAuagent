"""
ÉTAPE 9 — Rapport final : planche de synthèse
================================================
Génère une planche récapitulative multi-panneaux avec tous les résultats clés
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from lifelines import WeibullAFTFitter

print("=" * 60)
print("ÉTAPE 9 — RAPPORT FINAL & PLANCHE DE SYNTHÈSE")
print("=" * 60)

# ── Charger les résultats ─────────────────────────────────────
cox_summary = pd.read_csv('/home/user/EAuagent/models/cox_ph_summary_B.csv', index_col=0)
weibull_summary = pd.read_csv('/home/user/EAuagent/models/weibull_aft_summary_B.csv', index_col=[0, 1])
scoring = pd.read_csv('/home/user/EAuagent/models/scoring_troncons.csv')
comparison = pd.read_csv('/home/user/EAuagent/models/comparaison_modeles.csv')
causes_hr = pd.read_csv('/home/user/EAuagent/models/comparaison_causes_HR.csv')

# ══════════════════════════════════════════════════════════════
# PLANCHE 1 : SYNTHÈSE MODÉLISATION
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)

fig.suptitle('ANALYSE DE SURVIE DU RÉSEAU D\'EAU — SYNTHÈSE DES MODÈLES\n'
             'Base : 194 754 tronçons | 31 152 abandons (16%) | Modèle retenu : Weibull AFT',
             fontsize=16, fontweight='bold', y=0.98)

# ── Panel 1 : Tableau comparatif des modèles ─────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
table_data = [
    ['Cox PH', '-', '-', '0.586'],
    ['Weibull AFT', '370 974', '370 960', '0.750'],
    ['Log-Normal AFT', '379 979', '379 965', '0.748'],
    ['Log-Logistique AFT', '373 046', '373 032', '0.753'],
]
table = ax1.table(cellText=table_data,
                  colLabels=['Modèle', 'AIC', 'BIC', 'C-index'],
                  cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Colorier la ligne Weibull (meilleur AIC)
for j in range(4):
    table[2, j].set_facecolor('#d5f5e3')
    table[2, j].set_text_props(fontweight='bold')

ax1.set_title('Comparaison des modèles', fontsize=12, fontweight='bold', pad=20)

# ── Panel 2 : Paramètre de forme Weibull ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')

info_text = (
    "MODÈLE RETENU : Weibull AFT\n\n"
    "Paramètre de forme (ρ) = 2.78\n"
    "  → ρ > 1 : risque CROISSANT avec l'âge\n"
    "  → Confirmation du vieillissement du réseau\n\n"
    "C-index = 0.750\n"
    "  → Bonne capacité discriminante\n\n"
    "Variables les plus influentes :\n"
    "  ↑ Risque : DDP récent, FTVI, FTG, PEHD\n"
    "  ↓ Risque : BTM, LNG_log, flux circulation\n\n"
    "Top 10% à risque : 19 475 tronçons\n"
    "  60% FTVI, 31% FT, 6% PEHD\n"
    "  95% posés après 2010"
)
ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#eaf2f8', alpha=0.8))
ax2.set_title('Résumé du modèle retenu', fontsize=12, fontweight='bold', pad=20)

# ── Panel 3 : Forest plot Cox HR (simplifié) ─────────────────
ax3 = fig.add_subplot(gs[1, 0])

hr = cox_summary[['exp(coef)', 'p']].copy()
hr.columns = ['HR', 'p']
# Garder les significatifs
hr_sig = hr[hr['p'] < 0.05].sort_values('HR')

colors = ['#e74c3c' if p < 0.001 else '#f39c12' for p in hr_sig['p']]
ax3.barh(range(len(hr_sig)), hr_sig['HR'].values, color=colors, alpha=0.8)
ax3.set_yticks(range(len(hr_sig)))
ax3.set_yticklabels(hr_sig.index, fontsize=8)
ax3.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Hazard Ratio', fontsize=10)
ax3.set_title('Cox PH — Hazard Ratios significatifs\n(réf. = FT)', fontsize=11, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# ── Panel 4 : Weibull coefficients ───────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

lambda_coefs = weibull_summary.loc['lambda_']
lambda_coefs = lambda_coefs[lambda_coefs.index != 'Intercept']
coefs_sig = lambda_coefs[lambda_coefs['p'] < 0.05]['coef'].sort_values()

colors_w = ['#e74c3c' if lambda_coefs.loc[idx, 'p'] < 0.001 else '#f39c12'
            for idx in coefs_sig.index]
ax4.barh(range(len(coefs_sig)), coefs_sig.values, color=colors_w, alpha=0.8)
ax4.set_yticks(range(len(coefs_sig)))
ax4.set_yticklabels(coefs_sig.index, fontsize=8)
ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax4.set_xlabel('Coefficient AFT (>0 = durée plus longue)', fontsize=10)
ax4.set_title('Weibull AFT — Coefficients significatifs\n(réf. = FT)', fontsize=11, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# ── Panel 5 : Validation déciles ─────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])

decile_data = scoring.groupby('decile_risque').agg(
    taux_abandon=('event_bin', 'mean'),
    score_moyen=('risk_score_50ans', 'mean'),
).reset_index()

colors_d = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(decile_data)))
bars = ax5.bar(decile_data['decile_risque'], decile_data['taux_abandon'] * 100,
               color=colors_d, edgecolor='white')
for bar, val in zip(bars, decile_data['taux_abandon'] * 100):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=8)

ax5.set_xlabel('Décile de risque', fontsize=10)
ax5.set_ylabel('Taux d\'abandon réel (%)', fontsize=10)
ax5.set_title('Validation — Taux d\'abandon par décile\n(Weibull AFT)', fontsize=11, fontweight='bold')
ax5.set_xticks(range(1, 11))
ax5.grid(axis='y', alpha=0.3)

# ── Panel 6 : Durées médianes par matériau ───────────────────
ax6 = fig.add_subplot(gs[2, 1])

mat_medians = scoring.groupby('MAT_grp')['duree_mediane_pred'].median()
mat_medians = mat_medians[mat_medians.index.isin(['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM', 'FTVI'])]
mat_medians = mat_medians.sort_values()

colors_bar = ['#e74c3c' if v < 40 else '#f39c12' if v < 60 else '#2ecc71' for v in mat_medians.values]
ax6.barh(mat_medians.index, mat_medians.values, color=colors_bar, alpha=0.8)
for j, (mat, val) in enumerate(mat_medians.items()):
    label = f'{val:.0f} ans' if not np.isinf(val) and val < 200 else '> 120 ans'
    ax6.text(min(val, 130) + 1, j, label, va='center', fontsize=9)

ax6.set_xlabel('Durée médiane prédite (années)', fontsize=10)
ax6.set_title('Durée médiane de survie par matériau\n(Weibull AFT)', fontsize=11, fontweight='bold')
ax6.set_xlim(0, 150)
ax6.grid(axis='x', alpha=0.3)

# ── Panel 7 : Heatmap matériau × décile ──────────────────────
ax7 = fig.add_subplot(gs[3, 0])

mats_main = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM', 'FTVI']
heatmap_data = []
for mat in mats_main:
    row = []
    for d in range(1, 11):
        mask = (scoring['MAT_grp'] == mat) & (scoring['decile_risque'] == d)
        row.append(mask.sum())
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=mats_main, columns=range(1, 11))
heatmap_pct = heatmap_df.div(heatmap_df.sum(axis=1), axis=0) * 100

im = ax7.imshow(heatmap_pct.values, cmap='RdYlGn_r', aspect='auto')
ax7.set_xticks(range(10))
ax7.set_xticklabels(range(1, 11))
ax7.set_yticks(range(len(mats_main)))
ax7.set_yticklabels(mats_main, fontsize=9)
ax7.set_xlabel('Décile de risque', fontsize=10)
ax7.set_title('Matériau × Décile de risque (%)', fontsize=11, fontweight='bold')

for i in range(len(mats_main)):
    for j in range(10):
        val = heatmap_pct.values[i, j]
        if val >= 1:
            ax7.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7,
                     color='white' if val > 25 else 'black')

plt.colorbar(im, ax=ax7, label='%', shrink=0.8)

# ── Panel 8 : Recommandations ────────────────────────────────
ax8 = fig.add_subplot(gs[3, 1])
ax8.axis('off')

recommandations = (
    "RECOMMANDATIONS OPÉRATIONNELLES\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    "1. PRIORITÉ HAUTE — Matériaux à surveiller :\n"
    "   • FTVI : durée médiane 27 ans, 95% en décile 10\n"
    "   • PEHD : durée médiane 40 ans, concentration déciles 7-9\n"
    "   • FT ancien (<1960) : risque croissant avec l'âge\n\n"
    "2. MATÉRIAUX RÉSILIENTS :\n"
    "   • BTM : durée médiane 95 ans, 45% en décile 1\n"
    "   • FTG : durée médiane 87 ans\n"
    "   • PVC/POLY : durées médianes 74-75 ans\n\n"
    "3. FACTEURS D'ACCÉLÉRATION :\n"
    "   • Taux d'anomalie/an (signal le plus fort)\n"
    "   • Fuites détectées → risque correctif ×258\n"
    "   • Longueur du tronçon (HR=3 dans Cox)\n\n"
    "4. STRATÉGIE DE RENOUVELLEMENT :\n"
    "   • Cibler les 19 475 tronçons top 10%\n"
    "   • Programmer les FTVI posés 2010-2020\n"
    "   • Renforcer la surveillance des PEHD récents\n"
    "   • Maintenir les BTM/FTG (longue durée de vie)"
)

ax8.text(0.05, 0.95, recommandations, transform=ax8.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#fef9e7', alpha=0.9))

plt.savefig('/home/user/EAuagent/figures/etape9_planche_synthese.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : figures/etape9_planche_synthese.png")

# ══════════════════════════════════════════════════════════════
# PLANCHE 2 : RISQUES COMPÉTITIFS
# ══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('RISQUES COMPÉTITIFS — PRÉVENTIF vs CORRECTIF\n'
             '(cause-specific Cox, Dataset A : 27 653 préventifs + 3 499 correctifs)',
             fontsize=14, fontweight='bold')

# Panel 1 : HR comparés
ax = axes[0, 0]
causes = causes_hr.sort_values('ratio_HR', ascending=True)
# Garder seulement les matériaux et variables clés
key_vars = ['mat_FTG', 'mat_PEHD', 'mat_BTM', 'mat_POLY', 'mat_FTVI',
            'LNG_log', 'nb_anomalies', 'nb_fuites_detectees', 'DDP_year',
            'DT_FLUX_CIRCULATION_imp', 'taux_anomalie_par_an']
causes_key = causes[causes['Variable'].isin(key_vars)]

y = range(len(causes_key))
ax.barh([yi - 0.15 for yi in y], np.log2(causes_key['HR_preventif']),
        height=0.3, color='#f39c12', alpha=0.8, label='Préventif')
ax.barh([yi + 0.15 for yi in y], np.log2(causes_key['HR_correctif'].clip(1e-50)),
        height=0.3, color='#e74c3c', alpha=0.8, label='Correctif')
ax.set_yticks(y)
ax.set_yticklabels(causes_key['Variable'].values, fontsize=8)
ax.axvline(x=0, color='black', linestyle='--')
ax.set_xlabel('log₂(HR)', fontsize=10)
ax.set_title('HR par cause (log₂)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='x', alpha=0.3)

# Panel 2 : Variables à effet opposé
ax = axes[0, 1]
# Identifier les variables avec effets divergents
divergent = causes_hr[
    ((causes_hr['HR_preventif'] > 1) & (causes_hr['HR_correctif'] < 1)) |
    ((causes_hr['HR_preventif'] < 1) & (causes_hr['HR_correctif'] > 1))
].copy()
divergent = divergent[divergent['Variable'].isin(key_vars)]

if len(divergent) > 0:
    y = range(len(divergent))
    ax.barh([yi - 0.15 for yi in y], divergent['HR_preventif'] - 1,
            height=0.3, color='#f39c12', alpha=0.8, label='Préventif (HR-1)')
    ax.barh([yi + 0.15 for yi in y], divergent['HR_correctif'] - 1,
            height=0.3, color='#e74c3c', alpha=0.8, label='Correctif (HR-1)')
    ax.set_yticks(y)
    ax.set_yticklabels(divergent['Variable'].values, fontsize=8)
    ax.axvline(x=0, color='black', linestyle='--')
    ax.set_xlabel('HR - 1 (>0 = risque, <0 = protection)', fontsize=10)
    ax.legend(fontsize=9)
ax.set_title('Variables à effet OPPOSÉ\nselon la cause', fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Panel 3 : Distribution des événements par matériau
ax = axes[1, 0]
df_a = pd.read_csv('/home/user/EAuagent/data/dataset_A_competitif.csv')
mats_plot = ['FT', 'FTG', 'POLY', 'PEHD', 'PVC', 'BTM']
prev_counts = [((df_a['MAT_grp'] == m) & (df_a['event_code'] == 1)).sum() for m in mats_plot]
corr_counts = [((df_a['MAT_grp'] == m) & (df_a['event_code'] == 2)).sum() for m in mats_plot]

x = np.arange(len(mats_plot))
width = 0.35
ax.bar(x - width/2, prev_counts, width, label='Préventif', color='#f39c12', alpha=0.8)
ax.bar(x + width/2, corr_counts, width, label='Correctif', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(mats_plot)
ax.set_ylabel('Nombre d\'abandons', fontsize=10)
ax.set_title('Nombre d\'abandons par matériau et cause', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Panel 4 : Ratio correctif/total par matériau
ax = axes[1, 1]
ratios = [c / (p + c) * 100 if (p + c) > 0 else 0 for p, c in zip(prev_counts, corr_counts)]
colors_ratio = ['#e74c3c' if r > 15 else '#f39c12' if r > 10 else '#2ecc71' for r in ratios]
bars = ax.bar(mats_plot, ratios, color=colors_ratio, alpha=0.8, edgecolor='white')
for bar, val in zip(bars, ratios):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontsize=9)
ax.set_ylabel('% correctif parmi les abandons', fontsize=10)
ax.set_title('Part du correctif dans les abandons\npar matériau', fontsize=11, fontweight='bold')
ax.axhline(y=df_a['event_code'].eq(2).sum() / df_a['event_any'].sum() * 100,
           color='black', linestyle='--', alpha=0.5, label='Moyenne')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/EAuagent/figures/etape9_planche_competitif.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure sauvegardée : figures/etape9_planche_competitif.png")

# ══════════════════════════════════════════════════════════════
# RAPPORT TEXTE
# ══════════════════════════════════════════════════════════════

rapport = """
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
"""

with open('/home/user/EAuagent/resultats.md', 'w') as f:
    f.write(rapport)
print("\nRapport sauvegardé : resultats.md")

print("\n✓ Étape 9 terminée. Toutes les analyses sont complètes.")
