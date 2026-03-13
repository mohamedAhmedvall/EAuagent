"""
ÉTAPE 3+4 — Entraînement et évaluation de modèles de prédiction de casse
Split temporel : CUTOFF=2020, train < 2015, test 2015-2020
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, brier_score_loss,
    classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json

# ============================================================
# 1. Charger et préparer les données
# ============================================================
df = pd.read_csv('data/dataset_model.csv')
print(f"Dataset : {len(df)} lignes")

# Features
cat_features = ['MAT_grp']
num_features = ['age', 'DIAMETRE_imp', 'LNG', 'nb_anomalies',
                'nb_fuites_signalees', 'nb_fuites_detectees', 'taux_anomalie_par_an']
targets = ['failure_1y', 'failure_3y', 'failure_5y']

# ============================================================
# 2. Split temporel : train (posé avant 2000) / test (posé 2000-2019)
# ============================================================
# On utilise DDP_year pour un split temporel stable
# Train : tronçons les plus anciens, Test : tronçons plus récents
# Cela simule "prédire l'avenir à partir du passé"
train_mask = df['DDP_year'] < 2000
test_mask = df['DDP_year'] >= 2000

X_train_raw = df.loc[train_mask, num_features + cat_features]
X_test_raw = df.loc[test_mask, num_features + cat_features]

print(f"Train : {len(X_train_raw)} lignes (posés avant 2000)")
print(f"Test  : {len(X_test_raw)} lignes (posés 2000+)")

# ============================================================
# 3. Preprocessor
# ============================================================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
])

# ============================================================
# 4. Définir les modèles
# ============================================================
def get_models(n_pos, n_total):
    """Retourne les modèles à tester, adaptés au déséquilibre."""
    scale_pos = (n_total - n_pos) / max(n_pos, 1)
    
    models = {
        'LogisticRegression_balanced': Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(
                class_weight='balanced', max_iter=1000, C=0.1, penalty='l1',
                solver='saga', random_state=42
            ))
        ]),
        'RandomForest_balanced': Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=10, min_samples_leaf=20,
                class_weight='balanced', random_state=42, n_jobs=-1
            ))
        ]),
        'XGBoost_balanced': Pipeline([
            ('prep', preprocessor),
            ('clf', XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos,
                eval_metric='aucpr', early_stopping_rounds=50,
                random_state=42, verbosity=0, n_jobs=-1
            ))
        ]),
        'MLP_balanced': Pipeline([
            ('prep', preprocessor),
            ('clf', MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation='relu',
                max_iter=500, early_stopping=True, validation_fraction=0.15,
                random_state=42, learning_rate='adaptive'
            ))
        ]),
    }
    
    # Versions SMOTE
    models['LogisticRegression_SMOTE'] = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', LogisticRegression(max_iter=1000, C=0.1, penalty='l1',
                                    solver='saga', random_state=42))
    ])
    models['XGBoost_SMOTE'] = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            eval_metric='aucpr', random_state=42, verbosity=0, n_jobs=-1
        ))
    ])
    
    return models


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Entraîne et évalue un modèle, retourne les métriques."""
    # Fit
    if 'XGBoost_balanced' in model_name and 'SMOTE' not in model_name:
        # XGBoost avec early stopping
        prep = preprocessor.fit(X_train)
        X_tr_p = prep.transform(X_train)
        X_te_p = prep.transform(X_test)
        clf = model.named_steps['clf']
        clf.fit(X_tr_p, y_train, eval_set=[(X_te_p, y_test)], verbose=False)
        y_prob = clf.predict_proba(X_te_p)[:, 1]
        y_pred = clf.predict(X_te_p)
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    
    # Métriques
    results = {
        'model': model_name,
        'ROC_AUC': roc_auc_score(y_test, y_prob),
        'PR_AUC': average_precision_score(y_test, y_prob),
        'F1_macro': f1_score(y_test, y_pred, average='macro'),
        'F1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'Precision_class1': precision_score(y_test, y_pred, zero_division=0),
        'Recall_class1': recall_score(y_test, y_pred, zero_division=0),
        'Brier_score': brier_score_loss(y_test, y_prob),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    results['TN'] = int(cm[0, 0])
    results['FP'] = int(cm[0, 1])
    results['FN'] = int(cm[1, 0])
    results['TP'] = int(cm[1, 1])
    
    return results


# ============================================================
# 5. Boucle d'entraînement : 6 modèles × 3 horizons
# ============================================================
all_results = []

for target in targets:
    y_train = df.loc[train_mask, target]
    y_test = df.loc[test_mask, target]
    
    n_pos_train = y_train.sum()
    n_pos_test = y_test.sum()
    print(f"\n{'='*60}")
    print(f"TARGET: {target}")
    print(f"  Train: {len(y_train)} lignes, {n_pos_train} positifs ({n_pos_train/len(y_train)*100:.3f}%)")
    print(f"  Test : {len(y_test)} lignes, {n_pos_test} positifs ({n_pos_test/len(y_test)*100:.3f}%)")
    
    if n_pos_train < 5:
        print(f"  SKIP — pas assez de positifs dans le train")
        continue
    
    models = get_models(n_pos_train, len(y_train))
    
    for name, model in models.items():
        try:
            res = evaluate_model(model, X_train_raw.copy(), y_train.copy(),
                                X_test_raw.copy(), y_test.copy(), name)
            res['target'] = target
            all_results.append(res)
            print(f"  {name:35s} ROC={res['ROC_AUC']:.4f}  PR={res['PR_AUC']:.4f}  "
                  f"F1m={res['F1_macro']:.4f}  Recall={res['Recall_class1']:.4f}  "
                  f"TP={res['TP']} FP={res['FP']} FN={res['FN']}")
        except Exception as e:
            print(f"  {name:35s} ERREUR: {e}")

# ============================================================
# 6. Tableau comparatif final
# ============================================================
results_df = pd.DataFrame(all_results)
results_df.to_csv('data/model_results.csv', index=False)

print(f"\n{'='*60}")
print("TABLEAU COMPARATIF FINAL (trié par ROC-AUC décroissant)")
print('='*60)

for target in targets:
    sub = results_df[results_df['target'] == target].sort_values('ROC_AUC', ascending=False)
    if len(sub) == 0:
        continue
    print(f"\n--- {target} ---")
    cols = ['model', 'ROC_AUC', 'PR_AUC', 'F1_macro', 'F1_weighted',
            'Precision_class1', 'Recall_class1', 'Brier_score', 'TP', 'FP', 'FN', 'TN']
    print(sub[cols].to_string(index=False))
    
    best = sub.iloc[0]
    print(f"\n  >>> MEILLEUR : {best['model']} (ROC-AUC={best['ROC_AUC']:.4f}, PR-AUC={best['PR_AUC']:.4f})")

print("\nSauvegardé : data/model_results.csv")
