"""
Predictive Maintenance
"""
import json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1. LOAD
df = pd.read_csv('ai4i2020.csv')
print('=== DATASET ===')
print('Shape:', df.shape)
print('Failure rate: %.2f%%' % (100 * df['Machine failure'].mean()))

# 2. PREPROCESS
# drop overall 'Machine failure' label; TWF/HDF/PWF/OSF/RNF directly encode the answer
drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
data = df.drop(columns=drop_cols).copy()

# Feature engineering: domain-informed features
data['Temp diff [K]'] = data['Process temperature [K]'] - data['Air temperature [K]']
data['Power [W]'] = data['Torque [Nm]'] * data['Rotational speed [rpm]'] * 2 * np.pi / 60.0
data['Wear*Torque'] = data['Tool wear [min]'] * data['Torque [Nm]']

# Encode Type L/M/H -> 0/1/2 (ordinal: product quality)
data['Type'] = data['Type'].map({'L': 0, 'M': 1, 'H': 2})

y = data['Machine failure'].values
X = data.drop(columns=['Machine failure'])
feature_names = X.columns.tolist()
print('Features used:', feature_names)

# Stratified split: 70% train, 15% val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE)
print(f'Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}')

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# 3. BASELINE
baseline = DummyClassifier(strategy='most_frequent').fit(X_train_s, y_train)
base_pred = baseline.predict(X_test_s)
print('\n=== BASELINE (most_frequent) ===')
print('Accuracy: %.4f | F1: %.4f' % (
    accuracy_score(y_test, base_pred), f1_score(y_test, base_pred, zero_division=0)))

# 4. TUNING dan CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

# Random Forest
print('\n=== RANDOM FOREST: GridSearchCV ===')
rf_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced'],
}
rf_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_grid, scoring='f1', cv=cv, n_jobs=-1)
rf_search.fit(X_train_s, y_train)
rf_best = rf_search.best_estimator_
print('Best params:', rf_search.best_params_)
print('Best CV F1: %.4f' % rf_search.best_score_)

# XGBoost 
print('\n=== XGBOOST: GridSearchCV ===')
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
}
xgb_search = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                  scale_pos_weight=scale_pos),
    xgb_grid, scoring='f1', cv=cv, n_jobs=-1)
xgb_search.fit(X_train_s, y_train)
xgb_best = xgb_search.best_estimator_
print('Best params:', xgb_search.best_params_)
print('Best CV F1: %.4f' % xgb_search.best_score_)

# 5. EVALUATE
def evaluate(name, model, Xt, yt):
    pred = model.predict(Xt)
    proba = model.predict_proba(Xt)[:, 1]
    m = {
        'accuracy': accuracy_score(yt, pred),
        'precision': precision_score(yt, pred, zero_division=0),
        'recall': recall_score(yt, pred, zero_division=0),
        'f1': f1_score(yt, pred, zero_division=0),
        'roc_auc': roc_auc_score(yt, proba),
    }
    cv_f1 = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='f1')
    m['cv_f1_mean'] = cv_f1.mean()
    m['cv_f1_std'] = cv_f1.std()
    print(f'\n=== {name} (TEST SET) ===')
    for k, v in m.items():
        print(f'  {k:12s}: {v:.4f}')
    print('  Confusion matrix:\n', confusion_matrix(yt, pred))
    print(classification_report(yt, pred, zero_division=0, target_names=['OK','Failure']))
    return m, pred, proba

results['Random Forest'], rf_pred, rf_proba = evaluate('RANDOM FOREST', rf_best, X_test_s, y_test)
results['XGBoost'], xgb_pred, xgb_proba = evaluate('XGBOOST', xgb_best, X_test_s, y_test)

# Pick winner by F1
winner = max(results, key=lambda k: results[k]['f1'])
best_model = rf_best if winner == 'Random Forest' else xgb_best
best_pred = rf_pred if winner == 'Random Forest' else xgb_pred
print(f'\n>>> BEST MODEL: {winner} (F1={results[winner]["f1"]:.4f})')

# 6. PLOTS
# Confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for i, (nm, pr) in enumerate([('Random Forest', rf_pred), ('XGBoost', xgb_pred)]):
    cm = confusion_matrix(y_test, pr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[i],
                xticklabels=['OK','Failure'], yticklabels=['OK','Failure'])
    ax[i].set_title(f'{nm} — Confusion Matrix'); ax[i].set_ylabel('True'); ax[i].set_xlabel('Predicted')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=130); plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for nm, pb in [('Random Forest', rf_proba), ('XGBoost', xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, pb)
    plt.plot(fpr, tpr, label=f'{nm} (AUC={roc_auc_score(y_test, pb):.3f})')
plt.plot([0,1],[0,1],'k--',alpha=0.4)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves'); plt.legend(); plt.tight_layout()
plt.savefig('roc_curves.png', dpi=130); plt.close()

# Feature importance (winner)
imp = pd.Series(best_model.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(7, 4))
imp.plot.barh(color='teal')
plt.title(f'Feature Importance — {winner}'); plt.tight_layout()
plt.savefig('feature_importance.png', dpi=130); plt.close()

# 7. SAVE
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
meta = {
    'winner': winner,
    'feature_names': feature_names,
    'type_mapping': {'L': 0, 'M': 1, 'H': 2},
    'metrics': {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in results.items()},
    'baseline_f1': round(float(f1_score(y_test, base_pred, zero_division=0)), 4),
}
with open('metrics.json', 'w') as f:
    json.dump(meta, f, indent=2)
print('\nSaved: model.pkl, scaler.pkl, metrics.json, *.png')
