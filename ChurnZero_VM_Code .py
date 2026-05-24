"""
ChurnZero 26 — Banking Customer Churn Prediction
Model: LightGBM + XGBoost Ensemble
CV AUC: ~1.0000 (5-fold StratifiedKFold)

Requirements:
    pip install pandas numpy scikit-learn lightgbm xgboost imbalanced-learn

Usage:
    python churnzero_model.py
    → Outputs: predictions.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('ChurnZero_dataset_v1.csv')
test  = pd.read_csv('ChurnZero_test_v1.csv')
print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Churn rate: {train['churn'].mean():.2%}")


# ─── 2. Feature Engineering ───────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Product breadth — number of banking products held
    product_cols = [
        'savings_account_flag', 'current_account_flag', 'credit_card_flag',
        'personal_loan_flag', 'home_loan_flag', 'auto_loan_flag',
        'fixed_deposit_flag', 'investment_product_flag',
        'insurance_product_flag', 'demat_account_flag'
    ]
    df['product_count'] = df[product_cols].sum(axis=1)

    # Digital engagement relative to branch usage
    df['digital_to_branch_ratio'] = df['total_digital_logins'] / (df['branch_visit_count'] + 1)

    # Complaint severity composite
    df['complaint_severity'] = (
        df['unresolved_complaint_count'] * 2 +
        df['escalation_count'] * 3
    )

    # Campaign engagement rate
    df['campaign_engage_rate'] = (
        df['campaign_response_count'] / (df['campaign_received_count'] + 1)
    )

    # Retention offer receptiveness
    df['retention_receptiveness'] = (
        df['retention_offer_accepted'] / (df['retention_offer_received'] + 1)
    )

    # Inactivity risk composite
    df['inactivity_risk'] = (
        df['account_inactive_days'] * 0.5 +
        df['last_login_days'] * 0.3 +
        df['last_contacted_days'] * 0.2
    )

    # Credit stress composite
    df['credit_stress'] = (
        df['credit_utilization_ratio'] * 0.4 +
        df['late_credit_card_payment_count'] * 0.3 +
        df['emi_payment_delay_count'] * 0.3
    )

    # Customer health composite
    df['customer_health_score'] = (
        df['nps_score'] * 0.5 +
        df['satisfaction_score'] * 0.5
    )

    return df


print("\nEngineering features...")
train = engineer_features(train)
test  = engineer_features(test)


# ─── 3. Encode Categorical Features ──────────────────────────
cat_cols = train.select_dtypes(include='object').columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns...")

le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))


# ─── 4. Handle Missing Values ─────────────────────────────────
# Only app_rating_given has missing values (~56% missing)
median_rating = train['app_rating_given'].median()
train['app_rating_given'] = train['app_rating_given'].fillna(median_rating)
test['app_rating_given']  = test['app_rating_given'].fillna(median_rating)


# ─── 5. Prepare Feature Matrix ────────────────────────────────
X      = train.drop(columns=['customer_id', 'churn'])
y      = train['churn']
X_test = test.drop(columns=['customer_id'])

print(f"\nFeature matrix: {X.shape}")
print(f"Class balance: {y.value_counts().to_dict()}")


# ─── 6. Cross-Validation ──────────────────────────────────────
print("\nRunning 5-fold Stratified CV...")

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=40,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,  # handles class imbalance
    random_state=42,
    verbose=-1
)

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    verbosity=0
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Manual fold-by-fold CV for transparency
fold_aucs_lgbm = []
fold_aucs_xgb  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    lgbm.fit(X_tr, y_tr)
    xgb.fit(X_tr, y_tr)

    lgbm_preds = lgbm.predict_proba(X_val)[:, 1]
    xgb_preds  = xgb.predict_proba(X_val)[:, 1]
    ens_preds  = 0.5 * lgbm_preds + 0.5 * xgb_preds

    auc_l = roc_auc_score(y_val, lgbm_preds)
    auc_x = roc_auc_score(y_val, xgb_preds)
    auc_e = roc_auc_score(y_val, ens_preds)

    fold_aucs_lgbm.append(auc_l)
    fold_aucs_xgb.append(auc_x)
    print(f"  Fold {fold+1}: LGBM={auc_l:.4f} | XGB={auc_x:.4f} | Ensemble={auc_e:.4f}")

print(f"\nLightGBM Mean AUC : {np.mean(fold_aucs_lgbm):.4f} ± {np.std(fold_aucs_lgbm):.4f}")
print(f"XGBoost  Mean AUC : {np.mean(fold_aucs_xgb):.4f} ± {np.std(fold_aucs_xgb):.4f}")


# ─── 7. Train Final Models on Full Dataset ────────────────────
print("\nTraining final models on full dataset...")
lgbm.fit(X, y)
xgb.fit(X, y)


# ─── 8. Generate Predictions ──────────────────────────────────
lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
xgb_probs  = xgb.predict_proba(X_test)[:, 1]
ensemble_probs = 0.5 * lgbm_probs + 0.5 * xgb_probs

preds_df = pd.DataFrame({
    'customer_id': test['customer_id'],
    'churn_probability': ensemble_probs.round(6),
    'churn_prediction': (ensemble_probs >= 0.5).astype(int)
})

preds_df.to_csv('predictions.csv', index=False)
print(f"\nPredictions saved to predictions.csv")
print(f"Predicted churn: {preds_df['churn_prediction'].sum()} / {len(preds_df)}")
print(f"Predicted churn rate: {preds_df['churn_prediction'].mean():.2%}")


# ─── 9. Feature Importance ────────────────────────────────────
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': lgbm.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features (LightGBM):")
print(feat_imp.head(15).to_string(index=False))

feat_imp.to_csv('feature_importance.csv', index=False)
print("\nFeature importances saved to feature_importance.csv")
print("\nDone!")
