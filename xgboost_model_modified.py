# 將修正後的程式碼（最後輸出為 .xlsx 格式）寫入新的 Python 檔案

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
import optuna
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

# 1. 載入資料
file_path = 'ALL_feature_merged.csv'  # 替換為你的檔案路徑
df = pd.read_csv(file_path)

# 2. 資料前處理
df['AvgShippingDelay'] = df['AvgShippingDelay'].apply(lambda x: max(x, 0))
df = df[df['label'].isin(['churn', 'partial churn', 'loyal'])]

# 3. 編碼類別變數（label）
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 4. 只保留 RFM 特徵 + 會員編號
rfm_features = ['Recency', 'Frequency', 'Monetary']
X = df[rfm_features]
customer_ids = df['會員編號']
y = df['label']

# 5. 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. 資料分割
X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X_scaled, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

# 7. SMOTE
smote = BorderlineSMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_smote, y_train_smote = X, y

# 8. Optuna 調參


def objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 6)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 250)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    alpha = trial.suggest_float('alpha', 0, 10)
    lambda_ = trial.suggest_float('lambda', 0, 10)

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=alpha,
        reg_lambda=lambda_,
        random_state=42
    )

    model.fit(X_train_smote, y_train_smote)
    preds_proba = model.predict_proba(X_test)
    return log_loss(y_test, preds_proba)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("\\n最佳超參數：", study.best_params)
best_params = study.best_params

# 10. 訓練最佳模型
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['alpha'],
    reg_lambda=best_params['lambda'],
    random_state=42
)
xgb_model.fit(X_train_smote, y_train_smote)

# 11. 評估與輸出報告
y_pred = xgb_model.predict(X_test)
print("\\n分類報告：")
print(classification_report(y_test, y_pred))

# 12. 預測與輸出至 XLSX 檔案
pred_labels_num = xgb_model.predict(X_test)
pred_labels_text = label_encoder.inverse_transform(pred_labels_num)

label_map = {
    'churn': 'churn',
    'partial churn': 'partial churn',
    'loyal': 'loyal'
}
pred_labels_mapped = [label_map[label] for label in pred_labels_text]

output_df = pd.DataFrame({
    '會員編號': cust_test.values,
    'label': pred_labels_mapped
})

output_df.to_excel('prediction_results.xlsx', index=False)
df_test = pd.read_excel('test.xlsx')

df_test['會員編號'] = df_test['會員編號'].astype('Int64').astype(str)
output_df['會員編號'] = output_df['會員編號'].astype('Int64').astype(str)
# inner join 預測結果，只保留 test.xlsx 中的會員
filtered_output = pd.merge(
    df_test[['會員編號']], output_df[['會員編號', 'label']], on='會員編號', how='left')
# 輸出最終檔案
filtered_output.to_excel('filtered_prediction_results.xlsx', index=False)
print("\n✅ 僅包含 test.xlsx 中會員的預測結果已儲存為 'filtered_prediction_results.xlsx'")
