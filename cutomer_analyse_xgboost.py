# RFM + OrderFreqPerMonth 特徵的強化版 XGBoost（含 Optuna 自動超參數優化）

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import optuna
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter

# 載入資料
file_path = 'ALL_feature_merged.csv'  # 替換為你的檔案路徑
df = pd.read_csv(file_path)

# 資料前處理
df['AvgShippingDelay'] = df['AvgShippingDelay'].apply(lambda x: max(x, 0))
df = df[df['label'].isin(['churn', 'partial churn', 'loyal'])]

# 編碼類別變數（label）
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# 只保留 RFM + OrderFreqPerMonth 特徵
rfm_features = ['Recency', 'Frequency', 'Monetary']
X = df[rfm_features]
customer_ids = df['會員編號']
y = df['label']

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 資料分割：訓練集與測試集
X_train, X_test, y_train, y_test, cust_train, cust_test = train_test_split(
    X_scaled, y, customer_ids, test_size=0.2, random_state=42, stratify=y
)

# SMOTE類型選擇
smote = BorderlineSMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("SMOTE 後數據分佈：", Counter(y_train_smote))

# Optuna進行超參數優改善


def objective(trial):
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
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


# 進行超參數改善
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("\n最佳超參數：", study.best_params)
best_params = study.best_params

# 使用最佳超參數的 XGBoost 模型
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

# 紀錄每個 Epoch 的 Log Loss
log_losses_train = []
log_losses_test = []

for epoch in range(1, xgb_model.get_params()['n_estimators'] + 1):
    xgb_model.set_params(n_estimators=epoch)
    xgb_model.fit(X_train_smote, y_train_smote, verbose=False)

    y_train_pred_proba = xgb_model.predict_proba(X_train_smote)
    y_test_pred_proba = xgb_model.predict_proba(X_test)
    train_loss = log_loss(y_train_smote, y_train_pred_proba)
    test_loss = log_loss(y_test, y_test_pred_proba)

    log_losses_train.append(train_loss)
    log_losses_test.append(test_loss)

# =繪製 Loss 圖表
plt.figure(figsize=(10, 6))
plt.plot(log_losses_train, label='Training Loss')
plt.plot(log_losses_test, label='Validation Loss')
plt.title(
    'XGBoost Training and Validation Log Loss (RFM + OrderFreqPerMonth with Optuna)')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True, linestyle='-', alpha=0.7)
plt.show()

# 模型評估
y_pred = xgb_model.predict(X_test)
print("\n最佳 XGBoost 模型 (RFM + OrderFreqPerMonth with Optuna) - 分類結果：")
print(classification_report(y_test, y_pred))

# =Stacking=模型（XGBoost + LightGBM）
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('lgbm', LGBMClassifier(random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X_train_smote, y_train_smote)
y_stack_pred = stacking_model.predict(X_test)
print("\nStacking 集成模型 (RFM + OrderFreqPerMonth with Optuna) - 分類報告：")
print(classification_report(y_test, y_stack_pred))

# 匯出預測結果檔案（含會員編號與轉換後的 label）

# 原始資料中取得會員編號（假設欄位名稱為 'CustomerID'，請依實際欄位名稱替換）
customer_ids = df.loc[X_test.index, '會員編號']  # 假設會員編號欄叫這個

# 預測類別數字轉為文字標籤
pred_labels_num = xgb_model.predict(X_test)
pred_labels_text = label_encoder.inverse_transform(pred_labels_num)

# 將文字標籤
label_map = {
    'churn': '0',
    'partial churn': '1',
    'loyal': '2'
}
pred_labels_mapped = [label_map[label] for label in pred_labels_text]

# 組成輸出 DataFrame
output_df = pd.DataFrame({
    'CustomerID': customer_ids.values,
    'Prediction': pred_labels_mapped
})

# 匯出為 CSV 檔案
output_df.to_csv('prediction_results.csv', index=False)
print("\n預測結果已儲存至 'prediction_results.csv'")
