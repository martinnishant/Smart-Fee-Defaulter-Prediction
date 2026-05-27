import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor

df = pd.read_csv("data/processed/features.csv")

FEATURES = [
    "payment_ratio",
    "avg_delay_3m",
    "delay_streak",
    "total_arrears",
    "class_section_default_rate",
    "sibling_count",
    "transport_opted",
    "scholarship",
    "income_enc"
]

X = df[FEATURES]
y_cls = df["defaulted"]
y_reg = df["days_delayed"]

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

pos_weight = (y_cls == 0).sum() / (y_cls == 1).sum()

clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=pos_weight,
    eval_metric="auc",
    random_state=42
)

clf.fit(X_train, yc_train)

preds = clf.predict(X_test)
default_probs = clf.predict_proba(X_test)[:, 1]

print("--- Classifier Report ---")
print(classification_report(yc_test, preds))
print("ROC-AUC:", round(roc_auc_score(yc_test, default_probs), 4))

reg = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

reg.fit(X_train, yr_train)

delay_preds = reg.predict(X_test)
mae = mean_absolute_error(yr_test, delay_preds)

print("--- Regressor Report ---")
print("MAE:", round(mae, 2), "days")

df["default_prob"] = clf.predict_proba(X)[:, 1]
df["predicted_delay_days"] = reg.predict(X).clip(0)

risk_raw = (df["default_prob"] * 0.7) + ((df["predicted_delay_days"] / 180) * 0.3)
scaler = MinMaxScaler(feature_range=(0, 100))
df["risk_score"] = scaler.fit_transform(risk_raw.values.reshape(-1, 1)).round(1)

joblib.dump(clf, "outputs/clf_model.pkl")
joblib.dump(reg, "outputs/reg_model.pkl")

out = df[
    [
        "student_id",
        "student_name",
        "class",
        "section",
        "month",
        "fee_due",
        "fee_paid",
        "days_delayed",
        "default_prob",
        "predicted_delay_days",
        "risk_score"
    ]
].copy()

out.to_csv("outputs/risk_report.csv", index=False)

importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
importance.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")

print("Saved outputs/risk_report.csv")
print("Saved outputs/feature_importance.png")