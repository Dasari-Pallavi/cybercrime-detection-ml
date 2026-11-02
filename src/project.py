# ============================================================
# Cybercrime / Fraud Detection - Operational Analyst Project
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_recall_curve,
                             average_precision_score, roc_auc_score)

os.makedirs("images", exist_ok=True)
sns.set_style("whitegrid")

# ---------- 1. LOAD DATA ----------
df = pd.read_csv("data/creditcard.csv")
print("Shape:", df.shape)
print(df.head())

# ---------- 2. DATA QUALITY CHECK ----------
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ---------- 3. EDA ----------
counts = df["Class"].value_counts()
fraud_pct = counts[1] / len(df) * 100
print(f"\nFraud cases: {counts[1]} ({fraud_pct:.3f}% of all transactions)")

plt.figure(figsize=(5, 4))
sns.countplot(x="Class", data=df)
plt.yscale("log")
plt.title("Class Distribution (0 = Normal, 1 = Fraud) - log scale")
plt.savefig("images/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False)
plt.title("Transaction Amount by Class")
plt.savefig("images/02_amount_by_class.png", dpi=150, bbox_inches="tight")
plt.close()

df["Hour"] = (df["Time"] // 3600) % 24
hourly = df.groupby("Hour")["Class"].agg(["sum", "count"])
hourly["fraud_rate_%"] = hourly["sum"] / hourly["count"] * 100
plt.figure(figsize=(9, 4))
hourly["fraud_rate_%"].plot(kind="bar")
plt.title("Fraud Rate (%) by Hour of Day")
plt.ylabel("Fraud rate %")
plt.savefig("images/03_fraud_by_hour.png", dpi=150, bbox_inches="tight")
plt.close()

corr = df.drop(columns=["Hour"]).corr()["Class"].drop("Class").sort_values()
plt.figure(figsize=(7, 8))
corr.plot(kind="barh")
plt.title("Correlation of Features with Fraud")
plt.savefig("images/04_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 4. DATA PREPARATION ----------
X = df.drop(columns=["Class", "Hour"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train = X_train.copy()
X_test = X_test.copy()
test_amount = X_test["Amount"].values      # raw amounts for cost analysis

scaler_time = StandardScaler()
scaler_amount = StandardScaler()
X_train["Time"] = scaler_time.fit_transform(X_train[["Time"]])
X_test["Time"] = scaler_time.transform(X_test[["Time"]])
X_train["Amount"] = scaler_amount.fit_transform(X_train[["Amount"]])
X_test["Amount"] = scaler_amount.transform(X_test[["Amount"]])

# ---------- 5. MODELS ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = proba
    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, digits=3))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC :", round(average_precision_score(y_test, proba), 4))

    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"images/05_cm_{name.replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(6, 5))
for name, proba in results.items():
    p, r, _ = precision_recall_curve(y_test, proba)
    plt.plot(r, p, label=name)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
plt.title("Precision-Recall Curve")
plt.savefig("images/06_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------- 6. OPERATIONAL ANALYSIS ----------
# Assumption: each manual review costs 5 currency units (analyst time).
# A missed fraud costs the full transaction amount.
REVIEW_COST = 5

proba = results["Random Forest"]
y_arr = y_test.values

rows = []
for t in np.linspace(0.05, 0.95, 91):
    pred = (proba >= t).astype(int)
    alerts = pred.sum()
    tp = ((pred == 1) & (y_arr == 1)).sum()
    fp = ((pred == 1) & (y_arr == 0)).sum()
    missed_loss = test_amount[(pred == 0) & (y_arr == 1)].sum()
    total_cost = alerts * REVIEW_COST + missed_loss
    rows.append({"threshold": t, "alerts": alerts, "caught_fraud": tp,
                 "false_alarms": fp, "missed_loss": missed_loss,
                 "total_cost": total_cost})

ops = pd.DataFrame(rows)
best = ops.loc[ops["total_cost"].idxmin()]
no_model_cost = test_amount[y_arr == 1].sum()

print("\n===== OPERATIONAL RESULT =====")
print(best)
print(f"Cost with no detection : {no_model_cost:,.2f}")
print(f"Cost with model        : {best['total_cost']:,.2f}")
print(f"Saving                 : {no_model_cost - best['total_cost']:,.2f}")

plt.figure(figsize=(8, 4))
plt.plot(ops["threshold"], ops["total_cost"])
plt.axvline(best["threshold"], color="red", linestyle="--",
            label=f"Best threshold = {best['threshold']:.2f}")
plt.xlabel("Alert threshold"); plt.ylabel("Total operational cost")
plt.title("Choosing the Threshold by Total Cost"); plt.legend()
plt.savefig("images/07_threshold_cost.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(ops["alerts"], ops["caught_fraud"])
plt.xlabel("Number of alerts reviewed (analyst workload)")
plt.ylabel("Fraud cases caught")
plt.title("Workload vs Fraud Caught")
plt.savefig("images/08_workload_vs_caught.png", dpi=150, bbox_inches="tight")
plt.close()

ops.to_csv("data/threshold_analysis.csv", index=False)
print("\nDone. Charts saved in /images")