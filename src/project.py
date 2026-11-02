import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score

# Ensure the output directory for visualization anchors exists
os.makedirs('images', exist_ok=True)

print("⚡ [Phase 1/5] Initializing Operational Threat Detection Pipeline...")
# 1. LOAD AND AUDIT THE DATASET
# Operational Context: Credit Card transactions data from Kaggle (highly imbalanced)
try:
    df = pd.read_csv('data/creditcard.csv')
except FileNotFoundError:
    print("❌ Error: 'data/creditcard.csv' not found. Creating a simulation context...")
    # Fallback to structural simulation if execution happens before local file mapping
    np.random.seed(42)
    n_samples = 5000
    simulated_data = {
        'Time': np.sort(np.random.uniform(0, 1000, n_samples)),
        'V1': np.random.normal(0, 1, n_samples),
        'V2': np.random.normal(0, 1, n_samples),
        'Amount': np.random.exponential(50, n_samples),
        'Class': np.random.choice([0, 1], size=n_samples, p=[0.998, 0.002])
    }
    df = pd.DataFrame(simulated_data)

# 2. EXPLORATORY DATA ANALYSIS (EDA) & CLASS BALANCE AUDITING
print("\n📊 [Phase 2/5] Running System Baseline Profiling...")
total_tx = len(df)
fraud_tx = df['Class'].sum()
normal_tx = total_tx - fraud_tx
fraud_percentage = (fraud_tx / total_tx) * 100

print(f"  • Baseline System Capacity: {total_tx} transactions processed.")
print(f"  • Flagged Operational Exceptions (Class 1 - Fraud): {fraud_tx} ({fraud_percentage:.3f}%)")
print(f"  • Routine Verified Traffic (Class 0 - Normal): {normal_tx} ({100 - fraud_percentage:.3f}%)")

# Save EDA Figure 1: Operational Imbalance Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title('Operational Threat Distribution Baseline')
plt.xlabel('Transaction Classification (0 = Normal, 1 = Cybercrime Anomaly)')
plt.ylabel('System Volume Count')
plt.xticks([0, 1], ['Normal Logs', 'Malicious Anomalies'])
plt.tight_layout()
plt.savefig('images/01_class_distribution.png')
plt.close()

# Save EDA Figure 2: Transaction Volume vs Cyber Risk Analysis
plt.figure(figsize=(7, 4))
sns.boxplot(x='Class', y='Amount', data=df, showfliers=False, palette='coolwarm')
plt.title('Operational Metrics: Financial Exposure by Traffic Class')
plt.xlabel('Traffic Classification')
plt.ylabel('Transaction Value Metric')
plt.xticks([0, 1], ['Normal Logs', 'Malicious Anomalies'])
plt.tight_layout()
plt.savefig('images/02_financial_exposure.png')
plt.close()

# 3. DATA PARTITIONING & ARCHITECTURE MODELING
print("\n⚙️ [Phase 3/5] Partitioning Workloads & Deploying Classifiers...")
X = df.drop(columns=['Class'])
y = df['Class']

# Segment into 80% historical patterns and 20% validation production simulation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Employing an ensemble Random Forest framework to trace multi-layered patterns
classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12, n_jobs=-1)
classifier.fit(X_train, y_train)

# 4. TRADITIONAL PERFORMANCE AUDIT
y_pred = classifier.predict(X_test)
print("\n📈 [Phase 4/5] Standard Production Validation Summary:")
print(classification_report(y_test, y_pred, target_names=['Normal logs', 'Threat Incident']))

# 5. OPERATIONAL OPTIMIZATION AND ALERT THRESHOLD ANALYSIS
print("\n🛠️ [Phase 5/5] Operational Review Optimization Strategy...")
# Rather than accepting defaults, we test custom alert barriers to maximize threat capturing
probabilities = classifier.predict_proba(X_test)[:, 1]
custom_barriers = [0.1, 0.2, 0.3, 0.5]

barrier_metrics = []
for barrier in custom_barriers:
    adjusted_predictions = (probabilities >= barrier).astype(int)
    rec = recall_score(y_test, adjusted_predictions)
    prec = precision_score(y_test, adjusted_predictions)
    
    # Calculate operational workload metrics
    cm = confusion_matrix(y_test, adjusted_predictions)
    tn, fp, fn, tp = cm.ravel()
    total_alerts = fp + tp
    
    barrier_metrics.append({
        'Barrier': barrier,
        'Threat Capture Rate (Recall)': rec * 100,
        'Alert Accuracy (Precision)': prec * 100,
        'Operational Queue Burden': total_alerts
    })

metrics_table = pd.DataFrame(barrier_metrics)
print(metrics_table.to_string(index=False, formatters={
    'Threat Capture Rate (Recall)': '{:,.2f}%'.format,
    'Alert Accuracy (Precision)': '{:,.2f}%'.format
}))

print("\n🚀 Core Engine Pipeline Complete. All analytics visualizations saved inside 'images/'.")
