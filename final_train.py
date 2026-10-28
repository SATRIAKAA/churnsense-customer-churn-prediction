
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# 1. Konfigurasi
DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = Path("models/churn_pipeline.joblib")
THRESHOLD_PATH = Path("models/churn_threshold.txt")
THRESHOLD = 0.35

# 2. Load dan bersihkan data
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.drop(columns=["customerID"])

y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

# 3. Pisahkan development set dan test set
# Test set tidak dipakai untuk memilih threshold.
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Tentukan preprocessing
numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# 5. Buat dan latih model final pada 80% data
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000)),
])

model.fit(X_dev, y_dev)

# 6. Evaluasi test set sekali pada threshold terpilih
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

print("\n=== FINAL TEST RESULTS ===")
print(f"Threshold : {THRESHOLD:.2f}")
print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
print(f"Precision : {precision_score(y_test, predictions, zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, predictions, zero_division=0):.4f}")
print(f"F1-score  : {f1_score(y_test, predictions, zero_division=0):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0,
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 7. Simpan model dan threshold
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)
THRESHOLD_PATH.write_text(str(THRESHOLD), encoding="utf-8")

print(f"\nModel tersimpan: {MODEL_PATH}")
print(f"Threshold tersimpan: {THRESHOLD_PATH}")
