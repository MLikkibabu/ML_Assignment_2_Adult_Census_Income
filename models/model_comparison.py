import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading dataset...")

df = pd.read_csv("adult.csv")

# Replace ? with missing values
df.replace("?", np.nan, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("Dataset Shape:", df.shape)

# =====================================================
# TARGET COLUMN
# =====================================================

target = "income"

X = df.drop(columns=[target])
y = df[target]

# Convert target to binary
y = y.map({
    "<=50K": 0,
    ">50K": 1,
    "<=50K.": 0,
    ">50K.": 1
})

# =====================================================
# IDENTIFY FEATURE TYPES
# =====================================================

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical Columns:")
print(list(categorical_cols))

print("\nNumerical Columns:")
print(list(numerical_cols))

# =====================================================
# PREPROCESSING
# =====================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# Save test data
test_data = X_test.copy()
test_data[target] = y_test

test_data.to_csv("test_data.csv", index=False)

# =====================================================
# MODELS
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = []

# =====================================================
# STANDARD MODELS
# =====================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    filename = (
        "models/"
        + name.lower().replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(pipeline, filename)

# =====================================================
# NAIVE BAYES
# =====================================================

print("\nTraining Naive Bayes...")

X_train_nb = preprocessor.fit_transform(X_train)
X_test_nb = preprocessor.transform(X_test)

if hasattr(X_train_nb, "toarray"):
    X_train_nb = X_train_nb.toarray()

if hasattr(X_test_nb, "toarray"):
    X_test_nb = X_test_nb.toarray()

nb_model = GaussianNB()

nb_model.fit(X_train_nb, y_train)

y_pred = nb_model.predict(X_test_nb)
y_prob = nb_model.predict_proba(X_test_nb)[:, 1]

results.append([
    "Naive Bayes",
    accuracy_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    matthews_corrcoef(y_test, y_pred)
])

joblib.dump(nb_model, "models/naive_bayes.pkl")

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4))

results_df.to_csv("results.csv", index=False)

# =====================================================
# VISUALIZATION
# =====================================================

metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]

plot_df = results_df.melt(
    id_vars="Model",
    value_vars=metrics
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=plot_df,
    x="Model",
    y="value",
    hue="variable"
)

plt.title("Model Performance Comparison")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "outputs/metrics_comparison.png",
    dpi=300
)

plt.show()

print("\nFiles Created Successfully:")
print("- results.csv")
print("- test_data.csv")
print("- outputs/metrics_comparison.png")
print("- model files in models folder")

print("\nAssignment Completed Successfully!")
