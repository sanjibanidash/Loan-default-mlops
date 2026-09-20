import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# Train test split
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# SMOTE
from imblearn.over_sampling import SMOTE


# ==================================================
# MLFLOW CONFIGURATION
# ==================================================

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "Loan_Default_Prediction"
)


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("artifacts/train.csv")

# Save reference data
df.to_csv(
    "artifacts/reference_data.csv",
    index=False
)

logging.info("Dataset loaded successfully")


# ==================================================
# INPUT AND TARGET
# ==================================================

X = df.drop(
    columns=["TARGET", "SK_ID_CURR"]
)

y = df["TARGET"]


# ==================================================
# ENCODING CATEGORICAL COLUMNS
# ==================================================

X = pd.get_dummies(
    X,
    drop_first=True
)

# Clean column names
X.columns = X.columns.str.replace(
    '[^A-Za-z0-9_]+',
    '_',
    regex=True
)

# Save model columns
pickle.dump(
    X.columns.tolist(),
    open("artifacts/model_columns.pkl", "wb")
)

print("Encoding completed")

print("Model columns saved successfully")


# ==================================================
# HANDLE MISSING VALUES
# ==================================================

X = X.fillna(X.median())

print("Missing values handled")


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed")


# ==================================================
# APPLY SMOTE
# ==================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("SMOTE applied successfully")

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


# ==================================================
# MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=500
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "AdaBoost": AdaBoostClassifier(),

    "KNN": KNeighborsClassifier(),

    "XGBoost": XGBClassifier(
        n_estimators=50,
        scale_pos_weight=10,
        random_state=42,
        eval_metric="logloss"
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "CatBoost": CatBoostClassifier(
        verbose=0,
        auto_class_weights="Balanced",
        random_state=42
    )

}


# ==================================================
# BEST MODEL TRACKING
# ==================================================

best_model = None

best_model_name = ""

best_roc_auc = 0


# ==================================================
# TRAIN AND EVALUATE
# ==================================================

for name, model in models.items():

    print("\n" + "=" * 60)

    print(f"MODEL: {name}")

    print("=" * 60)

    with mlflow.start_run(run_name=name):

        # ==========================================
        # TRAIN MODEL
        # ==========================================

        model.fit(
            X_train_smote,
            y_train_smote
        )

        # ==========================================
        # PREDICTIONS
        # ==========================================

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ==========================================
        # METRICS
        # ==========================================

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        precision = report["1"]["precision"]

        recall = report["1"]["recall"]

        f1_score_value = report["1"]["f1-score"]

        print(f"\nAccuracy: {accuracy}")

        print(f"ROC-AUC Score: {roc_auc}")

        print(f"Precision: {precision}")

        print(f"Recall: {recall}")

        print(f"F1 Score: {f1_score_value}")

        # ==========================================
        # CONFUSION MATRIX
        # ==========================================

        cm = confusion_matrix(
            y_test,
            predictions
        )

        print("\nConfusion Matrix:")

        print(cm)

        plt.figure(figsize=(5, 5))

        plt.imshow(cm)

        plt.title(f"Confusion Matrix - {name}")

        plt.colorbar()

        plt.savefig("confusion_matrix.png")

        # ==========================================
        # CLASSIFICATION REPORT
        # ==========================================

        print("\nClassification Report:")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        # ==========================================
        # LOG PARAMETERS
        # ==========================================

        mlflow.log_param(
            "model_name",
            name
        )

        mlflow.log_param(
            "training_rows",
            len(X_train_smote)
        )

        mlflow.log_param(
            "testing_rows",
            len(X_test)
        )

        mlflow.log_param(
            "feature_count",
            X_train_smote.shape[1]
        )

        mlflow.log_param(
            "smote_applied",
            True
        )

        # ==========================================
        # LOG METRICS
        # ==========================================

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1_score_value
        )

        # ==========================================
        # LOG ARTIFACTS
        # ==========================================

        mlflow.log_artifact(
            "confusion_matrix.png"
        )

        # ==========================================
        # LOG MODEL
        # ==========================================

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("MLflow logging completed")

        # ==========================================
        # BEST MODEL CHECK
        # ==========================================

        if roc_auc > best_roc_auc:

            best_roc_auc = roc_auc

            best_model = model

            best_model_name = name


# ==================================================
# SAVE BEST MODEL
# ==================================================

pickle.dump(
    best_model,
    open("artifacts/best_model.pkl", "wb")
)

print("\nBest model saved successfully")

print(f"Best Model: {best_model_name}")

print(f"Best ROC-AUC Score: {best_roc_auc}")