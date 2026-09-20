import pandas as pd
import pickle

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("artifacts/train.csv")


# ==========================================
# INPUT AND TARGET
# ==========================================

X = df.drop(columns=["TARGET", "SK_ID_CURR"])

y = df["TARGET"]


# ==========================================
# ENCODING
# ==========================================

X = pd.get_dummies(X, drop_first=True)

X.columns = X.columns.str.replace(
    '[^A-Za-z0-9_]+',
    '_',
    regex=True
)


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

X = X.fillna(X.median())


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = pickle.load(
    open("artifacts/best_model.pkl", "rb")
)


# ==========================================
# GET PROBABILITIES
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# THRESHOLD LIST
# ==========================================

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]


# ==========================================
# TEST DIFFERENT THRESHOLDS
# ==========================================

for threshold in thresholds:

    print("\n" + "="*60)

    print(f"THRESHOLD: {threshold}")

    print("="*60)


    predictions = (
        probabilities >= threshold
    ).astype(int)


    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(f"\nROC-AUC Score: {roc_auc}")