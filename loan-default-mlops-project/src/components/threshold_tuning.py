import pandas as pd
import pickle

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("artifacts/train.csv")


# ==========================================
# INPUT / TARGET
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
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(
    open("artifacts/best_model.pkl", "rb")
)


# ==========================================
# PROBABILITIES
# ==========================================

probabilities = model.predict_proba(X_test)[:, 1]


# ==========================================
# BEST THRESHOLD TRACKING
# ==========================================

best_threshold = 0

best_f1 = 0


# ==========================================
# TEST THRESHOLDS
# ==========================================

for threshold in [x / 10 for x in range(1, 10)]:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_test,
        predictions
    )


    recall = recall_score(
        y_test,
        predictions
    )


    f1 = f1_score(
        y_test,
        predictions
    )


    print("\n" + "="*50)

    print(f"Threshold: {threshold}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall: {recall:.4f}")

    print(f"F1 Score: {f1:.4f}")


    # Track best threshold
    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold


# ==========================================
# FINAL BEST THRESHOLD
# ==========================================

print("\n" + "="*50)

print("BEST THRESHOLD FOUND")

print("="*50)

print(f"Best Threshold: {best_threshold}")

print(f"Best F1 Score: {best_f1:.4f}")


# ==========================================
# SAVE BEST THRESHOLD
# ==========================================

pickle.dump(
    best_threshold,
    open("artifacts/best_threshold.pkl", "wb")
)

print("\nBest threshold saved successfully")