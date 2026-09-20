import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load saved model
model = pickle.load(open("artifacts/best_model.pkl", "rb"))

print("Model loaded successfully")

# Load dataset
df = pd.read_csv("artifacts/train.csv")

# Separate features and target
X = df.drop(columns=["TARGET"])

# Apply same preprocessing
X = pd.get_dummies(X, drop_first=True)

X.columns = X.columns.str.replace(
    '[^A-Za-z0-9_]+',
    '_',
    regex=True
)

X = X.fillna(X.median())

# Get feature importance
importance = model.feature_importances_

# Create dataframe
feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort descending
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance_df.head(20))

# Plot top 20
plt.figure(figsize=(10,6))

plt.barh(
    feature_importance_df["Feature"][:20],
    feature_importance_df["Importance"][:20]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 20 Important Features")

plt.show()
