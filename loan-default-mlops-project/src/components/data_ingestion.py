import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv("notebook/data/application_train.csv")

print("Dataset loaded successfully")

# Check dataset shape
print(df.shape)

# Train-test split
train_set, test_set = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Train-test split completed")

# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)

# Save train dataset
train_set.to_csv("artifacts/train.csv", index=False)

# Save test dataset
test_set.to_csv("artifacts/test.csv", index=False)

print("Data ingestion completed")