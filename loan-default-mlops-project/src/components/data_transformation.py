from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import pandas as pd
import numpy as np

#Numerical pipeline
numerical_pipeline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ]
)

#Categorical pipeline
categorical_pipeline = Pipeline(
    steps=[
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore"))
    ]
)

print("Numerical and categorical pipelines created successfully")

#Example column lists
numerical_columns = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]

categorical_columns = ["CODE_GENDER", "FLAG_OWN_CAR"]

#Combine both pipelines

preprocessor = ColumnTransformer(
    [
        ("numerical_pipeline",numerical_pipeline,numerical_columns),
        ("categorical_pipeline",categorical_pipeline,categorical_columns)
    ]
)

print("Preprocessor created successfully")



# Load dataset

df = pd.read_csv("notebook/data/application_train.csv")


# Take sample columns

sample_df = df[
    [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "CODE_GENDER",
        "FLAG_OWN_CAR"
    ]
]


# Apply preprocessing

transformed_data = preprocessor.fit_transform(sample_df)


# Show first 5 rows

print(transformed_data[:5])