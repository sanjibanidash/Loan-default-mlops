import pandas as pd
import pickle


class PredictionPipeline:

    def __init__(self):

        # ==========================================
        # LOAD MODEL
        # ==========================================

        self.model = pickle.load(
            open(
                "artifacts/best_model.pkl",
                "rb"
            )
        )


        # ==========================================
        # LOAD MODEL COLUMNS
        # ==========================================

        self.model_columns = pickle.load(
            open(
                "artifacts/model_columns.pkl",
                "rb"
            )
        )


        # ==========================================
        # LOAD BEST THRESHOLD
        # ==========================================

        self.best_threshold = pickle.load(
            open(
                "artifacts/best_threshold.pkl",
                "rb"
            )
        )


    # ==========================================
    # PREDICTION METHOD
    # ==========================================

    def predict(self, data: dict):

        # Convert to dataframe
        input_data = pd.DataFrame([data])


        # Add missing columns
        for column in self.model_columns:

            if column not in input_data.columns:

                input_data[column] = 0


        # Keep same column order
        input_data = input_data[
            self.model_columns
        ]


        # Predict probability
        probability = self.model.predict_proba(
            input_data
        )[0][1]


        # Threshold logic
        if probability >= self.best_threshold:

            prediction = "Defaulter"

        else:

            prediction = "Non-Defaulter"


        return prediction, probability