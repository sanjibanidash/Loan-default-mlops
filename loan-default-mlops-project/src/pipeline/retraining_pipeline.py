import os
import sys


# ==========================================
# RETRAINING PIPELINE
# ==========================================

class RetrainingPipeline:


    def run_pipeline(self):

        print("\n" + "=" * 60)

        print("STARTING RETRAINING PIPELINE")

        print("=" * 60)


        # ==========================================
        # CURRENT PYTHON EXECUTABLE
        # ==========================================

        python_executable = sys.executable


        # ==========================================
        # STEP 1 → TRAIN MODEL
        # ==========================================

        print("\nTraining new model...")


        train_status = os.system(

            f'"{python_executable}" '
            f'-m src.components.model_trainer'

        )


        if train_status != 0:

            print("\nMODEL TRAINING FAILED")

            return


        # ==========================================
        # STEP 2 → THRESHOLD TUNING
        # ==========================================

        print("\nRunning threshold tuning...")


        threshold_status = os.system(

            f'"{python_executable}" '
            f'-m src.components.threshold_tuning'

        )


        if threshold_status != 0:

            print("\nTHRESHOLD TUNING FAILED")

            return


        print("\n" + "=" * 60)

        print("RETRAINING COMPLETED SUCCESSFULLY")

        print("=" * 60)