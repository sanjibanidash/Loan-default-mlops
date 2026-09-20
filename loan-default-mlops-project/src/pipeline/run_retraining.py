from src.pipeline.retraining_pipeline import (
    RetrainingPipeline
)


# ==========================================
# RUN RETRAINING
# ==========================================

if __name__ == "__main__":

    pipeline = RetrainingPipeline()

    pipeline.run_pipeline()