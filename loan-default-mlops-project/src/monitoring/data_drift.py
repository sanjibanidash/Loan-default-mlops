from src.pipeline.retraining_pipeline import (
    RetrainingPipeline
)

from datetime import datetime

import pandas as pd


# ==========================================
# LOAD DATASETS
# ==========================================

reference_data = pd.read_csv(
    "artifacts/reference_data.csv"
)

current_data = pd.read_csv(
    "artifacts/current_data.csv"
)


# ==========================================
# SELECT NUMERICAL COLUMNS
# ==========================================

numerical_columns = [

    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED"

]


# ==========================================
# DRIFT THRESHOLD
# ==========================================

DRIFT_THRESHOLD = 30


# ==========================================
# REPORT STORAGE
# ==========================================

report_lines = []


# ==========================================
# DRIFT FLAG
# ==========================================

drift_detected = False


# ==========================================
# CHECK DRIFT
# ==========================================

print("\n" + "=" * 60)

print("DATA DRIFT REPORT")

print("=" * 60)


for column in numerical_columns:

    # Reference mean
    reference_mean = reference_data[
        column
    ].mean()


    # Current mean
    current_mean = current_data[
        column
    ].mean()


    # ==========================================
    # PERCENTAGE DIFFERENCE
    # ==========================================

    difference = abs(
        (current_mean - reference_mean)
        / reference_mean
    ) * 100


    # ==========================================
    # REPORT LINES
    # ==========================================

    line1 = f"\nCOLUMN: {column}"

    line2 = (
        f"Reference Mean: "
        f"{reference_mean:.2f}"
    )

    line3 = (
        f"Current Mean: "
        f"{current_mean:.2f}"
    )

    line4 = (
        f"Difference %: "
        f"{difference:.2f}"
    )


    # ==========================================
    # PRINT REPORT
    # ==========================================

    print(line1)

    print(line2)

    print(line3)

    print(line4)


    # ==========================================
    # SAVE REPORT LINES
    # ==========================================

    report_lines.extend([

        line1,
        line2,
        line3,
        line4

    ])


    # ==========================================
    # DRIFT DETECTION
    # ==========================================

    if difference > DRIFT_THRESHOLD:

        status = "STATUS: DRIFT DETECTED"

        drift_detected = True

    else:

        status = (
            "STATUS: NO SIGNIFICANT DRIFT"
        )


    print(status)

    report_lines.append(status)


# ==========================================
# SAVE REPORT
# ==========================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)

report_path = (
    f"monitoring_reports/"
    f"drift_report_{timestamp}.txt"
)


with open(report_path, "w") as file:

    for line in report_lines:

        file.write(line + "\n")


print("\nDrift report saved successfully")

print(f"Report Path: {report_path}")


# ==========================================
# AUTOMATIC RETRAINING
# ==========================================

if drift_detected:

    print("\n" + "=" * 60)

    print("DRIFT DETECTED")

    print("STARTING AUTOMATIC RETRAINING")

    print("=" * 60)


    pipeline = RetrainingPipeline()

    pipeline.run_pipeline()


else:

    print("\nNo retraining required")