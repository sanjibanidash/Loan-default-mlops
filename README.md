# Loan Default Prediction & MLOps

An end-to-end machine learning and MLOps project for predicting loan default risk and supporting data-driven credit-risk decisions.

The project covers the complete ML lifecycle — from data ingestion and exploratory analysis to model training, threshold optimization, prediction, monitoring, and retraining.

---

## 📌 Business Problem

Loan defaults can result in significant financial losses for lending institutions.

The objective of this project is to build a machine learning system that predicts whether a borrower is likely to default on a loan.

Instead of relying only on model accuracy, the project also considers the business impact of classification errors and uses threshold analysis to support risk-based decision making.

### Key Objectives

- Predict loan default probability.
- Identify high-risk borrowers.
- Compare multiple machine learning models.
- Optimize the prediction threshold based on business considerations.
- Provide model predictions through an application interface.
- Monitor incoming data for drift.
- Support model retraining when required.
- Provide model explainability and feature importance.

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │     Loan Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Ingestion    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Transformation │
                    │ & Feature Engineering│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Training    │
                    │ Logistic Regression │
                    │ Random Forest       │
                    │ XGBoost             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │ & Threshold Tuning  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Model Prediction  │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │   Streamlit     │       │ Batch Prediction│
        │   Application   │       │      CSV        │
        └─────────────────┘       └─────────────────┘

                  Monitoring & MLOps
                         │
                         ▼
              ┌──────────────────────┐
              │   Data Drift         │
              │   Monitoring         │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Model Retraining     │
              └──────────────────────┘
