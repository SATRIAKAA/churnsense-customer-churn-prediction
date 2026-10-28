# Customer Churn Prediction

Machine Learning project to predict customer churn using the Telco Customer Churn dataset. This project covers data preprocessing, model training, model comparison, threshold tuning, evaluation, and serving predictions through a REST API built with FastAPI.

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying customers who are likely to churn can help businesses prioritize retention efforts and allocate resources more effectively.

This project builds a binary classification model that estimates whether a telecom customer is likely to churn based on their service subscriptions, contract details, billing information, and customer profile.

**Project goals:**

- Explore and clean customer data.
- Build a reproducible machine learning pipeline.
- Compare Logistic Regression and Random Forest.
- Tune the classification threshold using validation data.
- Evaluate the final model on a held-out test set.
- Serve predictions through a FastAPI REST API.
- Validate API behavior with automated tests.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- Joblib
- FastAPI
- Uvicorn
- pytest
- HTTPX

## Dataset

**Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains 7,043 customer records and 21 original columns, including customer demographics, subscribed services, contract types, billing information, and the churn label.

The target variable is `Churn`:

- `Yes`: customer churned.
- `No`: customer did not churn.

The `customerID` column is excluded from model training because it identifies customers rather than providing a useful predictive feature.

### Data preprocessing

- Convert `TotalCharges` to a numeric data type.
- Handle invalid or missing numeric values using median imputation.
- Impute missing categorical values using the most frequent category.
- Standardize numeric features.
- One-hot encode categorical features.
- Split data using stratification to preserve the target class distribution.

All preprocessing steps are included in the scikit-learn pipeline to ensure consistent transformations during training and inference.

## Models and Experiments

Two classification algorithms were compared using the same train-test split and preprocessing strategy:

1. Logistic Regression
2. Random Forest

### Initial model comparison

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |              0.8055 |        0.7963 |
| Precision |              0.6572 |        0.6516 |
| Recall    |              0.5588 |        0.5000 |
| F1-score  |              0.6040 |        0.5658 |
| ROC-AUC   |              0.8419 |        0.8349 |

Logistic Regression performed better on this initial evaluation and was selected for further experimentation.

## Threshold Tuning

The default classification threshold of 0.50 was not assumed to be optimal for the business objective.

Threshold selection was performed using a separate validation set. The threshold was selected by comparing precision, recall, and F1-score, without using the final test set.

| Threshold |  Precision |     Recall |   F1-score |
| --------: | ---------: | ---------: | ---------: |
|      0.30 |     0.5374 |     0.7487 |     0.6257 |
|  **0.35** | **0.5656** | **0.7032** | **0.6269** |
|      0.40 |     0.5941 |     0.6417 |     0.6170 |
|      0.45 |     0.6314 |     0.5909 |     0.6105 |
|      0.50 |     0.6601 |     0.5348 |     0.5908 |
|      0.55 |     0.6882 |     0.4840 |     0.5683 |
|      0.60 |     0.7100 |     0.3797 |     0.4948 |

A threshold of **0.35** was selected because it achieved the highest validation F1-score among the tested thresholds.

A lower threshold increases churn detection recall but can also produce more false positives. The appropriate threshold depends on the relative business costs of missed churners and unnecessary retention interventions.

## Final Test Results

The final model was trained on the development set, representing 80% of the data. The remaining 20% was held out for final evaluation.

| Metric             | Test result |
| ------------------ | ----------: |
| Accuracy           |      0.7651 |
| Precision          |      0.5443 |
| Recall             |      0.7059 |
| F1-score           |      0.6147 |
| ROC-AUC            |      0.8419 |
| Decision threshold |        0.35 |

### Confusion matrix

| Actual class | Predicted No Churn | Predicted Churn |
| ------------ | -----------------: | --------------: |
| No Churn     |                814 |             221 |
| Churn        |                110 |             264 |

### Interpretation

- The model correctly identified 264 customers who churned.
- It missed 110 customers who churned.
- It flagged 221 customers as potential churners who did not churn.
- Recall of approximately 70.6% means the model identified about 71% of actual churn cases in the test set.

These results represent an initial baseline, not a guarantee of business impact. A production deployment would require additional validation, monitoring, and evaluation against actual retention outcomes.

## Project Structure

```text
ML PROJECT/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── churn_pipeline.joblib
│   └── churn_threshold.txt
├── train.py
├── compare_models.py
├── tune_threshold.py
├── final_train.py
├── app.py
├── test_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with your actual repository URL and directory name.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset and model files

Download the dataset from Kaggle and place the CSV at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The prediction API also expects these files:

```text
models/churn_pipeline.joblib
models/churn_threshold.txt
```

Make sure the model artifacts are included in the repository or obtained through your chosen artifact distribution method.

## Run the API

Start the FastAPI development server:

```powershell
uvicorn app:app --reload
```

The API will be available locally at:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

## API Usage

### Endpoint

```http
POST /predict
Content-Type: application/json
```

### Example request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 906.0
}
```

### Example response

```json
{
  "churn_prediction": 1,
  "churn_label": "Churn",
  "churn_probability": 0.6778,
  "threshold": 0.35
}
```

The response above is an example of a successful prediction. Actual results depend on the input and the trained model.

- `churn_prediction`: `1` indicates predicted churn; `0` indicates predicted no churn.
- `churn_label`: human-readable prediction.
- `churn_probability`: estimated churn probability.
- `threshold`: decision threshold applied to the probability.

## Run Automated Tests

Run the test suite with:

```powershell
python -m pytest -v
```

The current test suite checks:

- API root response.
- Health check.
- Successful prediction response.
- Rejection of invalid input.

**Current test status:** 4 tests passed in the local development environment.

## Limitations and Future Improvements

Potential next steps include:

- Add cross-validation and confidence intervals for model evaluation.
- Evaluate threshold selection using explicit business costs.
- Add input validation for categorical values and business rules.
- Expand automated tests, including edge cases and model artifact failures.
- Add data drift and prediction monitoring.
- Package the application with Docker.
- Deploy the API to a cloud platform.
- Build a dashboard for exploring churn risk and model predictions.
- Evaluate retention strategies using real intervention outcomes.

## Disclaimer

This project is for educational and portfolio purposes. Predictions are estimates based on historical data and should not be treated as certain outcomes or used as the sole basis for customer decisions.

## Author

**Satria**

Aspiring Machine Learning Engineer
