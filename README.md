# Customer Behavioral Analysis in E-Commerce Retail

## 1. Project Summary
This repository contains a complete end-to-end machine learning system for customer behavioral analysis and churn prediction in an e-commerce retail context.

The project is designed to be:
- Academic-submission friendly (clear structure and documented workflow)
- Production-oriented (modular code, reusable preprocessing, API deployment)
- Robust against real-world data issues (missing values, inconsistent formats, outliers, imbalance risk, irrelevant and leakage-prone features)

Primary business objective:
- Predict customer churn probability and churn class early enough to support retention actions.

## 2. Key Features
- Data quality auditing utilities (missing values, outliers, correlations)
- Structured preprocessing pipeline
- Feature engineering from dates and IP fields
- Train/test-safe transformations to prevent data leakage
- Dimensionality reduction using PCA
- Multiple model training with hyperparameter tuning
- Standard classification evaluation metrics
- Reusable inference module for batch and single prediction
- Flask API with health check and prediction endpoint

## 3. Repository Structure
```text
projet_ml_retail/
|-- app/
|   |-- app.py
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- train_test/
|-- models/
|-- notebooks/
|   |-- retail_analysis_pipeline.ipynb
|-- reports/
|-- src/
|   |-- preprocessing.py
|   |-- train_model.py
|   |-- predict.py
|   |-- utils.py
|-- .gitignore
|-- README.md
|-- requirements.txt
```

### Folder Purpose
- `data/raw`: Original input dataset
- `data/processed`: Combined processed dataset export
- `data/train_test`: Train/test-ready processed matrices and targets
- `src`: Core ML pipeline code
- `models`: Serialized preprocessing and trained model artifacts
- `reports`: Metrics, model comparison, and preprocessing summaries
- `app`: Flask API layer for deployment
- `notebooks`: Analysis and visualization notebook(s)

## 4. Dataset
Expected raw file:
- `data/raw/retail_customers_COMPLETE_CATEGORICAL.csv`

The pipeline auto-detects common churn target names and currently uses `Churn`.

## 5. Technical Stack
- Python 3.13+
- pandas, numpy, scipy
- scikit-learn
- Flask
- joblib
- matplotlib, seaborn (used in notebook)

## 6. Setup and Installation
From project root (`projet_ml_retail`):

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 7. End-to-End Pipeline Execution

### Step 1: Preprocess Data
Command:
```bash
python src/preprocessing.py --input data/raw/retail_customers_COMPLETE_CATEGORICAL.csv
```

What this step does:
- Cleans inconsistent string formats
- Parses `RegistrationDate` robustly (mixed formats)
- Drops irrelevant/leakage-prone features when present
- Extracts basic numeric features from IP-like columns
- Handles missing values using mean, median, and KNN strategies
- Encodes categorical features (OneHot/Ordinal)
- Fits `StandardScaler` on train split only
- Saves train/test processed outputs and preprocessing artifact

Optional flags:
- `--target-col Churn`
- `--test-size 0.2`
- `--random-state 42`
- `--ordinal-cols LoyaltyTier,CustomerSegment`

### Step 2: Train and Tune Models
Command:
```bash
python src/train_model.py
```

Models trained:
- Logistic Regression
- Random Forest
- KNN
- Gradient Boosting

Training strategy:
- PCA inside model pipeline
- `GridSearchCV` for hyperparameter search
- Stratified CV for robust model selection
- Best model selected by validation ROC-AUC and saved

Metrics produced:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Step 3: Local Prediction (Script)
Recommended (PowerShell-safe) example:
```powershell
python -c "from src.predict import RetailChurnPredictor; p=RetailChurnPredictor(); print(p.predict({'Age':35,'Gender':'Female','Country':'United Kingdom','Recency':30,'Frequency':5,'MonetaryTotal':500}))"
```

Expected output shape:
```json
{"churn_probability": 0.12345, "predicted_class": 0}
```

### Step 4: Run Flask API
Start server:
```bash
python app/app.py
```

Endpoints:
- `GET /health`
- `POST /predict`

PowerShell POST example:
```powershell
$body = @{Age=35;Gender='Female';Country='United Kingdom';Recency=30;Frequency=5;MonetaryTotal=500} | ConvertTo-Json -Compress
Invoke-WebRequest -Uri "http://127.0.0.1:5000/predict" -Method Post -UseBasicParsing -ContentType "application/json" -Body $body
```

Sample response captured during validation:
```json
{"prediction":{"churn_probability":1.8e-05,"predicted_class":0}}
```

## 8. Artifacts Generated

### Processed Data
- `data/train_test/X_train_processed.csv`
- `data/train_test/X_test_processed.csv`
- `data/train_test/y_train.csv`
- `data/train_test/y_test.csv`
- `data/processed/retail_processed.csv`

### Model Files
- `models/preprocessing_pipeline.joblib`
- `models/best_model.joblib`

### Reports
- `reports/missing_values_report.csv`
- `reports/outlier_report.csv`
- `reports/model_comparison.csv`
- `reports/best_model_classification_report.txt`
- `reports/preprocessing_summary.json`
- `reports/training_summary.json`

## 9. Latest Training Snapshot
From the most recent run:
- Best model: Logistic Regression
- Train shape: 3497 x 137
- Test shape: 875 x 137
- Test Accuracy: 0.9977
- Test Precision: 1.0000
- Test Recall: 0.9931
- Test F1: 0.9966
- Test ROC-AUC: 0.9999

Note: Metrics this high may indicate the dataset is strongly separable. Continue leakage checks when adding new features.

## 10. Design Decisions and Best Practices
- No leakage by fitting transformers on training data only
- Same preprocessing object reused for training and inference
- Modular source files for maintainability and testing
- Explicit artifacts for reproducibility
- API model loading performed once at startup for performance

## 11. Notebook
Notebook available at:
- `notebooks/retail_analysis_pipeline.ipynb`

Notebook includes:
- Data overview and missingness checks
- Churn class distribution
- PCA explained variance chart
- Model comparison visualization
- Training summary inspection

## 12. Troubleshooting

### `ModuleNotFoundError: No module named 'flask'`
Install dependencies in the same environment used to run the app:
```bash
pip install -r requirements.txt
```

### PowerShell security warning when using `Invoke-WebRequest`
Use `-UseBasicParsing`.

### JSON quoting issues in PowerShell CLI prediction
Prefer dictionary-based `python -c` example shown above.

## 13. Reproducibility Notes
- Random seeds are set in train/test split and model components where applicable.
- Re-running pipeline regenerates all artifacts in `data/train_test`, `data/processed`, `models`, and `reports`.

## 14. Academic Submission Checklist
- [ ] Raw dataset present in `data/raw`
- [ ] Preprocessing executed successfully
- [ ] Training completed and reports generated
- [ ] API tested with at least one POST request
- [ ] Notebook opens and runs key cells
- [ ] README updated with final run results

## 15. License
For academic use. Add your institution-required license text if needed.
