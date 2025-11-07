# End-to-End Customer Churn Prediction Pipeline
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.5-blue)

This repository contains a complete, end-to-end MLOps pipeline for a customer churn prediction model. The goal is not just to build a model, but to create a reproducible, production-ready system that trains, tracks, and deploys a model to generate actionable business insights (i.e., a risk-ranked list of customers).

---

## 🚀 Key Features

* **End-to-End Pipeline:** The entire process, from data cleaning to model training and batch prediction, is run via Python scripts.
* **Experiment Tracking:** Uses **MLflow** to log, track, and compare all model experiments (parameters, metrics, and model artifacts).
* **Modular & Reproducible Code:** A clean separation of concerns:
    * `notebooks/`: For initial, messy EDA.
    * `src/`: For clean, modular, and production-ready code.
* **Batch Prediction:** Includes a `predict.py` script to load the "champion" model and score a new batch of customers, saving the output to a CSV.
* **Data Integrity:** Uses `scikit-learn`'s `Pipeline` and `ColumnTransformer` to prevent data leakage and ensure preprocessing is applied consistently during training and inference.

---

## ⚙️ MLOps Architecture & Workflow

This project is built around a modern MLOps workflow, separating discovery from production.

1.  **Exploration (Lab):** Initial data analysis and prototyping is done in `notebooks/01_eda.ipynb`.
2.  **Preprocessing (Factory):** All successful cleaning and feature engineering logic is "graduated" into a robust `scikit-learn` pipeline in `src/preprocessing.py`.
3.  **Training & Tracking (Factory):** `src/train.py` runs the full, automated training process. It loads data, builds the pipeline, trains the model, and logs *everything* (parameters, metrics, and the model itself) to **MLflow**.
4.  **Comparison & Registration (QC):** We use the `mlflow ui` to compare the performance of different runs (e.g., `LogisticRegression` vs. `XGBoost`) and select our "champion" model.
5.  **Prediction (Utility):** `src/predict.py` loads the champion model from its saved file and runs it on new, unseen data to generate a `churn_predictions.csv` report.

---

## 🛠️ Tech Stack

* **Python 3.10**
* **Data:** `pandas`
* **ML Pipeline:** `scikit-learn`
* **Modeling:** `LogisticRegression`, `XGBoost`
* **Experiment Tracking:** `mlflow`
* **Prototyping:** `Jupyter`

---

## 🏁 How to Run This Project

### 1. Prerequisites

* Git
* Python 3.9+
* Access to a terminal (command line)

### 2. Setup & Installation

Clone the repository and set up the virtual environment.

```sh
# 1. Clone this repository
# Make sure to use your own username and repo name!
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# 2. Create and activate a virtual environment
python -m venv .venv

# On Mac/Linux:
source .venv/bin/activate
# On Windows (cmd):
.\.venv\Scripts\activate

# 3. Install the required libraries
pip install -r requirements.txt

### 3. Run the Training Pipeline
This will run the src/train.py script. It cleans data, builds the pipeline, trains all experiment models, and logs results to MLflow.

Bash

python src/train.py
### 4. View Experiments with MLflow
To launch the MLflow dashboard and compare your runs:

Bash

# This command points MLflow to the database file created by train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
Now open your browser and go to http://127.0.0.1:5000 to see your results.

### 5. Run Batch Predictions
This simulates a daily job, loading your saved "champion" model to generate a risk report.

Bash

python src/predict.py
This will create a new file: data/processed/churn_predictions.csv.

# 🗂️ Project Structure
This is the structure of our project.
churn_project/
│
├── data/
│   ├── raw/         # The original, immutable data
│   └── processed/   # The final prediction reports
│
├── models/          # The final, trained .pkl model files
│
├── notebooks/       # "The Lab": Messy EDA and prototyping
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py # The robust scikit-learn pipeline
│   ├── train.py         # "The Factory": Main training script
│   └── predict.py       # "The Utility": Batch prediction script
│
├── .gitignore       # Critical file: Ignores data, models, venvs
├── mlflow.db        # The experiment tracking database
├── requirements.txt # The project's shopping list
└── README.md        # You are here

#Future Work (Next Steps)
Hyperparameter Tuning: Implement GridSearchCV or Optuna in the training script to find the optimal parameters for XGBoost and beat the baseline.
Model Registry: Formally "register" the champion model in the MLflow Model Registry.

📜 License
This project is licensed under the MIT License.
