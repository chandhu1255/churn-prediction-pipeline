import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib 
import os
from xgboost import XGBClassifier

# --- IMPORT CUSTOM MODULES ---
# Ensure preprocessing.py is in the same folder
from preprocessing import clean_data, build_preprocessing_pipeline

# --- 1. CONFIGURATION ---
EXPERIMENT_NAME = "Churn_Prediction_Tuning"
DB_PATH = "sqlite:///mlflow.db"
RAW_DATA_PATH = 'Data/raw_data.csv'
MODEL_OUTPUT_DIR = "models"
MODEL_FILENAME = "churn_model_v2_tuned.pkl"

# --- 2. LOAD DATA ---
print("Loading data...")
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"The file {RAW_DATA_PATH} was not found.")

df = pd.read_csv(RAW_DATA_PATH)

# --- 3. CLEAN DATA ---
print("Cleaning data...")
# Uses function from preprocessing.py
df_clean = clean_data(df)

# --- 4. DEFINE FEATURES & TARGET ---
TARGET = 'Churn'

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
]

X = df_clean[numeric_features + categorical_features]
y = df_clean[TARGET]

# --- 5. SPLIT DATA ---
print("Splitting data...")
# Stratify is crucial here to maintain the Churn ratio in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42, 
    stratify=y 
)

# --- 6. BUILD PIPELINE ---
print("Building base pipeline...")

# A. Calculate Class Weight (Negative / Positive) to handle imbalance
# This tells the model: "Pay X times more attention to the Churn class"
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")

# B. Build Preprocessor
preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

# C. Initialize Pipeline with XGBoost
# We set the 'static' parameters here (random_state, objective, class weight)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42, 
        scale_pos_weight=scale_pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss',
        objective='binary:logistic'
    ))
])

# --- 7. HYPERPARAMETER TUNING SETUP ---
print("Configuring Hyperparameter Search Space...")

# The 'classifier__' prefix is REQUIRED to target the XGBoost step inside the Pipeline
param_grid = {
    # Tree Structure (Complexity)
    'classifier__max_depth': [3, 4, 5, 6, 8],            # Shallow trees genearlize better
    'classifier__min_child_weight': [1, 3, 5, 7],        # Higher values prevent overfitting
    
    # Gradient Descent (Speed vs Accuracy)
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2], # 'eta'
    'classifier__n_estimators': [100, 200, 300, 500],    # Number of trees
    
    # Stochastic Sampling (Randomness to prevent overfitting)
    'classifier__subsample': [0.6, 0.8, 1.0],            # % of rows used per tree
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],     # % of columns used per tree
    'classifier__gamma': [0, 0.1, 0.5]                   # Minimum loss reduction to split
}

# Initialize RandomizedSearchCV
search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_grid,
    n_iter=30,             # Number of random combinations to try (Higher = longer but better)
    scoring='f1',          # We optimize for F1 because accuracy is misleading for Churn
    cv=3,                  # 3-Fold Cross Validation
    verbose=1,             # Print progress
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

# --- 8. TRAIN & LOG (MLflow) ---
print("Starting Hyperparameter Tuning with MLflow tracking...")

mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="XGBoost_Tuned_V2"):
    
    # EXECUTE SEARCH
    # This trains (n_iter * cv) models. 30 * 3 = 90 training runs.
    search.fit(X_train, y_train)
    
    # GET BEST MODEL
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    print(f"\nBest Parameters Found: {best_params}")
    
    # EVALUATE
    print("Evaluating best model on Test Set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Optimized Test F1 Score: {f1:.4f}")
    print(f"Optimized Test AUC-ROC: {auc:.4f}")

    # LOGGING
    # 1. Log the parameters that won
    mlflow.log_params(best_params)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    
    # 2. Log the metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # 3. Log the model artifact
    print("Logging model artifact to MLflow...")
    mlflow.sklearn.log_model(best_model, "model")

    # --- 9. SAVE MODEL TO DISK ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_FILENAME)
    
    joblib.dump(best_model, save_path)
    print(f"\nSUCCESS: Tuned model saved to {save_path}")

print("Training script finished.")