import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import joblib # Used for saving the pipeline
import os
from xgboost import XGBClassifier

# Import our custom modules 
from preprocessing import clean_data, build_preprocessing_pipeline

# 1. Load Data 
print("Loading data...")
file_path = 'Data/raw_data.csv'

df = pd.read_csv(file_path)

# 2. Clean Data 
# We call our custom function from preprocessing.py
print("Cleaning data...")
df_clean = clean_data(df)

# 3. Define Features & Target 
TARGET = 'Churn'

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
]

# Note: We'll treat SeniorCitizen as categorical
# (it's 0/1, so one-hot encoding works perfectly)
# We define our inputs (X) and our output (y)
X = df_clean[numeric_features + categorical_features]
y = df_clean[TARGET]

#  4. Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,        # Hold back 20% for testing
    random_state=42,      # For reproducible results
    stratify=y            # Ensures train/test have the same % of churn
)

# 5. Build The *Full* Pipeline 
print("Building full model pipeline...")

# We call our pipeline-builder from preprocessing.py
preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

# Now we chain our preprocessor and our model together
# This is our V1 model: a simple Logistic Regression
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, 
                                 scale_pos_weight=scale_pos_weight, 
                                 use_label_encoder=False, 
                                 eval_metric='logloss'))
])
# We'll store our experiment runs in a local file
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Churn_Prediction_V1")

# --- 7. Train & Log (The "Performance Review") ---
print("Training XGBoost model...")

with mlflow.start_run(run_name="XGBoost_Weighted"):
    
    # Train the entire pipeline
    model_pipeline.fit(X_train, y_train)

    # Evaluate on the Test Set
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for AUC

    # We use F1 and AUC, NOT accuracy (due to imbalance)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC-ROC: {auc:.4f}")

    # Log parameters to MLflow
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("class_weight", scale_pos_weight) # Log the value we calculated
    # Log metrics to MLflow
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # Log the trained model *pipeline* to MLflow
    mlflow.sklearn.log_model(model_pipeline, "model")

    # --- 8. Save Model (The "Shipping") ---
    # We save the *entire* pipeline to a file for later use
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "churn_model_v2.pkl")
    
    joblib.dump(model_pipeline, model_path)
    print(f"Model pipeline saved to {model_path}")

print("\nTraining script finished successfully.")