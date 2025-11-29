import pandas as pd
import joblib
import os

print("Prediction script started...")

# --- 1. Load the "Product" ---
print("Loading champion model pipeline...")
# We're running from the root 'churn_project' folder, so the path is simple
model_path = 'models/churn_model_v2.pkl'
model_pipeline = joblib.load(model_path)
print("Model loaded successfully.")

# --- 2. Load "New" Data ---
print("Loading new Data to predict on...")
# Path relative to the root 'churn_project' folder
file_path = 'Data/raw_data.csv' 
df_raw = pd.read_csv(file_path)

# We'll save the customerIDs for our final report
customer_ids = df_raw['customerID']

# We just need the raw data, *without* the Churn column
# The pipeline will do ALL the other work
df_predict = df_raw.drop(['customerID', 'Churn'], axis=1, errors='ignore')

# --- 3. Apply Manual Pre-Pipeline Cleaning ---
# CRITICAL FIX: We must apply the same manual conversion we did in training
# BEFORE we give the data to the pipeline.
# This turns the empty strings ' ' into NaN.
print("Applying manual pre-pipeline cleaning...")
df_predict['TotalCharges'] = pd.to_numeric(df_predict['TotalCharges'], errors='coerce')
    
# Now, our pipeline's SimpleImputer (set to 'median') will correctly
# find these NaN values and fill them.
# --- END FIX ---

# --- 4. Make Predictions ---
print("Making predictions...")
# The model_pipeline now does ALL the *other* preprocessing
# (Imputing the NaNs we just made, Scaling, One-Hot Encoding)
#
# We use .predict_proba() to get the *probability* (the "churn score")
# [:, 1] selects the probability for the "Yes" (1) class
churn_probabilities = model_pipeline.predict_proba(df_predict)[:, 1]

# --- 5. Create Final Report ---
print("Creating final report...")
# Create a new DataFrame with the customerID and the new score
df_report = pd.DataFrame({
    'customerID': customer_ids, 
    'ChurnProbability': churn_probabilities
})

# Sort the list from highest risk to lowest
df_report = df_report.sort_values(by='ChurnProbability', ascending=False)

# --- 6. Save Report ---
output_dir = 'data/processed'
os.makedirs(output_dir, exist_ok=True)
report_path = os.path.join(output_dir, 'churn_predictions.csv')

df_report.to_csv(report_path, index=False)

print(f"\nPrediction report saved to {report_path}")
print("\nTop 5 highest-risk customers:")
print(df_report.head())
print("\nPrediction script finished.")