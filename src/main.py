from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from src.database_mock import get_customer_from_db

app = FastAPI(title="Pro Churn Prediction Service")

# Load the champion model
model = joblib.load("models/churn_model_v2.pkl")

# The only thing the user sends now!
class CustomerLookup(BaseModel):
    customer_id: str

@app.post("/predict")
def predict_by_id(request: CustomerLookup):
    # 1. Enrichment: Get the full data from the "Database"
    customer_data = get_customer_from_db(request.customer_id)
    
    if not customer_data:
        raise HTTPException(status_code=404, detail="Customer ID not found")

    # 2. Preparation: Convert to DataFrame for the model
    # The model expects a list of dictionaries (rows)
    input_df = pd.DataFrame([customer_data])
    
    # 3. Inference: Run the prediction
    # We use our pipeline which handles all the categorical/numeric prep automatically
    probability = model.predict_proba(input_df)[0][1]
    prediction = "Churn" if probability > 0.5 else "Loyal"

    # 4. Response: Return the simplified result
    return {
        "customer_id": request.customer_id,
        "prediction": prediction,
        "churn_probability": f"{probability:.2%}",
        "status": "Success"
    }