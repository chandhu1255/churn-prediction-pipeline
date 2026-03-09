import pandas as pd
import os

# Find the path to your raw data
# Looking at your file structure, it's in Data/raw_data.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'raw_data.csv')

def get_customer_from_db(customer_id: str):
    #Simulates: SELECT * FROM customers WHERE customerID = 'customer_id'
    df = pd.read_csv(DATA_PATH)
    
    # Look for the specific customer
    customer_row = df[df['customerID'] == customer_id]
    
    if customer_row.empty:
        return None
    
    # Return as a dictionary
    return customer_row.to_dict(orient='records')[0]