
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (assuming the model is saved as 'model.pkl')
model = joblib.load("model.pkl")

# Define the input data structure
class PropertyData(BaseModel):
    type: str
    sector: str
    net_usable_area: float
    net_area: float
    n_rooms: int
    n_bathroom: int
    latitude: float
    longitude: float

# Endpoint to get predictions
@app.post("/predict/")
async def predict(data: PropertyData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction
    return {"predicted_price": prediction[0]}
