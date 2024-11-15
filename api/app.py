from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

import pickle

import pickle



# Initialize the FastAPI app
app = FastAPI()

# Load the saved model
with open("xgboost_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define a request model
class PredictionRequest(BaseModel):
    features: list  # List of feature values for prediction

@app.post("/predict")
async def predict(data: PredictionRequest):
    try:
        # Make prediction
        prediction = model.predict([data.features])[0]
        return {"predicted_price": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
