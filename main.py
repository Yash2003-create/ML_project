import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from your_notebook_script import preprocess_data, run_experiment  # Import from the .py file

# File paths for the saved model and label encoder
MODEL_PATH = "best_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

# Initialize FastAPI
app = FastAPI()

# Define the input data schema
class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# Load the model and label encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Income Prediction API. Use /predict for predictions and /experiment for custom experiments."}

@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        # Make predictions
        prediction = model.predict(data)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}

@app.post("/experiment")
def experiment():
    try:
        # Run custom experiment from your notebook logic
        result = run_experiment()  # Replace with the actual function from your notebook .py file
        return {"experiment_result": result}
    except Exception as e:
        return {"error": str(e)}
