from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Set base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and preprocessing tools
model = joblib.load(os.path.join(BASE_DIR, "linear_regression_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (important for frontend interaction)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route to confirm API is running
@app.get("/")
def read_root():
    return {"message": "Student Performance Prediction API is live!"}

# Input schema for prediction
class StudentData(BaseModel):
    Hours_Studied: float
    Attendance: float
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: float
    Previous_Scores: float
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: float
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Physical_Activity: float
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str

# Prediction route
@app.post("/predict")
def predict(data: StudentData):
    input_dict = data.dict()

    # Encode categorical variables
    for key in input_dict:
        if key in label_encoders:
            encoder = label_encoders[key]
            input_dict[key] = encoder.transform([input_dict[key]])[0]

    # Prepare data for model
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    return {"predicted_score": float(prediction)}
