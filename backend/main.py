from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load model and preprocessing tools
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = FastAPI()

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define input schema using Pydantic
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


@app.post("/predict")
def predict(data: StudentData):
    input_dict = data.dict()

    # Apply label encoding for categorical values
    for key in input_dict:
        if key in label_encoders:
            encoder = label_encoders[key]
            input_dict[key] = encoder.transform([input_dict[key]])[0]

    # Convert to array and reshape
    input_array = np.array(list(input_dict.values())).reshape(1, -1)

    # Scale
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]

    return {"predicted_score": float(prediction)}


# Uncomment to run directly
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
