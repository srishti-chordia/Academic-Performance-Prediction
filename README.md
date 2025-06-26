# Academic-Performance-Prediction

This project is a web application that predicts a student's exam score based on multiple academic, behavioral, and socio-economic factors. It uses a trained Machine Learning model deployed via a FastAPI backend and a simple HTML/CSS/JS frontend interface.


## 📌 Features

- Predicts student exam score based on 18 different input factors
- Interactive web UI with dropdowns and numeric fields
- FastAPI backend with deployed ML model
- Model trained using Linear Regression, scaled inputs, and encoded categorical variables
- Deployed on Render for seamless full-stack integration

---

## 🧠 ML Model Details

- **Model**: Linear Regression
- **Features Used**:
  - Hours Studied, Attendance, Sleep Hours, Tutoring Sessions, etc.
  - Categorical features: Parental Involvement, Internet Access, Gender, etc.
- **Preprocessing**:
  - `StandardScaler` used for numerical features
  - Label encoding for categorical variables
- **Artifacts**:
  - `linear_regression_model.pkl`
  - `scaler.pkl`
  - `label_encoders.pkl`

---

## 📁 Project Structure

student-performance/
├── backend/
│ ├── main.py # FastAPI app
│ ├── linear_regression_model.pkl
│ ├── scaler.pkl
│ └── label_encoders.pkl
├── frontend/
│ ├── index.html # Frontend UI
│ └── style.css # (Optional) Styling
├── dataset.csv # Original dataset (optional)
├── requirements.txt # Python dependencies
└── README.md

yaml
Always show details



## 🔧 Setup Instructions

### Backend (FastAPI)

```bash
# Clone repo
git clone https://github.com/your-username/student-performance.git
cd student-performance/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # or venv\\Scripts\\activate on Windows

# Install dependencies
pip install -r ../requirements.txt

# Run the app
uvicorn main:app --reload
App will run on http://localhost:8000

Frontend
Just open frontend/index.html in a browser.
Or use a static hosting platform to deploy it (like Render Static, Netlify, GitHub Pages, etc.).

📦 Deployment
Backend: Hosted using Render web service

Frontend: Can be opened locally or hosted as a static site

CORS is enabled to allow communication between frontend and backend

📊 Example Prediction Factors
Feature	Example Value
Hours_Studied	5.0
Attendance (%)	92.5
Parental_Involvement	High
Sleep_Hours	7.0
Gender	Female
Previous_Scores	75.0
Tutoring_Sessions	2.0

🛠️ Technologies Used
Python

FastAPI

scikit-learn

Joblib

HTML/CSS/JS (Vanilla)

Render (Hosting)

✨ Future Improvements
Add model selection (Linear Regression, Random Forest, etc.)

Use dropdown suggestions powered by ML for inputs

Improve UI/UX with full styling and error handling

Collect user inputs to continuously improve the model
"""


