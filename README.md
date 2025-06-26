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

