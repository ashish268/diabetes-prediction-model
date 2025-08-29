import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Diabetes Prediction Using Machine Learning ")

st.write("""
This app predicts **whether a patient is diabetic** based on health parameters.  
The model is trained on the **Pima Indians Diabetes Dataset**.

 Diabetes Prediction using Machine Learning "Diabetes" prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment.
""")
st.image('https://www.clinicaladvisor.com/wp-content/uploads/sites/11/2020/06/diabetes-care_G_1213259073.jpg')

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")  # Upload this dataset to your repo
    return df

df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.write(f"Model Accuracy: **{accuracy:.2f}**")

# -----------------------------
# Sidebar for User Input
# -----------------------------
st.sidebar.header("Enter Patient Data")
st.sidebar.image('https://sdmntprnorthcentralus.oaiusercontent.com/files/00000000-1820-622f-8c28-8cee707c59dc/raw?se=2025-08-29T07%3A54%3A11Z&sp=r&sv=2024-08-04&sr=b&scid=f4706fd9-4f89-5da1-8d60-c6d020cf132d&skoid=bbd22fc4-f881-4ea4-b2f3-c12033cf6a8b&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-08-28T20%3A51%3A38Z&ske=2025-08-29T20%3A51%3A38Z&sks=b&skv=2024-08-04&sig=oKlGNqBSuhWF6e7kFptnUUfgGnXkmYpk0odFl40zfAM%3D')

def user_input():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 10, 100, 33)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Patient Data")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction Result")
st.write("🟥 Diabetic" if prediction[0] == 1 else "🟩 Not Diabetic")

st.markdown('Designed by:**Ashish Luthra**  and  **Mishti Sehgal**')









