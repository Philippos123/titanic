import os 
import pandas as pd
import streamlit as st
import joblib

PORT = os.environ.get('PORT', 8501)

# Ladda modellen
model = joblib.load('titanic_model.pkl')

# Förutsäga överlevnad
def predict_survival(age, gender, fare, parch, pclass, sibsp):
    input_data = pd.DataFrame({
        'Pclass': [pclass],    
        'Sex': [1 if gender == 'female' else 0],  
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })
    survival_probability = model.predict_proba(input_data)[:, 1]  
    return survival_probability[0]

# Streamlit app
st.title("Titanic Survival Predictor")

# Användarinmatning
age = st.number_input("Ange din ålder:", min_value=0, max_value=120)
gender = st.selectbox("Välj kön:", ("male", "female"))
fare = st.number_input("Ange biljettpris du är villig att betala:", min_value=0.0)
parch = st.number_input("Ange antal familjemedlemmar som reser med (Parch):", min_value=0)
pclass = st.selectbox("Välj klass (Pclass):", (1, 2, 3))
sibsp = st.number_input("Ange antal syskon/spouse (SibSp):", min_value=0)

# Förutsäg när knappen trycks
if st.button("Förutsäg överlevnad"):
    chance_of_survival = predict_survival(age, gender, fare, parch, pclass, sibsp)
    st.write(f"Din chans att överleva är {chance_of_survival * 100:.2f}%")