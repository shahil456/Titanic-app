import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Titanic Survival Predictor")
st.title("🚢 Titanic Survival Predictor")

model = joblib.load("model.joblib")
features = json.load(open("features.json"))

pclass = st.selectbox("Passenger class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare", 0.0, 500.0, 30.0)
sibsp = st.slider("Siblings/spouses aboard", 0, 8, 0)
parch = st.slider("Parents/children aboard", 0, 6, 0)

if st.button("Predict"):
    row = pd.DataFrame([{
        "pclass": pclass, "sex_male": int(sex == "male"), "age": age,
        "sibsp": sibsp, "parch": parch, "fare": fare,
        "embarked_Q": 0, "embarked_S": 1, "alone": int(sibsp + parch == 0),
    }])
    row = row.reindex(columns=features, fill_value=0)   # ← the fix
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    st.success(f"{'Survived' if pred else 'Did not survive'} (probability: {prob:.1%})")