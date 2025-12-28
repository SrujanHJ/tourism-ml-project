import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Purchase Prediction", layout="centered")

@st.cache_resource
def load_model():
    repo_id = "srujanhj/tourism_wellness_best_model"  
    filename = "best_model.pkl"        
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = joblib.load(model_path)
    return model

model = load_model()

st.title("Tourism Wellness Package Purchase Prediction")
st.write("This app predicts whether the customer will purchase the Wellness Tourism Package.")

Age = st.number_input("Age", min_value=1, max_value=90, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
NumberOfTrips = st.number_input("Number of Trips Per Year", min_value=0, value=2)
Passport = st.selectbox("Passport Available", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=2)
DurationOfPitch = st.number_input("Duration Of Pitch (Minutes)", min_value=0, value=10)

input_df = pd.DataFrame({
    "Age": [Age],
    "CityTier": [CityTier],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfFollowups": [NumberOfFollowups],
    "DurationOfPitch": [DurationOfPitch]
})

if st.button("Predict"):
    result = model.predict(input_df)[0]
    if result == 1:
        st.success("Customer is LIKELY to purchase the Wellness Tourism Package")
    else:
        st.error("Customer is NOT LIKELY to purchase the Wellness Tourism Package")
