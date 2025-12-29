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

st.title("🏖 Tourism Wellness Package Purchase Prediction")
st.write("Fill the details below to predict whether the customer is likely to purchase the package.")

# -------- Inputs --------
Age = st.number_input("Age", min_value=18, max_value=90, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Pitch Duration (Minutes)", min_value=0, value=10)
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, value=1)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.number_input("Trips per Year", min_value=0, value=2)
Passport = st.selectbox("Passport Available", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, value=30000)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female", "Fe Male"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

input_df = pd.DataFrame({
    "Age": [Age],
    "CityTier": [CityTier],
    "DurationOfPitch": [DurationOfPitch],
    "NumberOfPersonVisiting": [NumberOfPersonVisiting],
    "NumberOfFollowups": [NumberOfFollowups],
    "PreferredPropertyStar": [PreferredPropertyStar],
    "NumberOfTrips": [NumberOfTrips],
    "Passport": [Passport],
    "PitchSatisfactionScore": [PitchSatisfactionScore],
    "OwnCar": [OwnCar],
    "NumberOfChildrenVisiting": [NumberOfChildrenVisiting],
    "MonthlyIncome": [MonthlyIncome],
    
    "TypeofContact": [TypeofContact],
    "Occupation": [Occupation],
    "Gender": [Gender],
    "ProductPitched": [ProductPitched],
    "MaritalStatus": [MaritalStatus],
    "Designation": [Designation]
})


if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.success(f"✅ Customer is LIKELY to purchase the package. Confidence: {prob:.2f}")
        else:
            st.error(f"❌ Customer is NOT likely to purchase. Confidence: {prob:.2f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Make sure model pipeline includes preprocessing inside the pickle.")
