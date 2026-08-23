
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_tourism_package_model_v1.joblib"
)

model = joblib.load(model_path)

st.title("Tourism Package Prediction App")

st.write("""
This application predicts whether a customer is likely to purchase the newly
introduced Wellness Tourism Package based on their demographic and interaction details.
""")

# -----------------------------
# Customer Information
# -----------------------------

Age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35,
    step=1
)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0,
    max_value=60,
    value=10,
    step=1
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0,
    max_value=1000000,
    value=25000,
    step=1000
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox(
    "Preferred Property Star Rating",
    [3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips",
    min_value=0,
    max_value=30,
    value=2,
    step=1
)

Passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

OwnCar = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# -----------------------------
# Prepare Input Data
# -----------------------------

input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "Designation": Designation
}])

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Customer is likely to purchase the Tourism Package"
        st.success(f"**Prediction:** {result}")
    else:
        result = "Customer is unlikely to purchase the Tourism Package"
        st.warning(f"**Prediction:** {result}")
