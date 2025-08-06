import streamlit as st
import pandas as pd
import numpy as np
import joblib


clf = joblib.load("rf_model.sav")
encoder = joblib.load("label_encoder.sav")


def predict(data):
   prediction =  clf.predict(data)
   output = encoder.inverse_transform(prediction)
   return output 


cities = [
    'Berlin', 'Norwich', 'Bridgeport', 'New Britain', 'West Haven', 'Stamford',
    'New Haven', 'North Haven', 'Waterbury', 'Milford', 'Wilton', 'Bristol',
    'Wallingford', 'Newington', 'Manchester', 'Torrington', 'Wethersfield',
    'Norwalk', 'Trumbull', 'Meriden', 'Coventry', 'Middletown', 'Glastonbury',
    'Woodbridge', 'Fairfield', 'Old Saybrook', 'Hartford', 'Enfield', 'Bloomfield',
    'Ansonia', 'West Hartford', 'New London', 'Cromwell', 'Darien', 'Westport',
    'Stratford', 'Suffield', 'Hamden', 'Shelton', 'Winchester', 'Plainville',
    'Danbury', 'East Haven', 'Naugatuck', 'Clinton', 'Wolcott', 'Greenwich',
    'Redding', 'Derby', 'East Windsor', 'South Windsor', 'Willimantic',
    'Southington', 'New Canaan', 'Vernon', 'Middlebury', 'Ridgefield',
    'North Branford', 'Plymouth', 'Windsor', 'Orange', 'Waterford', 'Guilford',
    'East Hartford', 'Portland', 'Newtown', 'Cheshire', 'Windsor Locks',
    'Branford', 'Monroe', 'Groton Town', 'Simsbury', 'Ledyard', 'Canton',
    'Watertown', 'New Milford', 'Thomaston', 'Stonington', 'Seymour', 'Bethel',
    'Brookfield', 'Farmington', 'Avon', 'Easton', 'East Lyme', 'Granby',
    'Rocky Hill', 'Weston', 'Putnam', 'Madison', 'East Hampton', 'Groton',
    'Plainfield', 'Montville', 'Groton Long Point'
]

location_areas = [
    'Abandoned/Condemned Structure', 'Air/Bus/Train Terminal', 'Amusement Park',
    'Arena/Stadium/Fairgrounds/Coliseum', 'ATM Separate from Bank',
    'Auto Dealership New/Used', 'Bank/Savings and Loan', 'Bar/Nightclub',
    'Camp/Campground', 'Church/Synagogue/Temple/Mosque',
    'Commercial/Office Building', 'Community Center', 'Construction Site',
    'Convenience Store', 'Cyberspace', 'Daycare Facility',
    'Department/Discount Store', 'Dock/Wharf/Freight/Modal Terminal',
    "Drug Store/Doctor's Office/Hospital", 'Farm Facility', 'Field/Woods',
    'Gambling Facility/Casino/Race Track', 'Government/Public Building',
    'Grocery/Supermarket', 'Highway/Road/Alley/Street/Sidewalk',
    'Hotel/Motel/Etc.', 'Industrial Site',
    'Jail/Prison/Penitentiary/Corrections Facility', 'Lake/Waterway/Beach',
    'Liquor Store', 'Military Installation', 'Park/Playground',
    'Parking/Drop Lot/Garage', 'Rental Storage Facility', 'Residence/Home',
    'Rest Area', 'Restaurant', 'School/College', 'School-College/University',
    'School-Elementary/Secondary', 'Service/Gas Station',
    'Shelter-Mission/Homeless', 'Shopping Mall', 'Specialty Store',
    'Tribal Lands', 'Other/Unknown'
]


st.title("Crime Prediction")

st.subheader("Enter Crime Details")

year = st.selectbox("Year", [2022, 2023])
month = st.selectbox("Month", list(range(1, 13)))
day = st.selectbox("Day", list(range(1, 32)))
hour = st.selectbox("Hour", list(range(0, 24)))
dayofweek = st.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)))
city = st.selectbox("City", sorted(cities))
location_area = st.selectbox("Location Area", sorted(location_areas))
population = st.number_input("Population", min_value=1000, value=20000)
crime_rate = st.number_input("Crime Rate per 1000 people", min_value=0, value=100)

# Prepare input for prediction
input_df = pd.DataFrame({
    "year": [year],
    "month": [month],
    "day": [day],
    "hour": [hour],
    "dayofweek": [dayofweek],
    "population": [population],
    "crime_rate_per_1000_people": [crime_rate],
    "city": [city],
    "location_area": [location_area]
})



input_encoded =  pd.get_dummies(input_df,columns =["city","location_area"],drop_first= False)

# Align with model's training features
X_train = pd.read_csv("training_columns.csv") # Make sure this file exists with your original column order
model_columns = X_train.columns

# Add any missing columns
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[model_columns]



# Prediction
if st.button("Predict Offense Category"):
    final_prediction = predict(input_encoded)
    st.success(f"Predicted offense category: **{final_prediction[0]}**")