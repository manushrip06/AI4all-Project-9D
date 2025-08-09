import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="CT Crime Prediction", layout="wide")
st.title("🕵️‍♂️ Connecticut Crime Prediction App")

# --- Load Model ---
model_path = 'lgbm_model.pkl'
columns_path = 'ohe_columns.pkl'

if not os.path.exists(model_path) or not os.path.exists(columns_path):
    st.error(f"""
    ❌ Model files not found!

    Please make sure the following files are in the same folder as this app:
    - `{model_path}`
    - `{columns_path}`

    You can generate them using your LightGBM training script like this:

    ```python
    import joblib
    joblib.dump(model, 'lgbm_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'ohe_columns.pkl')
    ```

    Then restart the app.
    """)
    st.stop()

# --- Load Model and Columns ---
model = joblib.load(model_path)
ohe_columns = joblib.load(columns_path)

# --- Load Data ---
try:
    df = pd.read_csv('combined_data.csv')
except FileNotFoundError:
    st.error("📂 [combined_data.csv](http://_vscodecontentref_/0) not found. Please place the dataset in the same directory.")
    st.stop()

# --- User Inputs ---
st.header("🔍 Enter Crime Details")
col1, col2, col3 = st.columns(3)

year = col1.selectbox("Year", sorted(df['year'].dropna().unique()))
hour = col2.slider("Hour of the Day", 0, 23, 12)
population = col3.number_input("Population of Area", min_value=1, value=10000)

city = st.selectbox("City", sorted(df['city'].dropna().unique()))
location_area = st.selectbox("Location Area", sorted(df['location_area'].dropna().unique()))

# --- Prepare Input Data ---
input_df = pd.DataFrame({
    'year': [year],
    'hour': [hour],
    'population': [population],
    'city': [city],
    'location_area': [location_area]
})

input_df['log_population'] = np.log1p(input_df['population'])
input_df.drop('population', axis=1, inplace=True)

input_encoded = pd.get_dummies(input_df, columns=['city', 'location_area'])

# Ensure all expected columns are present
for col in ohe_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

input_encoded = input_encoded[ohe_columns]

# --- Predict ---
prediction = model.predict(input_encoded)[0]
st.subheader("🎯 Predicted Crime Category")
st.success(f"**{prediction}**")

# --- Map Visualization ---
st.header("🗺️ Connecticut Crime Map (Simulated)")
st.markdown("Crime-prone areas based on historical trends (simulated for demo).")

# Simulated coordinates centered on Connecticut
np.random.seed(42)
map_data = pd.DataFrame(
    np.random.randn(200, 2) / [20, 20] + [41.6, -72.7],  # Centered on CT
    columns=['lat', 'lon']
)
map_data['offense_category_name'] = np.random.choice(df['offense_category_name'].dropna().unique(), size=200)

st.map(map_data[['lat', 'lon']])

# Simulated coordinates (you can replace this with real lat/lon from your dataset)
np.random.seed(42)
map_data = pd.DataFrame({
    'lat': np.random.uniform(41.3, 42.1, size=200),
    'lon': np.random.uniform(-73.7, -71.8, size=200),
    'offense_category_name': np.random.choice(df['offense_category_name'].dropna().unique(), size=200)
})

st.map(map_data[['lat', 'lon']])