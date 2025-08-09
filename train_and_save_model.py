import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv('combined_data.csv')

target_col = 'offense_category_name'

if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in combined_data.csv. Available columns: {df.columns.tolist()}")

# Drop missing targets
df.dropna(subset=[target_col], inplace=True)

# Use columns that exist in your CSV
feature_cols = ['year', 'hour', 'population', 'city', 'location_area']

for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in combined_data.csv. Available columns: {df.columns.tolist()}")

X = df[feature_cols].copy()
y = df[target_col]

X['log_population'] = np.log1p(X['population'])
X.drop('population', axis=1, inplace=True)

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=['city', 'location_area'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Save model and column names
joblib.dump(model, 'lgbm_model.pkl')
joblib.dump(X_train.columns.tolist(), 'ohe_columns.pkl')

print("✅ Model and column files saved successfully.")