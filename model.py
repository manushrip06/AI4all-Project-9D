import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import joblib

df = pd.read_csv('data.csv')
#df_sampled = df.sample(frac=0.1, random_state=42)
encoder = LabelEncoder()
#df["city"] = LabelEncoder().fit_transform(df["city"])
#df["location_area"] = LabelEncoder().fit_transform(df["location_area"])
df["offense_category_name"] = encoder.fit_transform(df["offense_category_name"] )

#X = df[["city","location_area","location_area","population"]]
#y = df["offense_category_name"]

X = pd.get_dummies(df.drop(columns=["offense_category_name","date","offense_name"]),columns =["city","location_area"],drop_first= True)
# Save training columns
pd.DataFrame(columns=X.columns).to_csv("training_columns.csv", index=False)
X = X.astype(float)
y = df["offense_category_name"]

print(X.dtypes)  # Check data types
print(X.isnull().sum())  # Check for missing values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Try changing the test_size parameter!!

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Train the Decision Tree classifier
clf = RandomForestClassifier(n_estimators=100,  criterion='entropy', max_depth=20,
                             min_samples_split=0.01, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='sqrt',
                             max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                             n_jobs=None, random_state=42, verbose=0, warm_start=False, class_weight=None,
                             ccp_alpha=0.0001, max_samples=None, monotonic_cst=None)

clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)
y_results = encoder.inverse_transform(y_pred)
print(y_results)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: \n{accuracy*100:.2f}% \n")

joblib.dump(clf, "rf_model.sav")
joblib.dump(encoder, "label_encoder.sav")