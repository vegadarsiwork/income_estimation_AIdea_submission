import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

os.makedirs("model", exist_ok=True)

# ------------------------
# MODEL 1: Census-based
# ------------------------
census = pd.read_csv("data/census.csv")

# Clean and extract features
census = census[census["TRU"].str.lower() == "total"]
census.rename(columns={
    "LITERACYRATE": "education_score",
    "HOUSEHOLDSIZE": "household_size"
}, inplace=True)

census = census[["education_score", "household_size"]].dropna().reset_index(drop=True)
X_census = census
y_census = np.random.uniform(0.2, 0.95, size=len(X_census))  # Fake repayment score

# Train and save model
census_model = GradientBoostingRegressor()
X_train, X_test, y_train, y_test = train_test_split(X_census, y_census, test_size=0.2)
census_model.fit(X_train, y_train)
joblib.dump(census_model, "model/census_model.joblib")
print(f"✅ Trained census model on {len(X_census)} rows.")

# ------------------------
# MODEL 2: NPCI-based
# ------------------------
npci = pd.read_csv("data/npci.csv")

X_npci = npci[["Usage"]].rename(columns={"Usage": "digital_adoption"}).dropna().reset_index(drop=True)
y_npci = np.random.uniform(0.2, 0.95, size=len(X_npci))  # Fake repayment score

# Train and save model
npci_model = GradientBoostingRegressor()
X_train, X_test, y_train, y_test = train_test_split(X_npci, y_npci, test_size=0.2)
npci_model.fit(X_train, y_train)
joblib.dump(npci_model, "model/npci_model.joblib")
print(f"✅ Trained NPCI model on {len(X_npci)} rows.")
