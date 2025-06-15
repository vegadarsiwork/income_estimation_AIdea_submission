import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

# ------------------------
# MODEL 3: Merged Census and NPCI
# ------------------------
merged = pd.merge(census, npci, left_index=True, right_index=True, how="outer")
merged.fillna(0, inplace=True)  # Fill missing values with 0

# Update column name for target income
merged.rename(columns={'target': 'target_income'}, inplace=True)

X_merged = merged.drop(columns=['target_income'], errors='ignore')
y_merged = merged['target_income']

# Train and reinforce model
model = GradientBoostingRegressor()
X_train, X_test, y_train, y_test = train_test_split(X_merged, y_merged, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Reinforce model
for _ in range(3):  # Reinforcement loop
    model.fit(X_train, y_train)

joblib.dump(model, "model/merged_model.joblib")
print(f"✅ Trained merged model on {len(merged)} rows.")

# ------------------------
# MODEL 4: Income and Repayment Models
# ------------------------
# Preprocess financial features
balance_columns = [col for col in merged.columns if "balance" in col]
merged['average_balance'] = merged[balance_columns].mean(axis=1)
merged['total_balance'] = merged[balance_columns].sum(axis=1)
merged['std_balance'] = merged[balance_columns].std(axis=1)

credit_columns = [col for col in merged.columns if "credit_limit" in col]
merged['total_credit_limit'] = merged[credit_columns].sum(axis=1)
merged['average_credit_limit'] = merged[credit_columns].mean(axis=1)

# Replace placeholder feature names with meaningful names
valid_columns = ['balance_1', 'balance_2', 'credit_limit_1', 'credit_limit_2', 'balance_3', 'pin', 'credit_limit_3', 'loan_amt_1', 'loan_amt_2', 'business_balance', 'total_emi_1', 'active_credit_limit_1', 'credit_limit_recent_1', 'credit_limit_4', 'loan_amt_large_tenure', 'primary_loan_amt', 'total_inquiries_1', 'total_inquiries_2', 'total_emi_2', 'balance_4', 'balance_5', 'loan_amt_3', 'balance_6', 'credit_limit_5', 'credit_limit_6', 'loan_amt_recent', 'total_inquiries_recent', 'credit_limit_7', 'credit_limit_8', 'age', 'credit_limit_9', 'credit_limit_10', 'balance_7', 'loan_amt_4', 'credit_score', 'credit_limit_11', 'balance_8', 'balance_9', 'loan_amt_5', 'repayment_1', 'balance_10', 'loan_amt_6', 'closed_loan', 'total_emi_3', 'loan_amt_7', 'total_emi_4', 'credit_limit_12', 'total_inquires_3', 'total_emi_5', 'credit_limit_13', 'repayment_2', 'repayment_3', 'repayment_4', 'total_emi_6', 'repayment_5', 'total_loans_1', 'closed_total_loans', 'repayment_6', 'total_emi_7', 'total_loans_2', 'total_inquires_4', 'balance_11', 'total_inquires_5', 'total_loan_recent', 'total_loans_3', 'total_loans_4', 'loan_amt_8', 'total_loans_5', 'repayment_7', 'balance_12', 'repayment_8', 'repayment_9', 'total_inquires_6', 'loan_amt_9', 'repayment_10']

# Ensure target_income is used with limited weightage
X = merged.drop(columns=['target_income'], errors='ignore')
X = pd.get_dummies(X, drop_first=True)
X = X[valid_columns]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=valid_columns)

# Save feature names for consistent preprocessing
joblib.dump(valid_columns, "model/feature_names.joblib")

# Train income model
income_model = GradientBoostingRegressor(
    n_estimators=300,         # Increase number of trees
    learning_rate=0.05,       # Lower learning rate for better generalization
    max_depth=4,              # Limit tree depth
    min_samples_split=10,     # Require more samples to split
    min_samples_leaf=5,       # Require more samples in leaf nodes
    random_state=42           # Ensure reproducibility
)
X_train, X_test, y_train, y_test = train_test_split(X, merged['target_income'], test_size=0.2)
income_model.fit(X_train, y_train)
joblib.dump(income_model, "model/income_model.joblib")

# Predict income for repayment model training
predicted_income = income_model.predict(X_train)

# Train repayment model
repayment_model = GradientBoostingRegressor()
repayment_model.fit(predicted_income.reshape(-1, 1), y_train)
joblib.dump(repayment_model, "model/repayment_model.joblib")
