import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os

# Load and preprocess data
def load_and_preprocess_data():
    testdata = pd.read_csv("data/trainingdata.csv")
    mapping = pd.read_csv("data/participant_col_mapping.csv")
    pincode_mapping = pd.read_csv("data/pincode_to_district_matched.csv")

    testdata = testdata.merge(pincode_mapping[['pincode', 'district']], left_on='pin', right_on='pincode', how='left')
    if 'district' not in testdata.columns or testdata['district'].isnull().all():
        print("Warning: 'district' column is missing or empty. Proceeding with AI-based predictions only.")
        testdata['district'] = None

    for col in mapping.columns:
        if col in testdata.columns:
            mapping_dict = mapping.set_index("code")["value"].to_dict()
            testdata[col] = testdata[col].map(mapping_dict)

    census = pd.read_csv("data/census.csv")
    npci = pd.read_csv("data/npci.csv")
    viirs = pd.read_csv("data/viirs_matched_cleaned.csv")

    scaler = StandardScaler()
    npci["scaled_usage"] = scaler.fit_transform(npci["Usage"].values.reshape(-1, 1))
    census = census.rename(columns={"LITERACYRATE": "education_score", "HOUSEHOLDSIZE": "household_size"})

    return testdata, census, npci, viirs

# Merge datasets
def merge_datasets(testdata, census, npci, viirs):
    if 'district' not in testdata.columns or testdata['district'].isnull().all():
        print("No district data available. Falling back to 'state' for merging.")
        merged = testdata.merge(census.rename(columns={'Name': 'state'}), on="state", how="left")
        merged = merged.merge(npci.rename(columns={'State': 'state'}), on="state", how="left")
        merged = merged.merge(viirs.rename(columns={'district': 'state'}), on="state", how="left")
    else:
        merged = testdata.merge(census.rename(columns={'Name': 'district'}), on="district", how="left")
        merged = merged.merge(npci.rename(columns={'District': 'district'}), on="district", how="left")
        merged = merged.merge(viirs, on="district", how="left")
    return merged

# Feature engineering
def preprocess_financial_features(df):
    balance_columns = [col for col in df.columns if "balance" in col]
    df['average_balance'] = df[balance_columns].mean(axis=1)
    df['max_balance'] = df[balance_columns].max(axis=1)
    df['min_balance'] = df[balance_columns].min(axis=1)
    df['std_balance'] = df[balance_columns].std(axis=1)
    df['total_balance'] = df[balance_columns].sum(axis=1)

    credit_columns = [col for col in df.columns if "credit_limit" in col]
    df['average_credit_limit'] = df[credit_columns].mean(axis=1)
    df['max_credit_limit'] = df[credit_columns].max(axis=1)
    df['min_credit_limit'] = df[credit_columns].min(axis=1)
    df['std_credit_limit'] = df[credit_columns].std(axis=1)
    df['total_credit_limit'] = df[credit_columns].sum(axis=1)

    if 'active_credit_limit_1' in df.columns:
        df['active_to_total_ratio'] = df['active_credit_limit_1'] / (df['total_credit_limit'] + 1e-6)

    if 'credit_limit_recent_1' in df.columns:
        df['credit_new_to_old_ratio'] = df['credit_limit_recent_1'] / (df['max_credit_limit'] + 1e-6)

    emi_columns = [col for col in df.columns if "emi" in col]
    df['average_total_emi'] = df[emi_columns].mean(axis=1)
    df['max_total_emi'] = df[emi_columns].max(axis=1)
    df['min_total_emi'] = df[emi_columns].min(axis=1)
    df['std_total_emi'] = df[emi_columns].std(axis=1)
    df['total_total_emi'] = df[emi_columns].sum(axis=1)

    repayment_columns = [col for col in df.columns if "repayment" in col]
    df['average_repayment'] = df[repayment_columns].mean(axis=1)
    df['max_repayment'] = df[repayment_columns].max(axis=1)
    df['min_repayment'] = df[repayment_columns].min(axis=1)
    df['std_repayment'] = df[repayment_columns].std(axis=1)
    df['total_repayment'] = df[repayment_columns].sum(axis=1)

    if 'total_loans_1' in df.columns:
        df['debt_to_repayment_ratio'] = df['total_loans_1'] / (df['total_repayment'] + 1e-6)

    inquiries_columns = [col for col in df.columns if "inquiries" in col]
    df['average_total_inquiries'] = df[inquiries_columns].mean(axis=1)
    df['max_total_inquiries'] = df[inquiries_columns].max(axis=1)
    df['min_total_inquiries'] = df[inquiries_columns].min(axis=1)
    df['std_total_inquiries'] = df[inquiries_columns].std(axis=1)
    df['total_total_inquiries'] = df[inquiries_columns].sum(axis=1)

    if 'closed_loan' in df.columns and 'total_loans_1' in df.columns:
        df['closed_ratio'] = df['closed_loan'] / (df['total_loans_1'] + 1e-6)

    if 'business_balance' in df.columns and 'total_balance' in df.columns:
        df['business_balance_ratio'] = df['business_balance'] / (df['total_balance'] + 1e-6)

    return df

# Train models
def train_models(merged_data):
    # Feature engineering
    merged_data = preprocess_financial_features(merged_data)

    # Prepare X, y
    X = merged_data.drop(columns=['target_income'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(X)
    valid_columns = [col for col, valid in zip(X.columns, imputer.statistics_) if not pd.isna(valid)]
    X = pd.DataFrame(imputed_data, columns=valid_columns)

    y = merged_data['target_income']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForest Regressor
    income_model = RandomForestRegressor(n_estimators=100, random_state=42)
    income_model.fit(X_train, y_train)

    # Overwrite old models and features
    os.makedirs("model", exist_ok=True)
    joblib.dump(income_model, "model/income_model.joblib")
    joblib.dump(valid_columns, "model/feature_names.joblib")

    # Optional: Save feature importances
    importances = income_model.feature_importances_
    pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False).to_csv("model/income_feature_importance.csv", index=False)

    # Repayment model
    predicted_income = income_model.predict(X_train).reshape(-1, 1)
    repayment_model = RandomForestRegressor(n_estimators=100, random_state=42)
    repayment_model.fit(predicted_income, y_train)
    joblib.dump(repayment_model, "model/repayment_model.joblib")

    return income_model, repayment_model

# Generate predictions
def generate_predictions(income_model, repayment_model, merged_data):
    X = pd.get_dummies(merged_data, drop_first=True)
    if 'id' in X.columns:
        X = X.drop(columns=['id', 'city'])

    valid_columns = joblib.load("model/feature_names.joblib")

    for col in valid_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[valid_columns]

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=valid_columns)

    predicted_income = income_model.predict(X)
    repayment_score = repayment_model.predict(predicted_income.reshape(-1, 1))

    predictions = pd.DataFrame({
        "id": merged_data["id"],
        "predicted_income": predicted_income,
        "repayment_score": repayment_score
    })

    predictions.to_csv("output/predictions.csv", index=False)

# Main
def main():
    testdata, census, npci, viirs = load_and_preprocess_data()
    merged_data = merge_datasets(testdata, census, npci, viirs)
    income_model, repayment_model = train_models(merged_data)
    generate_predictions(income_model, repayment_model, merged_data)

if __name__ == "__main__":
    main()
