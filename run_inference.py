import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer

# Load models
income_model = joblib.load("model/income_model.joblib")
repayment_model = joblib.load("model/repayment_model.joblib")
feature_names = joblib.load("model/feature_names.joblib")

# Load datasets
mapping = pd.read_csv("data/pincode_to_district_matched.csv")
mapping.columns = mapping.columns.str.strip().str.lower()

census = pd.read_csv("data/census.csv")
census.columns = census.columns.str.strip().str.upper()

npci = pd.read_csv("data/npci.csv")
npci.columns = npci.columns.str.strip().str.upper()

viirs = pd.read_csv("data/viirs_matched_cleaned.csv")
viirs.columns = viirs.columns.str.strip().str.lower()

def get_district_from_pincode(pincode: str):
    try:
        pincode = int(pincode)
    except ValueError:
        raise ValueError("❌ Invalid pincode format")

    row = mapping[mapping["pincode"] == pincode]
    if row.empty:
        raise ValueError("❌ Pincode not found in mapping file")
    return row.iloc[0]["district"].strip().lower()

def preprocess_features(district: str):
    # Census
    census_row = census[census["NAME"].str.lower() == district]
    if census_row.empty:
        raise ValueError(f"❌ District '{district}' not found in census.csv")

    # NPCI
    npci_row = npci[npci["DISTRICT"].str.lower() == district]
    if npci_row.empty:
        raise ValueError(f"❌ District '{district}' not found in npci.csv")

    # VIIRS
    viirs_row = viirs[viirs["district"].str.lower() == district]

    # Combine features
    features = {
        "education_score": census_row["LITERACYRATE"].values[0],
        "household_size": census_row["HOUSEHOLDSIZE"].values[0],
        "digital_adoption": npci_row["USAGE"].values[0],
        "nightlight": viirs_row["nightlight"].values[0] if not viirs_row.empty else 0
    }

    # Convert to DataFrame and align columns
    X = pd.DataFrame([features])
    missing_cols = [col for col in feature_names if col not in X.columns]
    missing_data = pd.DataFrame(0, index=X.index, columns=missing_cols)
    X = pd.concat([X, missing_data], axis=1)

    # Ensure column order matches feature_names
    X = X[feature_names]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X

def preprocess_financial_features(df):
    # Aggregate balance
    balance_columns = [col for col in df.columns if "balance" in col]
    df['average_balance'] = df[balance_columns].mean(axis=1)
    df['max_balance'] = df[balance_columns].max(axis=1)
    df['min_balance'] = df[balance_columns].min(axis=1)
    df['std_balance'] = df[balance_columns].std(axis=1)
    df['total_balance'] = df[balance_columns].sum(axis=1)

    # Aggregate credit limit
    credit_columns = [col for col in df.columns if "credit_limit" in col]
    df['average_credit_limit'] = df[credit_columns].mean(axis=1)
    df['max_credit_limit'] = df[credit_columns].max(axis=1)
    df['min_credit_limit'] = df[credit_columns].min(axis=1)
    df['std_credit_limit'] = df[credit_columns].std(axis=1)
    df['total_credit_limit'] = df[credit_columns].sum(axis=1)

    # Credit ratios
    if 'active_credit_limit_1' in df.columns:
        df['active_to_total_ratio'] = df['active_credit_limit_1'] / (df['total_credit_limit'] + 1e-6)

    if 'credit_limit_recent_1' in df.columns:
        df['credit_new_to_old_ratio'] = df['credit_limit_recent_1'] / (df['max_credit_limit'] + 1e-6)

    # Aggregate total EMI
    emi_columns = [col for col in df.columns if "emi" in col]
    df['average_total_emi'] = df[emi_columns].mean(axis=1)
    df['max_total_emi'] = df[emi_columns].max(axis=1)
    df['min_total_emi'] = df[emi_columns].min(axis=1)
    df['std_total_emi'] = df[emi_columns].std(axis=1)
    df['total_total_emi'] = df[emi_columns].sum(axis=1)

    # Aggregate repayments
    repayment_columns = [col for col in df.columns if "repayment" in col]
    df['average_repayment'] = df[repayment_columns].mean(axis=1)
    df['max_repayment'] = df[repayment_columns].max(axis=1)
    df['min_repayment'] = df[repayment_columns].min(axis=1)
    df['std_repayment'] = df[repayment_columns].std(axis=1)
    df['total_repayment'] = df[repayment_columns].sum(axis=1)

    # Debt to repayment ratio
    if 'total_loans_1' in df.columns:
        df['debt_to_repayment_ratio'] = df['total_loans_1'] / (df['total_repayment'] + 1e-6)

    # Aggregate total inquiries
    inquiries_columns = [col for col in df.columns if "inquiries" in col]
    df['average_total_inquiries'] = df[inquiries_columns].mean(axis=1)
    df['max_total_inquiries'] = df[inquiries_columns].max(axis=1)
    df['min_total_inquiries'] = df[inquiries_columns].min(axis=1)
    df['std_total_inquiries'] = df[inquiries_columns].std(axis=1)
    df['total_total_inquiries'] = df[inquiries_columns].sum(axis=1)

    # Closed ratio
    if 'closed_loan' in df.columns and 'total_loans_1' in df.columns:
        df['closed_ratio'] = df['closed_loan'] / (df['total_loans_1'] + 1e-6)

    # Business balance ratio
    if 'business_balance' in df.columns and 'total_balance' in df.columns:
        df['business_balance_ratio'] = df['business_balance'] / (df['total_balance'] + 1e-6)

    return df

def predict_income_and_repayment(pincode: str):
    district = get_district_from_pincode(pincode)
    X = preprocess_features(district)

    predicted_income = income_model.predict(X)[0]
    repayment_score = repayment_model.predict([[predicted_income]])[0]

    return round(predicted_income, 2), round(repayment_score, 2), district.title()

def predict(df: pd.DataFrame) -> pd.DataFrame:
    # Load pre-trained models
    income_model = joblib.load("model/income_model.joblib")
    feature_names = joblib.load("model/feature_names.joblib")

    # Preprocess input DataFrame
    X = pd.get_dummies(df, drop_first=True)  # Encode categorical columns

    # Align columns with training data
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

    # Predict income
    predicted_income = income_model.predict(X)

    # Prepare result DataFrame
    result_df = pd.DataFrame({
        "id": df["id"],
        "predicted_income": predicted_income
    })

    return result_df

if __name__ == "__main__":
    # Load input CSV file
    input_file = "data/testdata.csv"  # Replace with the actual input file path
    df = pd.read_csv(input_file)

    # Generate predictions
    result_df = predict(df)

    # Save predictions to CSV
    result_df.to_csv("output/predictions.csv", index=False)
    print("✅ Predictions saved to output/predictions.csv")
