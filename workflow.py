import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load Training and Test Data
def load_data():
    training_data = pd.read_csv("data/trainingdata.csv")
    test_data = pd.read_csv("data/testdata.csv")
    mapping = pd.read_csv("data/participant_col_mapping.csv")

    # Map columns for both training and test data
    rename_dict = mapping.set_index("column_name")["description"].to_dict()
    training_data.rename(columns=rename_dict, inplace=True)
    test_data.rename(columns=rename_dict, inplace=True)

    # Explicitly rename 'pin code' to 'pincode' in both training and test data
    if 'pin code' in training_data.columns:
        training_data.rename(columns={'pin code': 'pincode'}, inplace=True)
    if 'pin code' in test_data.columns:
        test_data.rename(columns={'pin code': 'pincode'}, inplace=True)

    # Debugging: Print column names in test_data
    print("Columns in test_data after renaming:", test_data.columns)

    # Verify 'pincode' column exists after renaming
    if 'pincode' not in test_data.columns:
        raise KeyError("The 'pincode' column is missing in the test data after renaming. Please ensure it is included.")
    if 'pincode' not in training_data.columns:
        raise KeyError("The 'pincode' column is missing in the training data after renaming. Please ensure it is included.")

    # Ensure 'target' column exists in training data
    if 'target' not in training_data.columns:
        raise KeyError("The 'target' column is missing in the training data. Please check participant_col_mapping.csv.")

    # Debugging: Inspect shapes and column names of training and test data
    print("Shape of training data:", training_data.shape)
    print("Columns in training data:", training_data.columns)
    print("Shape of test data:", test_data.shape)
    print("Columns in test data:", test_data.columns)

    # Debugging: Compare column counts before and after renaming
    print("Training data column count before renaming:", len(pd.read_csv("data/trainingdata.csv").columns))
    print("Training data column count after renaming:", len(training_data.columns))
    print("Test data column count before renaming:", len(pd.read_csv("data/testdata.csv").columns))
    print("Test data column count after renaming:", len(test_data.columns))

    # Debugging: Compare column names between training and test data
    training_columns = set(training_data.columns)
    test_columns = set(test_data.columns)

    missing_in_training = test_columns - training_columns
    missing_in_test = training_columns - test_columns

    print("Columns missing in training data:", missing_in_training)
    print("Columns missing in test data:", missing_in_test)

    return training_data, test_data

# Step 2: Feature Engineering
# Updated to apply feature engineering only to training data
def feature_engineering(data):
    # Aggregate financial features
    balance_columns = [col for col in data.columns if "balance" in col]
    data['average_balance'] = data[balance_columns].mean(axis=1)
    data['total_balance'] = data[balance_columns].sum(axis=1)
    data['std_balance'] = data[balance_columns].std(axis=1)

    credit_columns = [col for col in data.columns if "credit_limit" in col]
    data['average_credit_limit'] = data[credit_columns].mean(axis=1)
    data['total_credit_limit'] = data[credit_columns].sum(axis=1)

    return data

# Step 3: External Data Integration
# Updated to validate mapping file and handle missing district column
def integrate_external_data(data):
    def load_and_filter_columns(filepath, required_columns):
        df = pd.read_csv(filepath)
        available_columns = [col for col in required_columns if col in df.columns]
        return df[available_columns]

    # Validate mapping file
    pincode_mapping = pd.read_csv("data/pincode_to_district_matched.csv")
    if 'pincode' not in pincode_mapping.columns or 'district' not in pincode_mapping.columns:
        raise KeyError("The mapping file must contain 'pincode' and 'district' columns.")

    # Map pincode to district
    if 'pincode' in data.columns:
        data = data.merge(pincode_mapping, on='pincode', how='left')
    else:
        raise KeyError("The 'pincode' column is missing in the training data. Please ensure it is included.")

    # Ensure 'district' column is of consistent data type (string)
    if 'district' not in data.columns:
        raise KeyError("The 'district' column is missing after mapping pincodes. Please check the mapping file.")

    data['district'] = data['district'].astype(str)

    census = load_and_filter_columns("data/census.csv", ['Name', 'TRU', 'No_HH', 'TOT_P', 'P_LIT'])
    census.rename(columns={'Name': 'district'}, inplace=True)

    npci = load_and_filter_columns("data/npci.csv", ['District', 'Usage'])
    viirs = load_and_filter_columns("data/viirs_matched_cleaned.csv", ['district', 'nightlight'])
    amenities = load_and_filter_columns("data/district_amenity_counts.csv", ['district', 'banks', 'schools', 'post_offices'])

    census['district'] = census['district'].astype(str)
    npci['District'] = npci['District'].astype(str)
    viirs['district'] = viirs['district'].astype(str)
    amenities['district'] = amenities['district'].astype(str)

    # Merge external datasets with training data
    data = data.merge(census, on="district", how="left")
    data = data.merge(npci.rename(columns={'District': 'district'}), on="district", how="left")
    data = data.merge(viirs, on="district", how="left")
    data = data.merge(amenities, on="district", how="left")

    return data

# Step 4: Model Training
# Updated to include reinforcement loop for stability
def train_model(training_data):
    X_train = training_data.drop(columns=['target', 'unique_id'], errors='ignore')
    y_train = training_data['target']

    # Filter numeric columns for scaling and imputation
    numeric_columns_train = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_columns_train]

    # Apply imputation to numeric columns
    imputer = SimpleImputer(strategy="mean")
    X_train_numeric_imputed = imputer.fit_transform(X_train_numeric)

    X_train_numeric = pd.DataFrame(X_train_numeric_imputed, columns=numeric_columns_train)

    # Train model with reinforcement loop
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    for _ in range(3):  # Reinforcement loop
        model.fit(X_train_numeric, y_train)

    joblib.dump(model, "model/trained_model.joblib")
    return model

# Step 5: Model Evaluation
# Updated to include robust evaluation metrics
def evaluate_model(model, test_data):
    X_test = test_data.drop(columns=['target', 'unique_id'], errors='ignore')
    y_test = test_data['target']

    # Filter numeric columns for scaling and imputation
    numeric_columns_test = X_test.select_dtypes(include=['float64', 'int64']).columns
    X_test_numeric = X_test[numeric_columns_test]

    # Apply imputation to numeric columns
    imputer = SimpleImputer(strategy="mean")
    X_test_numeric_imputed = imputer.transform(X_test_numeric)

    X_test_numeric = pd.DataFrame(X_test_numeric_imputed, columns=numeric_columns_test)

    # Predict and evaluate
    y_pred = model.predict(X_test_numeric)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Main Workflow
# Updated to reflect changes in training and evaluation
def main():
    training_data, test_data = load_data()
    training_data = feature_engineering(training_data)
    training_data = integrate_external_data(training_data)
    model = train_model(training_data)
    evaluate_model(model, test_data)

if __name__ == "__main__":
    main()
