import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# Step 1: Load Training and Test Data
# Updated to use 'pin' instead of 'pincode'
def load_data():
    training_data = pd.read_csv("data/trainingdata.csv")
    test_data = pd.read_csv("data/testdata.csv")
    mapping = pd.read_csv("data/participant_col_mapping.csv")

    # Map columns for both training and test data
    rename_dict = mapping.set_index("column_name")["description"].to_dict()
    training_data.rename(columns=rename_dict, inplace=True)
    test_data.rename(columns=rename_dict, inplace=True)

    # Explicitly rename 'pin code' to 'pin' in both training and test data
    if 'pin code' in training_data.columns:
        training_data.rename(columns={'pin code': 'pin'}, inplace=True)
    if 'pin code' in test_data.columns:
        test_data.rename(columns={'pin code': 'pin'}, inplace=True)

    # Debugging: Print column names in test_data
    print("Columns in test_data after renaming:", test_data.columns)

    # Verify 'pin' column exists after renaming
    if 'pin' not in test_data.columns:
        raise KeyError("The 'pin' column is missing in the test data after renaming. Please ensure it is included.")
    if 'pin' not in training_data.columns:
        raise KeyError("The 'pin' column is missing in the training data after renaming. Please ensure it is included.")

    # Ensure 'target' column exists in training data
    if 'target' not in training_data.columns:
        raise KeyError("The 'target' column is missing in the training data. Please check participant_col_mapping.csv.")

    # Debugging: Inspect shapes and column names of training and test data
    print("Shape of training data:", training_data.shape)
    print("Columns in training data:", training_data.columns)
    print("Shape of test data:", test_data.shape)
    print("Columns in test_data:", test_data.columns)

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
# Updated to use ensemble models (XGBoost, Random Forest, Linear Regression)
def train_income_model(training_data):
    # Load participant column mapping
    mapping = pd.read_csv("data/participant_col_mapping.csv")
    rename_dict = mapping.set_index("column_name")["description"].to_dict()

    # Adjust participant columns to match renamed columns
    participant_columns = [rename_dict.get(col, col) for col in mapping[mapping['description'] != 'target']['column_name']]
    participant_columns = [col if col != 'pin code' else 'pin' for col in participant_columns]

    training_data = training_data[participant_columns + ['target']]

    X_train = training_data.drop(columns=['target'], errors='ignore')
    y_train = training_data['target']

    # Filter numeric columns for imputation
    numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train_numeric = X_train[numeric_columns]

    # Debugging: Print shape of numeric columns
    print("Shape of X_train_numeric:", X_train_numeric.shape)

    # Debugging: Print numeric columns before imputation
    print("Numeric columns before imputation:", numeric_columns)

    # Debugging: Print all columns in X_train_numeric
    print("All columns in X_train_numeric:", X_train_numeric.columns.tolist())

    # Debugging: Check for missing values in X_train_numeric
    missing_values = X_train_numeric.isnull().sum()
    print("Missing values in X_train_numeric:", missing_values[missing_values > 0])

    # Skip imputation if no missing values
    if missing_values.sum() == 0:
        print("No missing values detected. Skipping imputation.")
        X_train_imputed = X_train_numeric.values
    else:
        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X_train_imputed = imputer.fit_transform(X_train_numeric)

    # Debugging: Print shape of imputed data
    print("Shape of X_train_imputed after imputation:", X_train_imputed.shape)

    # Debugging: Identify extra columns in X_train_numeric
    extra_columns = set(X_train_numeric.columns) - set(numeric_columns)
    print("Extra columns in X_train_numeric:", extra_columns)

    # Validate numeric_columns
    if len(numeric_columns) != 78:
        print("Warning: numeric_columns does not match the expected count of 78.")

    # Convert to DataFrame to retain column names
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=numeric_columns)

    # Remove duplicate columns from numeric_columns
    numeric_columns = numeric_columns.drop_duplicates()

    # Debugging: Print numeric columns after removing duplicates
    print("Numeric columns after removing duplicates:", numeric_columns.tolist())

    # Filter imputed data to retain only numeric columns
    X_train_imputed = pd.DataFrame(X_train_imputed, columns=numeric_columns)
    X_train_imputed = X_train_imputed[numeric_columns.intersection(X_train_imputed.columns)]

    # Debugging: Print columns after filtering
    print("Columns after filtering imputed data:", X_train_imputed.columns.tolist())

    # Debugging: Print shape and columns of X_train_imputed after filtering
    print("Shape of X_train_imputed after filtering:", X_train_imputed.shape)
    print("Columns in X_train_imputed after filtering:", X_train_imputed.columns.tolist())

    # Validate feature names before fitting
    valid_features = [col for col in numeric_columns if col in X_train_imputed.columns]
    invalid_features = [col for col in numeric_columns if col not in X_train_imputed.columns]

    print("Valid features for training:", valid_features)
    print("Invalid features excluded from training:", invalid_features)

    X_train_valid = X_train_imputed[valid_features]

    # Train ensemble models
    xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    lr_model = LinearRegression()

    xgb_model.fit(X_train_valid, y_train)
    rf_model.fit(X_train_valid, y_train)
    lr_model.fit(X_train_valid, y_train)

    # Save models
    joblib.dump(xgb_model, "model/xgb_income_model.joblib")
    joblib.dump(rf_model, "model/rf_income_model.joblib")
    joblib.dump(lr_model, "model/lr_income_model.joblib")

    # Save feature names
    joblib.dump(valid_features, "model/feature_names.joblib")

    return xgb_model, rf_model, lr_model, X_train_valid, y_train

def train_repayment_model(xgb_model, rf_model, lr_model, X_train_imputed, y_train):
    # Predict income using ensemble models
    xgb_pred = xgb_model.predict(X_train_imputed)
    rf_pred = rf_model.predict(X_train_imputed)
    lr_pred = lr_model.predict(X_train_imputed)

    # Combine predictions for repayment model training
    combined_predictions = pd.DataFrame({
        "xgb_pred": xgb_pred,
        "rf_pred": rf_pred,
        "lr_pred": lr_pred
    })

    repayment_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    repayment_model.fit(combined_predictions, y_train)

    joblib.dump(repayment_model, "model/repayment_model.joblib")
    return repayment_model

# Step 5: Model Evaluation
# Updated to include robust evaluation metrics
def evaluate_model(model, test_data):
    # Load feature names used during training
    feature_names = joblib.load("model/feature_names.joblib")

    # Remove duplicates from feature names
    feature_names = list(dict.fromkeys(feature_names))

    # Select numeric columns from test data
    X_test_numeric = test_data.select_dtypes(include=['float64', 'int64'])

    # Ensure no duplicate columns in test data
    X_test_numeric = X_test_numeric.loc[:, ~X_test_numeric.columns.duplicated()]

    # Debugging: Print feature names before alignment
    print("Feature names in test data before alignment:", X_test_numeric.columns.tolist())

    # Debugging: Print feature names loaded from training
    print("Feature names loaded from training:", feature_names)

    # Debugging: Print numeric columns in test data before alignment
    print("Numeric columns in test data before alignment:", X_test_numeric.columns.tolist())

    # Load participant column mapping
    mapping = pd.read_csv("data/participant_col_mapping.csv")
    rename_dict = mapping.set_index("column_name")["description"].to_dict()

    # Apply the mapping to rename test data columns
    X_test_numeric.rename(columns=rename_dict, inplace=True)

    # Debugging: Print renamed test data columns
    print("Test data columns after renaming using participant_col_mapping:", X_test_numeric.columns.tolist())

    # Align test data columns with training feature names
    aligned_columns = [col for col in feature_names if col in X_test_numeric.columns]
    missing_columns = [col for col in feature_names if col not in X_test_numeric.columns]

    # Debugging: Print aligned and missing columns
    print("Aligned columns:", aligned_columns)
    print("Missing columns:", missing_columns)

    # Reindex test data to match training feature names
    X_test_numeric = X_test_numeric.reindex(columns=aligned_columns, fill_value=0)

    # Derive new features using mathematical operations
    balance_columns = [col for col in X_test_numeric.columns if 'balance' in col]
    credit_limit_columns = [col for col in X_test_numeric.columns if 'credit_limit' in col]

    X_test_numeric['average_balance'] = X_test_numeric[balance_columns].mean(axis=1)
    X_test_numeric['total_balance'] = X_test_numeric[balance_columns].sum(axis=1)
    X_test_numeric['std_balance'] = X_test_numeric[balance_columns].std(axis=1)

    X_test_numeric['average_credit_limit'] = X_test_numeric[credit_limit_columns].mean(axis=1)
    X_test_numeric['total_credit_limit'] = X_test_numeric[credit_limit_columns].sum(axis=1)

    # Validate data types before applying imputation
    if X_test_numeric.empty:
        raise ValueError("Test data is empty after reindexing. Ensure the test data contains valid numeric columns.")

    if not all(X_test_numeric.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number))):
        raise ValueError("Test data contains non-numeric values. Ensure all columns are numeric before applying imputation.")

    # Apply imputation to numeric columns
    imputer = SimpleImputer(strategy="mean")
    try:
        X_test_numeric_imputed = pd.DataFrame(imputer.fit_transform(X_test_numeric), columns=X_test_numeric.columns)
    except ValueError as e:
        print("Error during imputation:", e)
        print("Ensure the test data matches the expected input format.")
        return None

    # Debugging: Print feature names after imputation
    print("Feature names after imputation:", X_test_numeric_imputed.columns.tolist())

    # Predict using the model
    try:
        y_pred = model.predict(X_test_numeric_imputed)
    except ValueError as e:
        print(f"Error during prediction: {e}")
        print("Ensure the feature dimensions match the model's expected input.")
        return None

    mae = mean_absolute_error(test_data['target'], y_pred)
    mse = mean_squared_error(test_data['target'], y_pred)
    r2 = r2_score(test_data['target'], y_pred)

    print(f"📊 Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

def evaluate_repayment_model(repayment_model, test_data, xgb_model, rf_model, lr_model):
    X_test = test_data.drop(columns=['target', 'unique_id'], errors='ignore')

    # Filter numeric columns for scaling and imputation
    numeric_columns_test = X_test.select_dtypes(include=['float64', 'int64']).columns
    X_test_numeric = X_test[numeric_columns_test]

    # Load the imputer used during training
    imputer = joblib.load("model/income_imputer.joblib")
    X_test_numeric_imputed = imputer.transform(X_test_numeric)

    # Ensure feature names are strings for XGBoost compatibility
    X_test_numeric_imputed = pd.DataFrame(X_test_numeric_imputed, columns=[str(col) for col in numeric_columns_test])

    # Debugging: Print feature names passed to XGBoost
    print("Feature names passed to XGBoost:", X_test_numeric_imputed.columns.tolist())

    # Generate predictions from ensemble models
    xgb_pred = xgb_model.predict(X_test_numeric_imputed)
    rf_pred = rf_model.predict(X_test_numeric_imputed)
    lr_pred = lr_model.predict(X_test_numeric_imputed)

    # Combine predictions for repayment model evaluation
    combined_predictions = pd.DataFrame({
        "xgb_pred": xgb_pred,
        "rf_pred": rf_pred,
        "lr_pred": lr_pred
    })

    # Debugging: Print shape of combined predictions
    print("Shape of combined_predictions:", combined_predictions.shape)

    # Predict repayment scores
    try:
        y_pred = repayment_model.predict(combined_predictions)
    except ValueError as e:
        print("Error during repayment model prediction:", e)
        print("Expected input shape:", repayment_model.n_features_in_ if hasattr(repayment_model, 'n_features_in_') else "Unknown")
        raise

    # Evaluate repayment model
    y_test = test_data['target']
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Repayment Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

# Main Workflow
# Updated to train separate models and exclude external data
def main():
    training_data, test_data = load_data()
    training_data = feature_engineering(training_data)

    # Train income model
    xgb_model, rf_model, lr_model, X_train_imputed, y_train = train_income_model(training_data)

    # Train repayment model
    repayment_model = train_repayment_model(xgb_model, rf_model, lr_model, X_train_imputed, y_train)

    # Evaluate models
    evaluate_model(xgb_model, test_data)
    evaluate_model(rf_model, test_data)
    evaluate_model(lr_model, test_data)
    evaluate_model(repayment_model, test_data)
    evaluate_repayment_model(repayment_model, test_data, xgb_model, rf_model, lr_model)

if __name__ == "__main__":
    main()
