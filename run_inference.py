import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

def generate_predictions(income_model, repayment_model, merged_data):
    X = merged_data
    X = pd.get_dummies(X, drop_first=True)  # Encode categorical columns

    # Load saved feature names
    valid_columns = joblib.load("model/feature_names.joblib")

    # Align columns with training data
    for col in valid_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[valid_columns]

    # Handle missing values
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

    return predictions

if __name__ == "__main__":
    # Load input CSV file
    input_file = "data/testdata.csv"  # Replace with the actual input file path
    df = pd.read_csv(input_file)

    # Load models
    income_model = joblib.load("model/income_model.joblib")
    repayment_model = joblib.load("model/repayment_model.joblib")

    # Generate predictions
    result_df = generate_predictions(income_model, repayment_model, df)

    print("✅ Predictions saved to output/predictions.csv")
