import pandas as pd

# Load feature importance CSV
importance_df = pd.read_csv("output/feature_importance.csv")

# Define external data features
external_features = [
    "education_score", "household_size", "scaled_usage", "district", "TRU", "No_HH", "TOT_P", "P_LIT"
]

# Define test data features
test_data_features = [
    "var_0", "var_1", "var_2", "var_3", "var_4", "var_5", "age", "gender", "city", "state", "residence_ownership"
]

# Group features
external_data_importance = importance_df[importance_df["Feature"].isin(external_features)]
test_data_importance = importance_df[importance_df["Feature"].isin(test_data_features)]

# Save grouped data to CSV files
external_data_importance.to_csv("output/external_data_importance.csv", index=False)
test_data_importance.to_csv("output/test_data_importance.csv", index=False)

print("✅ Grouped feature importance saved to CSV files:")
print("External Data Importance: output/external_data_importance.csv")
print("Test Data Importance: output/test_data_importance.csv")