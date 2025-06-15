import pandas as pd

# Load saved feature importances
importance_file = "model/income_feature_importance.csv"

try:
    df = pd.read_csv(importance_file)
    print("📊 Top 15 Features by Importance:\n")
    print(df.sort_values(by="importance", ascending=False).head(15).to_string(index=False))
except FileNotFoundError:
    print("❌ Feature importance file not found. Run pipeline.py first to train and save models.")
