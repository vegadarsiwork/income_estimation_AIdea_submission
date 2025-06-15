import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load trained model and feature names
model = joblib.load("model/income_model.joblib")
feature_names = joblib.load("model/feature_names.joblib")

# Categorize features
internal_features = [f for f in feature_names if f not in ["education_score", "digital_adoption", "nightlight"]]
external_features = [f for f in feature_names if f in ["education_score", "digital_adoption", "nightlight"]]

# Get feature importances from model
importances = model.feature_importances_

# Group importances by type
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances,
    "type": ["external" if f in external_features else "internal" for f in feature_names]
})

# Sum total importance by type
importance_by_type = importance_df.groupby("type")["importance"].sum()

# Normalize to percentages
importance_percent = (importance_by_type / importance_by_type.sum()) * 100

# Print result
print("🔍 Weightage of feature types used to predict income:\n")
for ftype, percent in importance_percent.items():
    print(f"{ftype.title()} Features: {percent:.2f}% weight")

# Optional: Top features
print("\n⭐ Top 5 Influential Features:")
print(importance_df.sort_values("importance", ascending=False).head(5))
