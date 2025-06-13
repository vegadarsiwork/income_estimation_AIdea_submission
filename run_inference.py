import pandas as pd
import joblib

# Load models
census_model = joblib.load("model/census_model.joblib")
npci_model = joblib.load("model/npci_model.joblib")

# Load datasets
mapping = pd.read_csv("data/pincode_to_district_matched.csv")
mapping.columns = mapping.columns.str.strip().str.lower()

census = pd.read_csv("data/census.csv")
census.columns = census.columns.str.strip().str.upper()

npci = pd.read_csv("data/npci.csv")
npci.columns = npci.columns.str.strip().str.upper()

def get_district_from_pincode(pincode: str):
    try:
        pincode = int(pincode)
    except ValueError:
        raise ValueError("❌ Invalid pincode format")

    row = mapping[mapping["pincode"] == pincode]
    if row.empty:
        raise ValueError("❌ Pincode not found in mapping file")
    return row.iloc[0]["district"].strip().upper()

def get_features(district: str):
    # Census
    census_row = census[census["NAME"] == district]
    if census_row.empty:
        raise ValueError(f"❌ District '{district}' not found in census.csv")
    
    try:
        edu = census_row["LITERACYRATE"].values[0]
        hh = census_row["HOUSEHOLDSIZE"].values[0]
    except KeyError as e:
        raise ValueError(f"❌ Missing column in census.csv: {e}")

    # NPCI
    npci_row = npci[npci["DISTRICT"] == district]
    if npci_row.empty:
        raise ValueError(f"❌ District '{district}' not found in npci.csv")

    try:
        digi = npci_row["USAGE"].values[0]
    except KeyError as e:
        raise ValueError(f"❌ Missing column in npci.csv: {e}")

    return edu, hh, digi

def predict_repayment(pincode: str):
    district = get_district_from_pincode(pincode)
    edu, hh, digi = get_features(district)

    # Build input frames
    census_input = pd.DataFrame([{
        "education_score": edu,
        "household_size": hh
    }])
    npci_input = pd.DataFrame([{
        "digital_adoption": digi
    }])

    census_score = census_model.predict(census_input)[0]
    npci_score = npci_model.predict(npci_input)[0]

    final_score = (0.5 * census_score) + (0.5 * npci_score)
    return round(final_score, 3), district.title()

if __name__ == "__main__":
    user_pin = input("Enter Pincode: ").strip()
    try:
        score, district = predict_repayment(user_pin)
        print(f"\n📍 District: {district}")
        print(f"💡 Repayment Potential Score: {score}")
    except ValueError as e:
        print(e)
