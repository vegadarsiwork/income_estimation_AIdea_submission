import pandas as pd

# Load and preprocess data
def load_and_preprocess_data():
    """
    Load and preprocess data from various sources.
    Returns:
        testdata (DataFrame): Preprocessed test data with renamed columns.
        census (DataFrame): Census data.
        npci (DataFrame): NPCI data.
        viirs (DataFrame): VIIRS data.
        amenities (DataFrame): Amenities data.
    """
    # Load test data
    testdata = pd.read_csv("data/trainingdata.csv")

    # Load column mapping
    mapping_df = pd.read_csv("data/column_mapping.csv")
    rename_dict = dict(zip(mapping_df["column_name"], mapping_df["description"]))
    testdata.rename(columns=rename_dict, inplace=True)

    # Load external datasets
    census = pd.read_csv("data/census.csv")
    npci = pd.read_csv("data/npci.csv")
    viirs = pd.read_csv("data/viirs_matched_cleaned.csv")
    amenities = pd.read_csv("data/district_amenity_counts.csv")

    # Preprocess external datasets
    census.rename(columns={'Name': 'district'}, inplace=True)
    npci.rename(columns={'District': 'district'}, inplace=True)

    # Ensure consistent data types for merging
    for df in [census, npci, viirs, amenities]:
        df.rename(columns=lambda x: x.strip().lower(), inplace=True)
        if 'district' not in df.columns:
            raise ValueError("Missing 'district' column in one of the external datasets.")
        df['district'] = df['district'].astype(str)

    return testdata, census, npci, viirs, amenities


def apply_feature_engineering_and_weightages(data):
    weightages = {
        'average_balance': 2.5,
        'total_balance': 2.5,
        'std_balance': 2,
        'total_credit_limit': 2,
        'average_credit_limit': 2,
        'active_to_total': 4,
        'average_total_emi': 2.5,
        'total_total_emi': 2.5,
        'total_loan_amt': 2,
        'average_loan_amt': 2,
        'std_loan_amt': 2,
        'total_total_loans': 1.5,
        'average_total_loans': 1.5,
        'closed_ratio': 3,
        'timeliness_ratio_repayment': 7,
        'total_repayment': 1,
        'average_repayment': 1,
        'increase_ratio_inquiries': 3,
        'loan_new_ratio': 2,
        'digital_adoption': 8,
        'literacy_rate': 5,
        'viirs_night_light_data': 6,
        'osm_poi_density': 6,
        'age': 4,
        'gender': 1,
        'marital_status': 2,
        'residence_ownership': 4,
        'city': 1.5,
        'state': 1.5,
        'device_model': 2,
        'device_category': 2,
        'platform': 1,
        'device_manufacturer': 1
    }

    for feature, weight in weightages.items():
        if feature in data.columns:
            data[f'weighted_{feature}'] = data[feature] * weight

    return data


# External Data Integration
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

    # Ensure 'district' column exists and is of type string
    if 'district' not in data.columns:
        raise KeyError("The 'district' column is missing after mapping pincodes. Please check the mapping file.")
    data['district'] = data['district'].astype(str)

    # Load external datasets
    census = load_and_filter_columns("data/census.csv", ['Name', 'TRU', 'No_HH', 'TOT_P', 'P_LIT'])
    census.rename(columns={'Name': 'district'}, inplace=True)

    npci = load_and_filter_columns("data/npci.csv", ['District', 'Usage'])
    viirs = load_and_filter_columns("data/viirs_matched_cleaned.csv", ['district', 'nightlight'])
    amenities = load_and_filter_columns("data/district_amenity_counts.csv", ['district', 'banks', 'schools', 'post_offices'])

    # Standardize column names and types
    census['district'] = census['district'].astype(str)
    npci['District'] = npci['District'].astype(str)
    viirs['district'] = viirs['district'].astype(str)
    amenities['district'] = amenities['district'].astype(str)

    # Merge external datasets with main data
    data = data.merge(census, on="district", how="left")
    data = data.merge(npci.rename(columns={'District': 'district'}), on="district", how="left")
    data = data.merge(viirs, on="district", how="left")
    data = data.merge(amenities, on="district", how="left")

    return data
