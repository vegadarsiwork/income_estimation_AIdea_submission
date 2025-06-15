import pandas as pd

def col_mapping():
    # Load testdata and participant_col_mapping
    testdata = pd.read_csv("data/testdata.csv")
    mapping = pd.read_csv("data/participant_col_mapping.csv")

    # Ensure mapping has the required columns
    if 'column_name' not in mapping.columns or 'description' not in mapping.columns:
        raise ValueError("Mapping file must contain 'column_name' and 'description' columns.")

    # Rename columns in testdata based on mapping
    rename_dict = mapping.set_index("column_name")["description"].to_dict()
    testdata.rename(columns=rename_dict, inplace=True)

    # Explicitly rename 'pin code' to 'pincode' in testdata
    if 'pin code' in testdata.columns:
        testdata.rename(columns={'pin code': 'pincode'}, inplace=True)

    # Debugging: Verify 'pincode' column exists after renaming
    if 'pincode' not in testdata.columns:
        raise KeyError("The 'pincode' column is missing in the test data after renaming. Please ensure it is included.")

    # Save the updated testdata to a new CSV file
    testdata.to_csv("data/mapped_testdata.csv", index=False)
    print("✅ Renamed columns in testdata and saved to data/mapped_testdata.csv")

col_mapping()