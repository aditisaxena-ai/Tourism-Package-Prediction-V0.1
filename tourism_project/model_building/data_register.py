
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

# Ensure the data file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Define expected columns
expected_columns = [
    'Unnamed: 0',
    'CustomerID',
    'ProdTaken',
    'Age',
    'TypeofContact',
    'CityTier',
    'DurationOfPitch',
    'Occupation',
    'Gender',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'ProductPitched',
    'PreferredPropertyStar',
    'MaritalStatus',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'Designation',
    'MonthlyIncome'
]

# Verify expected columns are present
missing_cols = [
    col for col in expected_columns
    if col not in df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing expected columns in {DATA_PATH}: {missing_cols}"
    )

print(f"Dataset '{DATA_PATH}' registered successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("First 5 rows:\n", df.head())
