import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# --- Data Cleaning and Preprocessing (as performed in the notebook) ---

# Drop 'Unnamed: 0' and 'CustomerID'
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

# Fix categorical inconsistencies:
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})
df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Handle missing/invalid data (as identified in the notebook)
invalid_pitch = df['DurationOfPitch'] > 60
invalid_income = (df['MonthlyIncome'] < 5000) | (df['MonthlyIncome'] > 40000)
invalid_trips = df['NumberOfTrips'] > 10
invalid_mask = invalid_pitch | invalid_income | invalid_trips
df = df[~invalid_mask].reset_index(drop=True)

# Drop remaining duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Define X and y
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

# Split data into training and testing sets
# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create directory for train/test splits if it doesn't exist
os.makedirs("tourism_project/model_building/splits", exist_ok=True)

# Save splits locally
Xtrain.to_csv("tourism_project/model_building/splits/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/model_building/splits/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/model_building/splits/ytrain.csv", index=False)
ytest.to_csv("tourism_project/model_building/splits/ytest.csv", index=False)

print("Data prepared: train/test splits written to tourism_project/model_building/splits/.")
print(f"Shape of Xtrain: {Xtrain.shape}, ytrain: {ytrain.shape}")
print(f"Shape of Xtest: {Xtest.shape}, ytest: {ytest.shape}")
