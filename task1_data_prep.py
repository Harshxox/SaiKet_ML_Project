# Import necessary enterprise libraries
import pandas as pd
import numpy as np

# ==========================================
# STEP 1: LOAD DATA
# ==========================================
file_name = "Telco_Customer_Churn_Dataset.csv"
df = pd.read_csv(file_name)

print("Original Dataset Shape:", df.shape)

# ==========================================
# STEP 2: HANDLE MISSING & INCONSISTENT DATA
# ==========================================
# 'TotalCharges' has blank spaces (" ") for some zero-tenure customers.
# We convert spaces to NaN (Not a Number) and change the column to numeric.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))

# Check how many missing values we have
missing_values = df.isnull().sum().sum()
print(f"Found {missing_values} missing values in TotalCharges. Dropping them...")

# Since it's only 11 rows out of 7043 (less than 0.2%), dropping them is the safest practice.
df.dropna(inplace=True)

# Drop 'customerID' as it is a unique identifier with no predictive value
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# ==========================================
# STEP 3: CATEGORICAL VARIABLE ENCODING
# ==========================================
# 3a. Binary Encoding for Yes/No columns
binary_mapping = {'Yes': 1, 'No': 0}
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_columns:
    df[col] = df[col].map(binary_mapping)

# Map gender separately
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# 3b. One-Hot Encoding for remaining multi-category columns
# We use drop_first=True to avoid the dummy variable trap (multicollinearity)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Convert any newly created boolean OHE columns to integers (1/0)
bool_cols = df_encoded.select_dtypes(include='bool').columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# ==========================================
# STEP 4: VERIFY AND SAVE
# ==========================================
print("\nFinal Preprocessed Dataset Shape:", df_encoded.shape)
print("\nFirst 5 rows of the ready dataset:")
print(df_encoded.head())

# Save this cleaned dataset so we can use it in Task 2
df_encoded.to_csv("Task1_Cleaned_Dataset.csv", index=False)
print("\nSuccess! Saved preprocessed data as 'Task1_Cleaned_Dataset.csv'")