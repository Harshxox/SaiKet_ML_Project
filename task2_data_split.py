# Import necessary enterprise libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# STEP 1: LOAD THE CLEANED DATA
# ==========================================
# We load the output from Task 1
try:
    df = pd.read_csv("Task1_Cleaned_Dataset.csv")
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Could not find 'Task1_Cleaned_Dataset.csv'. Make sure you are in the correct folder.")
    exit()

# ==========================================
# STEP 2: SEPARATE FEATURES (X) AND TARGET (y)
# ==========================================
# 'y' is our target variable (what we want to predict)
y = df['Churn']

# 'X' is our features (everything else, so we drop the Churn column)
X = df.drop('Churn', axis=1)

# ==========================================
# STEP 3: PERFORM THE 80/20 SPLIT
# ==========================================
# test_size=0.2 means 20% for testing, 80% for training.
# random_state=42 ensures reproducibility (we get the same split every time).
# stratify=y ensures proportional representation of Churn in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# STEP 4: VERIFY AND SAVE
# ==========================================
print("\n--- SPLIT RESULTS ---")
print(f"Total dataset size: {len(df)} rows")
print(f"Training Features (X_train): {X_train.shape[0]} rows, {X_train.shape[1]} columns")
print(f"Testing Features (X_test): {X_test.shape[0]} rows, {X_test.shape[1]} columns")
print(f"Training Target (y_train): {y_train.shape[0]} rows")
print(f"Testing Target (y_test): {y_test.shape[0]} rows")

# Save the split datasets so we can use them in Task 3 and beyond
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nSuccess! Saved 4 new files: X_train.csv, X_test.csv, y_train.csv, and y_test.csv.")