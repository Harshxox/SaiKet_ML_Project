# Import necessary enterprise libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# ==========================================
# STEP 1: LOAD THE OPTIMIZED TRAINING DATA
# ==========================================
try:
    # We load the VIP features and the target answers
    X_train = pd.read_csv("X_train_selected.csv")
    y_train = pd.read_csv("y_train.csv")
    print("Successfully loaded the optimized training data.")
except FileNotFoundError:
    print("Error: Could not find the training data. Make sure Task 3 was successful.")
    exit()

# Format target variable correctly for the machine learning model
y_train = y_train.values.ravel()

# ==========================================
# STEP 2: INITIALIZE THE WINNING MODEL
# ==========================================
print("\nInitializing the winning model: Logistic Regression...")
# We use the exact same settings that won the tournament in Task 4
final_model = LogisticRegression(max_iter=1000, random_state=42)

# ==========================================
# STEP 3: TRAIN THE MODEL
# ==========================================
print("Training the AI on the customer data...")
# The .fit() command is what actually makes the machine learning model learn
final_model.fit(X_train, y_train)
print("Training complete! The model has learned the churn patterns.")

# ==========================================
# STEP 4: SAVE THE TRAINED MODEL
# ==========================================
# We save the trained model as a .pkl (pickle) file. 
# This packages the AI so it can be used later without retraining.
model_filename = "trained_logistic_regression_model.pkl"
joblib.dump(final_model, model_filename)

print(f"\n✅ Success! The trained AI model has been permanently saved as '{model_filename}'")