# Import necessary enterprise libraries
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

# ==========================================
# STEP 1: LOAD THE TEST DATA & THE MODEL
# ==========================================
try:
    # Load the 20% unseen test data and the VIP features
    X_test = pd.read_csv("X_test_selected.csv")
    y_test = pd.read_csv("y_test.csv")
    
    # Load the AI "brain" we saved in Task 5
    model = joblib.load("trained_logistic_regression_model.pkl")
    print("Successfully loaded the test data and the trained AI model.\n")
except FileNotFoundError:
    print("Error: Could not find the required files. Make sure Task 3 and Task 5 were successful.")
    exit()

# Format target variable correctly
y_test = y_test.values.ravel()

# ==========================================
# STEP 2: MAKE PREDICTIONS (THE EXAM)
# ==========================================
print("Asking the AI to predict churn for unseen customers...")
# Predict if the customer churns (1) or stays (0)
y_pred = model.predict(X_test)

# Predict the *probability* of churn (needed for the ROC-AUC score)
y_proba = model.predict_proba(X_test)[:, 1]

# ==========================================
# STEP 3: CALCULATE AND DISPLAY METRICS
# ==========================================
print("\n" + "="*50)
print("🏆 FINAL MODEL EVALUATION REPORT 🏆")
print("="*50)

# Calculate the specific metrics requested by SaiKet Systems
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy:  {accuracy * 100:.2f}%  (Overall correctness)")
print(f"Precision: {precision * 100:.2f}%  (When predicting Churn, how often is it right?)")
print(f"Recall:    {recall * 100:.2f}%  (Out of all actual Churners, how many did it catch?)")
print(f"F1-Score:  {f1 * 100:.2f}%  (Harmonic mean of Precision and Recall)")
print(f"ROC-AUC:   {roc_auc * 100:.2f}%  (Ability to distinguish Churn vs. No-Churn)")
print("="*50)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"]))

print("\n🎉 ALL TASKS COMPLETE! You are ready to record your LinkedIn video.")