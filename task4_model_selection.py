# Import necessary enterprise libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# ==========================================
# STEP 1: LOAD THE OPTIMIZED TRAINING DATA
# ==========================================
# We load the VIP features we successfully selected in Task 3
try:
    X_train = pd.read_csv("X_train_selected.csv")
    y_train = pd.read_csv("y_train.csv")
    print("Successfully loaded optimized training data.")
except FileNotFoundError:
    print("Error: Could not find 'X_train_selected.csv'. Make sure Task 3 was successful.")
    exit()

# Format target variable correctly for the models (1D array)
y_train = y_train.values.ravel()

# ==========================================
# STEP 2: DEFINE THE CANDIDATE MODELS
# ==========================================
# We define the 4 algorithms required by the SaiKet Systems prompt
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# ==========================================
# STEP 3: CROSS-VALIDATION TOURNAMENT
# ==========================================
print("\n--- MODEL SELECTION TOURNAMENT ---")
print("Evaluating models using 5-Fold Cross-Validation (this may take a few seconds)...\n")

best_score = 0
best_model_name = ""

for name, model in models.items():
    # cv=5 means the algorithm tests itself 5 times to ensure the score is reliable
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    avg_accuracy = np.mean(scores)
    
    print(f"Algorithm: {name}")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print("-" * 35)
    
    # Keep track of the highest scoring model
    if avg_accuracy > best_score:
        best_score = avg_accuracy
        best_model_name = name

# ==========================================
# STEP 4: DECLARE THE WINNER
# ==========================================
print(f"\n🏆 TOURNAMENT WINNER: {best_model_name}!")
print(f"With the highest accuracy of {best_score * 100:.2f}%, we will select {best_model_name} as our final model.")