# Customer Churn Analysis and Prediction 

**Author:** HARSHDEEP SHARMA  
**Organization:** SaiKet Systems (MSME Internship Program)  

## 📌 Project Overview
This project focuses on analyzing and predicting customer churn within a telecommunications dataset. By identifying at-risk customers through predictive modeling, this solution provides actionable intelligence to reduce churn rates and improve overall customer retention strategies. The project encompasses a full end-to-end Machine Learning pipeline, from raw data preprocessing to model deployment packaging.

## ⚙️ Key Technical Elements & Pipeline
This project was executed in a structured, six-stage pipeline to ensure robust data handling and model accuracy:

1. **Data Preparation (`task1_data_prep.py`)**
   - Cleaned the dataset by handling hidden missing values (`NaN`) in the `TotalCharges` column.
   - Performed categorical variable encoding, utilizing Binary Encoding for two-choice features and One-Hot Encoding (OHE) for multi-category features to ensure machine learning readiness.

2. **Train/Test Data Splitting (`task2_data_split.py`)**
   - Segregated the dataset into an **80% Training Set** and a **20% Testing Set**.
   - Applied stratified splitting to maintain the exact proportion of churned vs. retained customers across both subsets, preventing class imbalance skewing.

3. **Feature Selection (`task3_feature_selection.py`)**
   - Implemented a Random Forest Classifier to calculate individual feature importance scores.
   - Filtered out statistical noise by dropping low-impact columns, reducing the dataset to the **Top 16 highly predictive features** (driven heavily by `TotalCharges`, `MonthlyCharges`, and `tenure`).

4. **Model Selection Tournament (`task4_model_selection.py`)**
   - Conducted a comparative analysis of four binary classification algorithms: *Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting*.
   - Utilized 5-Fold Cross-Validation on the training data. **Logistic Regression** emerged as the optimal model with an 80.02% cross-validation accuracy.

5. **Model Training & Packaging (`task5_model_training.py`)**
   - Trained the optimized Logistic Regression algorithm on the filtered training dataset.
   - Serialized and exported the finalized model as a `.pkl` (pickle) file for future deployment and inference.

6. **Final Model Evaluation (`task6_model_evaluation.py`)**
   - Executed the "final exam" by testing the saved model against the 20% unseen testing data.
   - Generated a comprehensive classification report utilizing enterprise-standard metrics.

## 📊 Final Evaluation Metrics
The deployed Logistic Regression model achieved the following performance metrics on unseen data:
- **Accuracy:** 80.17%
- **ROC-AUC Score:** 83.84% (Excellent capability in distinguishing between churn and non-churn profiles)
- **Precision (Churn):** 65.99%
- **Recall (Churn):** 51.87%

## 💻 Tech Stack
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)
- **Model Serialization:** Joblib

## 🚀 How to Run Locally
1. Clone the repository and ensure your virtual environment is active.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
3.Place the Telco_Customer_Churn_Dataset.csv in the root directory.

4. Execute each task script sequentially to follow the pipeline:
```bash
python task1_data_prep.py
python task2_data_split.py
python task3_feature_selection.py
python task4_model_selection.py
python task5_model_training.py
python task6_model_evaluation.py
```