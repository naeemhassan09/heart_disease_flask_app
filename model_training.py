"""
model_training.py

This script demonstrates the steps of CRISP-DM methodology:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment (Saving the best model)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score
import pickle
from sklearn.preprocessing import StandardScaler

def train_and_save_model(data_path="./data/heart.csv"):
    # To Verify the file path
    print("\nFile Path: ", data_path)
    # --------------------------
    # 1. Business Understanding
    # --------------------------
    # The goal of this project is to help predict whether a person is likely to suffer from 
    # heart disease based on a set of common medical indicators. Heart disease is a leading 
    # cause of death worldwide, and early detection can be crucial for timely treatment. 
    #
    # In many clinical settings, doctors rely on various test results and patient symptoms 
    # to make diagnostic decisions. By training a machine learning model on historical 
    # patient data, we aim to create a system that can assist healthcare professionals 
    # by providing a second opinion based on data-driven predictions.
    #
    # For this assignment, we're using a well-known Kaggle dataset that contains features 
    # such as age, sex, cholesterol level, resting blood pressure, and more. The model will 
    # learn patterns in these features to classify patients as either at risk (1) or not at risk (0)
    # of having heart disease.
    #
    # Ultimately, this task aligns with the Business Understanding phase of CRISP-DM, where 
    # we define the project objective from a real-world perspective—in this case, supporting 
    # better health outcomes through data science.



    # Let's Read the Data
    df = pd.read_csv(data_path)
    # Quick checks
    print(df.head())
    print(df.info())

     # --------------------------
    # 2. Data Understanding
    # --------------------------
    # After establishing the goal of predicting heart disease, the next step is to explore
    # and understand the structure and content of our dataset. For this project, we’re using
    # the Kaggle dataset titled “Heart Attack Analysis & Prediction Dataset” which includes
    # a variety of patient health attributes.
    #
    # Some of the most relevant columns include:
    # - age: Age of the patient
    # - sex: Gender (1 = male, 0 = female)
    # - cp: Chest pain type (categorical)
    # - trestbps: Resting blood pressure (mm Hg)
    # - chol: Serum cholesterol level (mg/dl)
    # - fbs: Fasting blood sugar (> 120 mg/dl)
    # - restecg: Resting electrocardiographic results (categorical)
    # - thalach: Maximum heart rate achieved
    # - exang: Exercise-induced angina (1 = yes, 0 = no)
    # - oldpeak: ST depression induced by exercise
    # - slope, ca, thal: Additional diagnostic measures
    # - target: Output variable (1 = heart disease, 0 = no heart disease)
    #
    # The `target` column is our label — the value we want to predict.
    #
    # Since some versions of the dataset might have a different label name (e.g., 'output'
    # instead of 'target'), we include a line to rename the column to a consistent name
    # (`heart_disease`) for use throughout the rest of the script.
    #
    # This phase of CRISP-DM helps us identify what data we’re working with, its types,
    # and what preprocessing may be needed to prepare it for machine learning.
    
    df.rename(columns={'target': 'heart_disease'}, inplace=True)  # example rename if needed

      # --------------------------
    # 3. Data Preparation
    # --------------------------
    # At this stage, we prepare the dataset for training by handling missing values,
    # encoding categorical variables, and scaling numerical features.

    # First, handle missing values if any
    df.dropna(inplace=True)

    # Separate features and target
    X = df.drop(columns=['heart_disease'])  # target is our output label
    y = df['heart_disease']

    # Identify categorical columns that need encoding
    # These are typically represented as integers but should be treated as categories
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # One-Hot Encode categorical features
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    # ColumnTransformer lets us apply transformers to specific columns only
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'  # Leave the other columns (numerical) as they are
    )

    # Apply the preprocessing (encoding)
    X_encoded = preprocessor.fit_transform(X)

    # Scale all features (encoded + numerical)
  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Now X_scaled is ready to be used for training
        # --------------------------
    # Train-Test Split and Transformation
    # --------------------------

    # Split the original data before applying preprocessing
    # This ensures the model doesn't "peek" at test data during transformation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the preprocessor on training data and transform both sets
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Apply scaling after encoding
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # At this point, X_train_scaled and X_test_scaled are fully preprocessed
    # and ready to be used for training and evaluating ML models.
  
 # --------------------------
    # 4. Modeling
    # --------------------------
    # In this phase, we train and compare different machine learning models to find
    # the best one for predicting heart disease based on the processed data.

    # We’ll use three widely used classification models:
    # - Logistic Regression: good for linear decision boundaries and interpretability
    # - Support Vector Machine (SVC): effective for non-linear separation
    # - Naive Bayes: fast and works well on smaller datasets with categorical features

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine (RBF Kernel)": SVC(kernel='rbf'),
        "Naive Bayes": GaussianNB()
    }

    best_model_name = None
    best_score = 0.0
    best_model = None

    print("\nModel Performance Comparison:\n")

    for model_name, model in models.items():
        # Train the model on the training set
        model.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test_scaled)

        # Evaluate model using accuracy and recall
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        print(f"{model_name} -> Accuracy: {acc:.4f}, Recall: {rec:.4f}")

        # Select the best model based on accuracy (could also prioritize recall if needed)
        if acc > best_score:
            best_score = acc
            best_model_name = model_name
            best_model = model

    print(f"\n✅ Best Model based on Accuracy: {best_model_name} ({best_score:.4f})")

    # --------------------------
    # 5. Evaluation
    # --------------------------
    # We already printed metrics above. Additional evaluation steps could be cross-validation:
    # cross_val = cross_val_score(best_model, X, y, cv=5)
    # print("CV Accuracy for best model:", cross_val.mean())

    # --------------------------
    # 6. Deployment
    # --------------------------
    # In the final phase of CRISP-DM, we prepare our trained model for deployment.
    # This means saving the trained model along with any preprocessing objects
    # so that they can be reused later during prediction (e.g., in our Flask web app).

    # We'll save:
    # - The best performing model
    # - The preprocessor (OneHotEncoder inside ColumnTransformer)
    # - The feature scaler (StandardScaler)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump((best_model, preprocessor, scaler), f)

    print("✅ Deployment complete! Model, encoder, and scaler saved to models/best_model.pkl")


# If running this script directly, we can execute the training
if __name__ == "__main__":
    train_and_save_model()