# Heart Disease Prediction - Group Assignment

**Module Title:** Machine Learning & Pattern Recognition  
**Assignment Title:** Design and Implementation of Machine Learning Models  
**Course/Stage:** Award (e.g., MSc in AI)  

**Group Members:**  
- *Sameer Jahangir (20054516) (Group B)*  
- *Naeem ul Hassan (20054701) (Group B)*  
- *Muhammad Bin Ashraf (20054513) (Group A)*  
- *( ) (Group A)*  

This repository contains a simple Flask-based machine learning app that predicts heart disease likelihood using a classification model trained on the [Kaggle Heart Disease dataset](https://www.kaggle.com/datasets/pritsheta/heart-attack?resource=download).

Our project follows the **CRISP-DM** methodology:
1. **Business Understanding**
2. **Data Understanding**
3. **Data Preparation**
4. **Modeling**
5. **Evaluation**
6. **Deployment**

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Assignment Requirements](#assignment-requirements)  
4. [Setup & Usage](#setup--usage)  
   1. [Create & Activate Virtual Environment](#create--activate-virtual-environment)  
   2. [Install Dependencies](#install-dependencies)  
   3. [Train the Model](#train-the-model)  
   4. [Run the Flask App](#run-the-flask-app)  
5. [CRISP-DM Implementation](#crisp-dm-implementation)  
6. [Group Contributions](#group-contributions)  
7. [License](#license)

---

## Project Overview

This project aims to **predict heart disease** (binary classification) based on medical attributes (e.g., age, blood pressure, cholesterol). We leverage various machine learning techniques—Logistic Regression, Support Vector Machine (SVM), and Naive Bayes—and compare their performance to identify the best classifier.

---

## Repository Structure

```bash
heart_disease_flask_app/
├── app.py                  ✅ entry point for Render
├── api/
│   └── index.py            ✅ real Flask app
├── templates/
│   ├── index.html
│   └── train.html
├── models/
│   └── best_model.pkl
├── data/
│   └── heart.csv
├── model_training.py
├── requirements.txt

```

# Assignment Requirements

Below is a concise summary of the assignment requirements in **Markdown** format, focusing on **CRISP-DM methodology**, **machine learning model development**, and **deployment**:

---

## 1. CRISP-DM Methodology

- **Business Understanding**: Clearly outline the problem context and objectives.  
- **Data Understanding**: Describe the dataset, variables, and initial observations.  
- **Data Preparation**: Explain the cleaning, transformation, and any feature engineering steps.  
- **Modeling**: Apply at least three (3) machine learning models and discuss the underlying mathematical foundations.  
- **Evaluation**: Compare each model’s performance using suitable metrics (accuracy, recall, precision, F1-score, etc.).  
- **Deployment**: Choose the best model and demonstrate how you would deploy it (e.g., a Flask web application).

---

## 2. Business Understanding

- Clarify the **business objectives** of your project (e.g., predict heart disease risk).
- Justify why the predictive solution is necessary or valuable.

---

## 3. Data Preparation

- Describe **any missing data handling** (e.g., dropping, imputing).
- Detail **categorical encoding** (One-Hot, Ordinal, etc.) if applicable.
- Include **scaling/normalization** approaches (StandardScaler, MinMaxScaler, etc.).
- Document **feature selection or engineering** steps, if any.

---

## 4. Modeling

1. **Model Selection**  
   - Present **three** different machine learning models (e.g., Logistic Regression, SVM, Naive Bayes).  
   - Include **mathematical explanations** or the key concepts behind each algorithm.

2. **Implementation**  
   - Use appropriate **Python libraries** (scikit-learn, pandas, etc.).  
   - Show your **training** process and **hyperparameter choices**.

---

## 5. Evaluation & Validation

- Utilize **appropriate metrics** for classification (accuracy, recall, precision, F1-score, ROC AUC, etc.).  
- Explain **why these metrics** are relevant to your heart disease prediction goal.  
- If necessary, detail **cross-validation** or hold-out test set methods to measure performance stability.  
- Provide a brief **comparison** of the model results to pick the best performer.

---

## 6. Deployment

- **Deploy** the top-performing model (e.g., in a **Flask** web app or another environment).
- Elaborate on how end-users or stakeholders can **interact** with your model (e.g., web form inputs, API endpoints).
- Summarize **key insights or findings** from your data, models, and their practical implications.

---

## 7. Group Report & Contributions

- Attach group meeting notes as an appendix.
- Include a statement of **each member’s contributions** (roles, tasks, estimated hours).
- Summarize any **challenges** and **lessons learned**.

---



# Setup & Usage

This section provides step-by-step instructions for setting up a local development environment, training the machine learning model, and running the Flask application.

---

## 1. Clone the Repository

```bash
git clone https://github.com/naeemhassan09/heart_disease_flask_app.git
cd heart_disease_flask_app
```


## 2. Create & Activate a Virtual Environment
	1.	Create a virtual environment (named venv here for convenience):

	python -m venv venv

	Activate the virtual environment:
	•	Windows: 
	venv\Scripts\activate

		•	macOS/Linux:
		source venv/bin/activate

⸻

## 3. Install Dependencies

Use the requirements.txt file to install all necessary packages:
pip install -r requirements.txt

## 4. Data Preparation

Place your Heart Disease dataset (e.g., heart.csv) in the data/ folder:
```bash
heart_disease_flask_app/
├── data/
│   └── heart.csv
```
___

## 5. Train the Model

Run the model training script to:
	•	Load the dataset from data/heart.csv.
	•	Clean and preprocess the data.
	•	Train multiple models.
	•	Save the best model to models/best_model.pkl.
```bash
	python model_training.py
```

⸻

## 6. Run the Flask App

Start the Flask application:
```bash
python app.py
```

By default, Flask hosts on http://127.0.0.1:5000/.

Open that URL in your browser to access a simple web form. Enter the required features (e.g., age, blood pressure, cholesterol, etc.) and click Predict. The page will display the model’s prediction.