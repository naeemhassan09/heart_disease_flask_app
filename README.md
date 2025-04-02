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
├── data/
│   └── heart.csv               # Heart Disease dataset (Kaggle)
├── models/
│   └── best_model.pkl          # Trained (serialized) best model
├── templates/
│   └── index.html              # Minimal Flask HTML form
├── app.py                      # Flask application
├── model_training.py           # Script to preprocess, train & save best model
├── requirements.txt            # Python dependencies
└── README.md                   # This file