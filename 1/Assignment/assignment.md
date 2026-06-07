# 📘 Assignment: Logistic Regression

## 📌 Objective

The goal of this assignment is to **understand and apply the complete Machine Learning (ML) pipeline** using **Logistic Regression** for a **binary classification problem**.

By the end of this assignment, you should be able to:
- Apply the ML pipeline to a classification task
- Train and evaluate a logistic regression model
- Interpret coefficients and intercepts in a classification setting
- Compare single-feature vs multi-feature logistic regression models
---

## 🧠 The Machine Learning Pipeline

For **both tasks**, you must explicitly follow and document **all** of the following steps:

1. **Data Retrieval and Collection**
2. **Data Cleaning**
3. **Feature Design**
4. **Algorithm Selection**
5. **Loss Function Selection**
6. **Model Learning (Training)**
7. **Model Evaluation**

⚠️ *Each step must be clearly explained. Skipping steps will result in loss of marks.*

---

## 📊 Dataset

You will use a **Heart Disease dataset** containing medical attributes related to cardiovascular health. [Dataset Details](https://www.kaggle.com/datasets/neurocipher/heartdisease)

### Target Variable (Label)
- `HeartDisease`
  - Binary outcome:
    - `1` → Presence of heart disease
    - `0` → No heart disease

---

## 🧪 Task 1: Logistic Regression with a Single Feature

### 🎯 Problem Statement

Build a **logistic regression model** using **only one input feature** to predict whether a patient has heart disease.

- **Input Feature:** `Cholesterol`
- **Target Variable:** `HeartDisease`

---

### 🔍 Step-by-Step Requirements

#### 0️⃣ Background
- AI, ML, DL, Data Science

#### 1️⃣ Data Retrieval and Collection
- Load the Heart Disease dataset
- Display the dataset shape and column names
- Briefly describe the dataset

#### 2️⃣ Data Cleaning
- Check for missing values
- Handle missing or invalid cholesterol values
- Ensure the target variable is binary
- Verify data types

#### 3️⃣ Feature Design
- Select `Cholesterol` as the only input feature
- Separate features (`X`) and target (`y`)
- Explain why cholesterol may be a relevant predictor

#### 4️⃣ Algorithm Selection
- Choose **Logistic Regression**
- Explain why logistic regression is suitable for binary classification

#### 5️⃣ Loss Function Selection
- Use **Binary Cross-Entropy (Log Loss)**
- Briefly explain how this loss function works

#### 6️⃣ Model Learning (Training)
- Split the dataset into training and testing sets
- Train the logistic regression model
- Explain how model parameters are learned

#### 7️⃣ Model Evaluation
Evaluate the model using appropriate **classification metrics**, such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Provide a short interpretation of the results.
- Plot the **sigmoid curve** for Task 1

---

## 🧪 Task 2: Logistic Regression with Multiple Features

### 🎯 Problem Statement

Build a **logistic regression model** using **all available input features** to predict heart disease.

- **Input Features:** All variables except `HeartDisease`
- **Target Variable:** `HeartDisease`

---

### 🔍 Step-by-Step Requirements

Repeat **all seven ML pipeline steps** from Task 1 with the following changes:

#### 3️⃣ Feature Design
- Use all available features
- Encode categorical variables (if any)
- Apply feature scaling where appropriate
- Explain why using multiple features may improve prediction performance


---

## 📊 Model Comparison

Compare **Task 1 and Task 2** models:

- Which model performs better and why?
- How does adding more features affect accuracy and recall?
- Trade-offs between interpretability and performance

---

## 📦 Deliverables

Submit:

- A **Jupyter Notebook (.ipynb)**:
  - Code
  - Clear markdown explanations for each ML pipeline step
- Well-labeled outputs and visualizations (if used)

---

## 📌 Reminder

> **Logistic regression predicts probabilities, not classes directly.**

Good luck! 🚀