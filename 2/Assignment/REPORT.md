# Lab Report: Linear Regression for Salary Prediction

## 1. Title
**Linear Regression Analysis: Predicting Employee Salary based on Years of Experience**

---

## 2. Objectives
- To understand and apply the complete **Machine Learning (ML) pipeline** to a regression problem.
- To implement **Simple Linear Regression** (Single Feature) and **Polynomial Regression** (Multiple Features).
- To evaluate and compare model performance using **Mean Squared Error (MSE)** and **R² Score**.
- To interpret model parameters (coefficients and intercepts) and understand their physical meaning.

---

## 3. Background & Theory

### 🧠 Artificial Intelligence (AI)
Artificial Intelligence is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

### 🤖 Machine Learning (ML)
Machine Learning is a subset of AI that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. In this lab, we use **Supervised Learning**, where the model learns from labeled data.

### 🕸️ Deep Learning (DL)
Deep Learning is a specialized subset of ML based on artificial neural networks with multiple layers (hence "deep"). While linear regression is a simple statistical method, DL can model extremely complex non-linear relationships.

### 📊 Data Science (DS)
Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from noisy, structured, and unstructured data. The ML pipeline is a core component of Data Science.

### 📈 Linear Regression
Linear Regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features). 
- **Simple Linear Regression:** $y = \beta_0 + \beta_1x + \epsilon$
- **Polynomial Regression:** $y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \epsilon$

The goal is to find the "best-fit" line (or curve) that minimizes the difference between predicted and actual values.

---

## 🧪 Task 1: Simple Linear Regression (Single Feature)

### 1️⃣ Data Retrieval and Collection
- **Action:** Loaded the `Salary_Data.csv` dataset using pandas.
- **Observations:** The dataset contains 30 records and 2 columns: `YearsExperience` and `Salary`.
- **Shape:** (30, 2)

### 2️⃣ Data Cleaning
- **Action:** Checked for missing values using `df.isnull().sum()`. 
- **Treatment:** No missing values were found. Data types were verified as `float64` for experience and `int64` for salary, which are appropriate for regression.

### 3️⃣ Feature Design
- **Action:** Separated the independent feature ($X$) and the dependent label ($y$).
- **Features ($X$):** `YearsExperience`
- **Label ($y$):** `Salary`
- **Reasoning:** Years of experience is a logical predictor for salary in a professional context.

### 4️⃣ Algorithm Selection
- **Algorithm:** Simple Linear Regression.
- **Reason:** Linear regression is appropriate here because we are predicting a continuous numerical value (Salary) and we expect a linear trend where more experience leads to higher pay.

### 5️⃣ Loss Function Selection
- **Loss Function:** Mean Squared Error (MSE).
- **Definition:** $MSE = \frac{1}{n} \sum (y_{actual} - y_{predicted})^2$. It measures the average squared difference between estimated values and the actual value, penalizing larger errors more heavily.

### 6️⃣ Model Learning (Training)
- **Process:** The data was split into Training (75%) and Testing (25%) sets. The `LinearRegression()` model from scikit-learn was fitted to the training data.
- **Learning:** The model uses the **Ordinary Least Squares (OLS)** method to find the coefficients that minimize the sum of squared residuals.

### 7️⃣ Model Evaluation
- **Mean Squared Error (MSE):** ~38,802,588.99
- **R² Score:** ~0.9347
- **Interpretation:** An R² score of 0.93 means that 93.5% of the variance in salary is explained by years of experience. This indicates a very strong linear relationship.

---

## 🧪 Task 2: Polynomial Regression (Degree 2)

### 3️⃣ Feature Design (Updated)
- **Action:** Applied `PolynomialFeatures(degree=2)` to `YearsExperience`.
- **New Features:** $[x, x^2]$ (Experience and Experience squared).
- **Reasoning:** To check if the salary growth accelerates or decelerates over time (non-linear relationship).

### 6️⃣ Model Learning (Training)
- **Process:** Trained a Linear Regression model on the transformed polynomial features.

### 7️⃣ Model Evaluation
- **Mean Squared Error (MSE):** ~40,643,285.39
- **R² Score:** ~0.9316
- **Interpretation:** The performance is slightly lower than the simple linear model on the test set, suggesting that a simple linear fit is sufficient and more complex features might lead to slight overfitting on this small dataset.

---

## 📈 Model Interpretation (Mandatory)

**For the Simple Linear Regression Model:**
- **Coefficient ($\beta_1$):** 9371.02
- **Intercept ($\beta_0$):** 25478.13

### **Explanations:**
1. **What does the coefficient represent?**
   The coefficient of **9371.02** means that for every additional year of experience, the predicted salary increases by approximately **$9,371.02**. It represents the "slope" or the rate of change.

2. **What does the intercept mean in this context?**
   The intercept of **25478.13** represents the predicted salary for an individual with **zero years of experience**. It is the baseline salary (starting pay) according to the model.

---

## 📊 Discussion & Conclusion

### **Discussion on Assumptions**
Linear regression assumes:
1. **Linearity:** The relationship between $X$ and $y$ is linear. (Validated by high R²).
2. **Independence:** Observations are independent.
3. **Homoscedasticity:** Constant variance of errors.
4. **Normality:** Residuals are normally distributed.

### **Conclusion**
- The Simple Linear Regression model performed exceptionally well with an R² of **0.9347**.
- Adding polynomial features (Task 2) did **not** improve the model significantly for this dataset, indicating that the relationship is naturally linear.
- **Recommendation:** For this specific dataset, the Simple Linear Regression model is preferred due to its simplicity and high interpretability.

---

## 📦 Deliverables
- **Notebook:** `Lab2-LinearRegression/Assignment/assignment_solution_fixed.ipynb`
- **Dataset:** `Salary_Data.csv`
