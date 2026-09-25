# PulseCheck

Run the local app from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

The supplied CSV contains 70,000 rows of 10 input features but no `Heart_Risk` target column. The app therefore trains a logistic-regression demo model using a transparent proxy label based on weighted indicators. Training uses a reproducible stratified 80/20 split and exposes accuracy, precision, recall, F1, and row counts at `/api/health`; these metrics describe the proxy labels only. Replace the fallback in `train_model()` with real labeled outcomes before using this for clinical or production decisions
# Heart Disease Risk Prediction using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based system designed to predict the risk of heart disease using patient-related health and medical attributes available in the dataset.

The project uses Python and popular Machine Learning libraries to preprocess the data, train classification models, evaluate their performance, and make predictions.

> **Note:** This project is developed for educational and academic purposes. It is not intended to provide medical diagnosis or replace professional medical advice.

---

## 📂 Dataset

**Dataset Name:**

`heart_disease_risk_dataset_earlymed.csv`

**Dataset Location:**

```text
C:\Users\klegd\Downloads\heart_disease_risk_dataset_earlymed.csv
```

The dataset contains various health-related attributes that can be used as input features for predicting the target heart-disease-risk class.

Before training the model, the dataset is inspected for:

* Missing values
* Duplicate records
* Numerical features
* Categorical features
* Target/output column
* Data types

---

## 🎯 Objectives

The main objectives of this project are:

1. To understand and preprocess a real-world healthcare dataset.
2. To identify important features related to heart disease risk.
3. To apply Machine Learning classification algorithms.
4. To train models using the available dataset.
5. To predict heart disease risk based on input features.
6. To evaluate the performance of different ML models.
7. To compare model performance using standard evaluation metrics.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**
* **Joblib**

---

## 🤖 Machine Learning Algorithms

The project can implement the following classification algorithms:

### 1. Logistic Regression

Used as a basic classification model for predicting the target class.

### 2. Decision Tree Classifier

Uses decision rules based on the input features to classify the data.

### 3. Random Forest Classifier

Combines multiple decision trees to produce a classification result and can also provide feature-importance information.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Categorical Data Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Prediction
```

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

* Loading the CSV file using Pandas
* Checking dataset dimensions
* Checking data types
* Checking missing values
* Removing or handling duplicate records
* Handling categorical variables
* Encoding categorical values into numerical values
* Separating input features and target variable
* Splitting the dataset into training and testing data
* Applying feature scaling when required

---

## 📊 Model Evaluation

The trained models are evaluated using the following metrics:

### Accuracy

Measures the percentage of predictions that are correct.

### Precision

Measures how many of the samples predicted as a particular class actually belong to that class.

### Recall

Measures how many samples belonging to a particular class were correctly identified.

### F1 Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

### Classification Report

Provides precision, recall, F1-score, and support for each target class.

---

## 📈 Visualizations

The project can include the following visualizations:

* Dataset distribution
* Correlation heatmap
* Confusion matrix
* Model accuracy comparison
* Feature importance
* Target-class distribution

These visualizations help understand the dataset and model performance.

---

## 💻 Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## ▶️ How to Run the Project

### Step 1: Install Python

Install Python on your computer.

### Step 2: Install Required Libraries

Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 3: Open Jupyter Notebook

Run:

```bash
jupyter notebook
```

### Step 4: Load the Dataset

Use the dataset path:

```python
file_path = r"C:\Users\klegd\Downloads\heart_disease_risk_dataset_earlymed.csv"
```

### Step 5: Run the Notebook

Execute the cells in order:

1. Import libraries
2. Load dataset
3. Explore data
4. Preprocess data
5. Split data
6. Train models
7. Make predictions
8. Evaluate models
9. Compare results
10. Save the selected model

---

## 💾 Saving the Model

The trained model can be saved using Joblib:

```python
import joblib

joblib.dump(model, "heart_disease_model.pkl")
```

The saved model can later be loaded using:

```python
model = joblib.load("heart_disease_model.pkl")
```

---

## 🔮 Sample Prediction

After training the model, new patient-related feature values can be supplied to the model to generate a prediction.

```python
prediction = model.predict(new_data)

print("Predicted Class:", prediction)
```

The exact inp
