# Breast Cancer Diagnosis Prediction using PCA and Multiple Machine Learning Models

## 📌 Overview
This project predicts whether a breast tumor is malignant or benign using supervised machine learning algorithms.

Dimensionality reduction was performed using Principal Component Analysis (PCA) to reduce 30 original features to 4 principal components while preserving maximum variance.

Multiple classification models were implemented and compared to identify the best-performing approach.

---

## 🎯 Problem Statement
Early and accurate classification of breast cancer improves patient outcomes. 

The objective of this project is to apply and compare multiple machine learning algorithms to classify tumors based on extracted numerical features.

---

## 📊 Dataset
- **Dataset:** Wisconsin Breast Cancer Dataset (WBCD)
- **Total Features:** 30 numerical predictors
- **Target Variable:**
  - Malignant (M)
  - Benign (B)

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Checked for missing values
- Standardized features using StandardScaler
- Converted labels to binary format
- Train-test split (80/20)

### 2️⃣ Dimensionality Reduction
- Applied PCA
- Reduced 30 features to 4 principal components
- Retained over 95% of total variance

### 3️⃣ Models Implemented
The following models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes (Gaussian NB)

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 94.8%    | 94%       | 95%    | 94%      |
| SVM                 | 95.6%    | 95%       | 96%    | 95%      |
| KNN                 | 93.9%    | 93%       | 94%    | 93%      |
| Decision Tree       | 92.7%    | 92%       | 93%    | 92%      |
| Random Forest       | 96.2%    | 96%       | 96%    | 96%      |
| **Naive Bayes**     | **97.4%**| **97%**   | **98%**| **97%**  |

### 🏆 Best Performing Model
Naive Bayes achieved the highest performance with an accuracy of **97.4%**, demonstrating strong classification capability even after dimensionality reduction using PCA.

---

## 🛠️ Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📂 Project Structure


