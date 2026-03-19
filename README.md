# CS4630_LA10_Stevenson
# Student Dropout and Academic Success Prediction

## Overview

This project focuses on predicting student academic outcomes using machine learning. The goal is to classify students into one of three categories:

* **Dropout**
* **Enrolled**
* **Graduate**

Using demographic, academic, and socio-economic data, multiple classification models are trained and evaluated to determine the most effective approach.

---

## Dataset

* Source: UCI Machine Learning Repository
* Dataset: *Predict Students' Dropout and Academic Success*
* Total records: **4,424 students**
* Features: **37 variables**

### Key Feature Categories:

* Academic performance (grades, approvals, evaluations)
* Demographics (age, gender, nationality)
* Socio-economic factors (parent occupation, GDP, inflation)
* Financial status (tuition fees)

---

## Class Distribution

The dataset is imbalanced:

| Class    | Count | Percentage |
| -------- | ----- | ---------- |
| Graduate | 2209  | 49.9%      |
| Dropout  | 1421  | 32.1%      |
| Enrolled | 794   | 17.9%      |

### Handling Imbalance

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Balanced all classes to equal representation in training data

---

## Models Used

The following machine learning models were implemented:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)

---

## Model Performance

| Model         | Accuracy |
| ------------- | -------- |
| SVM           | **72%**  |
| Decision Tree | 69%      |
| Naive Bayes   | 64%      |
| KNN           | 61%      |

### Best Model: SVM

* Most balanced performance across all classes
* Strong precision and recall for **Graduate** and **Dropout**

---

## Key Findings

### 1. Academic Performance is the Strongest Predictor

Top features:

* Curricular units (2nd semester approved)
* Curricular units (2nd semester grade)
* Admission grade

Students with better academic performance are significantly more likely to graduate.

---

### 2. Financial Stability Matters

* Tuition fees up to date is a key factor

Students with financial difficulties are more likely to drop out.

---

### 3. Socio-economic Factors Influence Outcomes

* Parent occupation
* GDP

External environment plays a role in student success.

---

### 4. "Enrolled" is Hardest to Predict

* Lowest precision and recall across all models
* Overlaps with both Dropout and Graduate categories

---

## Evaluation Metrics

Models were evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1-score (important for imbalanced data)

---

## Hyperparameter Tuning

* GridSearchCV applied to Decision Tree
* Best parameters:

  * `max_depth = 10`
  * `min_samples_split = 5`

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib & Seaborn

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/repo-name.git
cd repo-name
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python main.py
```

---

## Future Improvements

* Try advanced models (Random Forest, XGBoost)
* Feature selection and dimensionality reduction
* Improve prediction for "Enrolled" class
* Cross-validation for more robust evaluation

---

## Author

* Rachel Stevenson
* CS4630 - Machine Learning Assignment

---

## Status

✔ Completed
✔ Models trained and evaluated
✔ Results analyzed

---
