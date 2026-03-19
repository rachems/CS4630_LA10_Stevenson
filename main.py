# main.py
# Student Dropout Prediction - Multi-class Classification

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# 2. LOAD DATA
# Make sure your CSV path is correct
df = pd.read_csv("data/data.csv", sep=";")

df.columns = df.columns.str.strip().str.replace('"', '')

print("First 5 rows:")
print(df.head())

# 3. EXPLORE DATA
print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['Target'].value_counts())

print("\nClass Distribution (Normalized):")
print(df['Target'].value_counts(normalize=True))

# 4. PREPROCESSING
X = df.drop('Target', axis=1)
y = df['Target']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. HANDLE IMBALANCE (SMOTE)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())

# 6. FEATURE SCALING
scaler = StandardScaler()

X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 7. FEATURE ANALYSIS 
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 8. DEFINE EVALUATION FUNCTION
def evaluate_model(name, model, X_test, y_test):
    print(f"\n===== {name} =====")
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# 9. TRAIN MODELS

#Decision Tree
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train_res, y_train_res)

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_res, y_train_res)

#Naive Bayes
nb = GaussianNB()
nb.fit(X_train_res, y_train_res)

#SVM
svm = SVC(class_weight='balanced', random_state=42)
svm.fit(X_train_res, y_train_res)

# 10. EVALUATE MODELS
evaluate_model("Decision Tree", dt, X_test, y_test)
evaluate_model("KNN", knn, X_test, y_test)
evaluate_model("Naive Bayes", nb, X_test, y_test)
evaluate_model("SVM", svm, X_test, y_test)

# 11. HYPERPARAMETER TUNING (Decision Tree Example)
print("\nRunning GridSearch for Decision Tree...")

params = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    scoring='f1_macro',
    cv=5
)

grid.fit(X_train_res, y_train_res)

print("\nBest Parameters:", grid.best_params_)

best_dt = grid.best_estimator_

# Evaluate tuned model
evaluate_model("Tuned Decision Tree", best_dt, X_test, y_test)

# 12. FEATURE IMPORTANCE (OPTIONAL)
importances = pd.Series(best_dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importances.head(10))

# Plot feature importance
plt.figure(figsize=(8, 6))
importances.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 13. DONE
print("\nProject Completed Successfully!")