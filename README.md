# 🌸 Iris Multi-Class Classification (scikit-learn Pipeline)

> End-to-end machine learning project demonstrating a **multi-class classifier** with a **reproducible pipeline**, **comprehensive evaluation suite**, and **model persistence**.

---

## 🔍 Overview

This project builds a **Logistic Regression** model within a **scikit-learn Pipeline** to classify Iris flowers into **Setosa, Versicolor, and Virginica**. It emphasizes clean workflow design, robust evaluation (including **ROC-AUC for multi-class**), and saving the trained model for reuse.

---

## 🎯 Objectives

* Implement a **multi-class classification** model
* Use a **Pipeline** for preprocessing + modeling
* Evaluate with **accuracy, confusion matrix, precision, recall, F1, ROC-AUC**
* **Persist** the trained model (`.pkl`) for deployment

---

## 📊 Dataset

* **Source:** `sklearn.datasets.load_iris()`
* **Samples:** 150 | **Features:** 4 (sepal/petal length & width) | **Classes:** 3
* Clean dataset (no missing values) → ideal for demonstrating workflow

---

## ⚙️ Methodology

1. **Load Data** → `load_iris()` → Pandas DataFrame
2. **Split** → 80% train / 20% test
3. **Pipeline**

   * `StandardScaler` (feature scaling)
   * `LogisticRegression(max_iter=200)`
4. **Train** → `pipeline.fit(X_train, y_train)`
5. **Predict** → `pipeline.predict(X_test)`
6. **Evaluate** → Accuracy, Confusion Matrix, Precision/Recall/F1, ROC-AUC (OvR)
7. **Persist** → Save pipeline as `iris_model.pkl`

---

## 📈 Results (Typical)

| Metric    | Value       |
| --------- | ----------- |
| Accuracy  | 0.95 – 1.00 |
| Precision | High        |
| Recall    | High        |
| F1 Score  | High        |

> Note: Exact values may vary slightly due to train/test split.

---

## 🧪 Evaluation Suite

* **Confusion Matrix** (class-wise performance)
* **Classification Report** (Precision, Recall, F1)
* **ROC-AUC (One-vs-Rest)** for multi-class discrimination
* **Metric Table** for quick summary

---

## 💾 Model Persistence

Trained pipeline saved as:

```bash
iris_model.pkl
```

* Includes **scaler + model** → ready for inference without retraining

---

## 🗂️ Project Structure

```bash
ml-day2-iris-classifier/
│── notebook.ipynb        # End-to-end implementation
│── iris_model.pkl        # Trained pipeline (scaler + model)
│── README.md             # Project documentation
```

---

## 🚀 Quick Start

### 1) Clone

```bash
git clone https://github.com/<your-username>/ml-day2-iris-classifier.git
cd ml-day2-iris-classifier
```

### 2) Install

```bash
pip install -r requirements.txt
```

> If you don’t have a requirements file, install:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3) Run

```bash
jupyter notebook
```

Open `notebook.ipynb` and run all cells.

---

## 🧠 Key Learnings

* Designing **reproducible ML pipelines**
* Handling **multi-class evaluation** (OvR ROC-AUC)
* Interpreting **confusion matrix & classification metrics**
* **Saving/loading** models for real-world usage

---

## 🔮 Future Improvements

* Compare with **Decision Trees / Random Forest / SVM**
* **Hyperparameter tuning** (GridSearchCV)
* Add **cross-validation**
* Build a simple **API (FastAPI/Flask)** for inference

---

## 👨‍💻 Author

**Hemanth Gorijala**
AI/ML Trainee

---

## 📌 One-Line Summary

A clean, production-style ML workflow for **multi-class classification** with strong evaluation and reusable model artifacts.
