<div align="center">

# 🛌 Sleep Disorder Classification using Machine Learning

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-green?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

*A machine learning approach to predict sleep disorders using health and lifestyle data*

[View Notebook](https://colab.research.google.com/assets/colab-badge.svg)

---

</div>

## 📋 Overview

Sleep is a vital component of human health, yet sleep disorders such as insomnia and sleep apnea have become increasingly prevalent in modern society. This project implements a **Decision Tree Classifier** to predict the presence of sleep disorders, enabling early detection that can significantly improve quality of life and prevent long-term medical complications.

### 🎓 Academic Context

- **Institution:** University of Asia Pacific (UAP)
- **Department:** Computer Science and Engineering (CSE)
- **Course:** CSE 404 - Artificial Intelligence and Expert Systems Lab

### 👤 Project Owner

| Name | ID |
|------|-----|
| Bokhtear Md. Abid | - |

---

## 🎯 Problem Statement

In our fast-paced world, stress, lifestyle choices, and health conditions contribute to rising sleep disorder rates. This project addresses the need for **predictive healthcare** by leveraging machine learning to identify individuals at risk of sleep disorders based on their health metrics and lifestyle patterns.

---

## 🛠️ Technologies & Tools

| Technology | Purpose |
|-----------|---------|
| **Python 3.x** | Core programming language |
| **Jupyter/Kaggle Notebook** | Development environment |
| **pandas & numpy** | Data manipulation & numerical computation |
| **matplotlib & seaborn** | Data visualization |
| **scikit-learn** | Machine learning framework |
| **kagglehub** | Dataset acquisition |

---

## 📊 Dataset

**Source:** [Sleep Health and Lifestyle Dataset](https://www.kaggle.com) (Kaggle)

- **Format:** CSV
- **Total Instances:** 374 rows
- **After Preprocessing:** 352 rows
- **Features:** 13

### 📝 Feature Description

| Feature | Description | Type |
|---------|-------------|------|
| **Gender** | Male / Female | Categorical |
| **Age** | Age of individual | Numerical |
| **Occupation** | Type of profession | Categorical |
| **Sleep Duration** | Average sleep hours per day | Numerical |
| **Quality of Sleep** | Subjective rating (1-10) | Numerical |
| **Physical Activity Level** | Exercise minutes per day | Numerical |
| **Stress Level** | Stress rating (1-10) | Numerical |
| **BMI Category** | Weight classification | Categorical |
| **Heart Rate** | Resting pulse rate (bpm) | Numerical |
| **Daily Steps** | Number of steps per day | Numerical |
| **Blood Pressure** | Systolic/Diastolic (split) | Numerical |
| **Sleep Disorder** | Target: None / Insomnia / Sleep Apnea | Categorical |

---

## 🔧 Data Preprocessing Pipeline

### 1️⃣ Complex Column Handling
- Split **Blood Pressure** column (e.g., "120/80") into:
  - `BP_Systolic`
  - `BP_Diastolic`

### 2️⃣ Missing Value Treatment
- Identified and removed rows with missing/invalid values
- Applied `dropna()` for data integrity

### 3️⃣ Categorical Encoding
Applied **Label Encoding** for:
- Gender
- Occupation
- BMI Category
- Sleep Disorder (target variable)

### 4️⃣ Final Dataset
- **Rows:** 352
- **Columns:** 13
- **Clean & Ready** for model training

---

## 📈 Exploratory Data Analysis

### Statistical Insights

| Metric | Value |
|--------|-------|
| Mean Sleep Duration | 7.05 hours |
| Median Sleep Duration | 7.0 hours |
| Stress Level Std Dev | 2.13 |

### 🔗 Correlation Insights

- ✅ **Positive correlation:** Stress Level → Sleep Disorder
- ✅ **Positive correlation:** Heart Rate → Sleep Disorder
- ✅ **Negative correlation:** Physical Activity → Sleep Disorder

---

## 🤖 Model Development

### Model Architecture

**Algorithm:** Decision Tree Classifier

**Rationale:**
- ✔️ Handles mixed data types (numerical & categorical)
- ✔️ High interpretability via feature importance
- ✔️ Captures non-linear relationships effectively

### Training Configuration

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization & training
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

- **Training Set:** 80%
- **Test Set:** 20%
- **Random State:** 42 (reproducibility)

---

## 📊 Model Performance

### Evaluation Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.87 (87%) |
| **Precision** | 0.85 |
| **Recall** | 0.84 |
| **F1-Score** | 0.84 |

### 🎯 Performance Highlights

- **87% accuracy** demonstrates strong predictive capability
- **High recall** ensures effective identification of at-risk individuals
- Balanced precision-recall trade-off for practical deployment

---

## 🔍 Feature Importance Analysis

<div align="center">

| Feature | Importance Score |
|---------|-----------------|
| 🔴 **Stress Level** | 0.24 |
| 🟠 **Sleep Duration** | 0.21 |
| 🟡 **Physical Activity** | 0.15 |
| 🟢 **Quality of Sleep** | 0.13 |
| 🔵 **BMI Category** | 0.10 |
| 🟣 **Age** | 0.07 |
| 🟤 **Heart Rate** | 0.05 |
| ⚪ **Daily Steps** | 0.03 |
| ⚫ **BP Systolic** | 0.02 |

</div>

### 💡 Key Takeaways

- **Stress Level** and **Sleep Duration** are the most influential predictors
- Findings align with medical literature on sleep disorders
- **Physical Activity** acts as a protective factor

---

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Distribution Plot** - Sleep duration patterns
2. **Correlation Heatmap** - Feature relationships
3. **Confusion Matrix** - Classification performance
4. **Feature Importance Chart** - Predictor rankings

---

## 💡 Results & Discussion

### ✅ Achievements

- Successfully classified sleep disorders with **87% accuracy**
- Identified **stress** and **sleep duration** as primary risk factors
- Demonstrated the protective effect of **physical activity**
- Model suitable for preventive healthcare screening

### ⚠️ Limitations

- Relatively small dataset (352 samples) may limit generalization
- Class imbalance with fewer "Sleep Apnea" cases
- Potential overfitting inherent to decision trees

### 🚀 Future Improvements

- **Ensemble Methods:** Implement Random Forest or XGBoost for robustness
- **Larger Dataset:** Acquire more diverse samples for better generalization
- **Hyperparameter Tuning:** GridSearchCV or RandomizedSearchCV optimization
- **Cross-Validation:** K-fold validation for performance stability
- **Feature Engineering:** Create interaction features for improved predictions

---

## 📁 Project Structure

```
sleep-disorder-classification/
│
├── data/
│   └── sleep_health_lifestyle.csv
│
├── notebooks/
│   └── sleep_disorder_analysis.ipynb
│
├── models/
│   └── decision_tree_model.pkl
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sleep-disorder-classification.git

# Navigate to directory
cd sleep-disorder-classification

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

---

## 📱 Notebook Access

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🎓 Conclusion

This project successfully demonstrates the application of machine learning in healthcare predictive analytics. By achieving **87% accuracy** in sleep disorder classification, the model provides valuable insights for:

- Early detection and intervention
- Preventive healthcare strategies
- Lifestyle modification recommendations
- Public health awareness initiatives

The identification of **stress level** and **sleep duration** as key predictors aligns with clinical research and validates the model's practical utility in real-world healthcare applications.

---

<div align="center">

### 🏛️ Academic Information

**University of Asia Pacific**  
Department of Computer Science and Engineering  
CSE 404 - Artificial Intelligence and Expert Systems Lab

---

**Made with ❤️ for better sleep health**

![Footer](https://img.shields.io/badge/Machine%20Learning-Healthcare-red?style=for-the-badge)

</div>
