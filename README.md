Sleep Disorder Classification using Machine Learning
https://img.shields.io/badge/Machine-Learning-blue
https://img.shields.io/badge/Python-3.x-green
https://img.shields.io/badge/Scikit--Learn-Latest-orange
https://img.shields.io/badge/Status-Completed-success

📋 Project Overview
University: University of Asia Pacific (UAP)
Department: Computer Science and Engineering (CSE)

👥Owner
Name	
Bokhtear Md. Abid	

🎯 Problem Statement
Sleep is a vital component of human health. In modern society, sleep disorders such as insomnia and sleep apnea have become increasingly common due to stress, lifestyle choices, and health conditions. Early detection of such disorders can significantly improve the quality of life and prevent long-term medical complications.

This project implements a Decision Tree Classifier to predict the presence of sleep disorders using health and lifestyle data.

🛠️ Tools and Technologies
Tool/Library	Purpose
Python 3.x	Programming Language
Jupyter/Kaggle Notebook	Implementation Platform
pandas, numpy	Data Manipulation & Numerical Computation
matplotlib, seaborn	Data Visualization
scikit-learn	Machine Learning Model & Evaluation
kagglehub	Dataset Download
OS library	File Path Handling
📊 Dataset
Dataset Name: Sleep Health and Lifestyle Dataset

Source: Kaggle

Type: CSV

Instances: ~374 rows (352 after preprocessing)

Features Used
Feature	Description
Gender	Male / Female
Age	Age of individual
Occupation	Type of profession
Sleep Duration	Average sleep hours per day
Quality of Sleep	Subjective rating (scale 1-10)
Physical Activity Level	Minutes of exercise per day
Stress Level	Stress rating (scale 1-10)
BMI Category	Underweight / Normal / Overweight / Obese
Heart Rate	Resting pulse rate (bpm)
Daily Steps	Number of steps per day
Blood Pressure	Systolic/Diastolic (split into two columns)
Sleep Disorder	Target variable (None / Insomnia / Sleep Apnea)
🔧 Data Preprocessing
[i] Handling Complex Columns
Split "Blood Pressure" column (e.g., "120/80") into two numeric columns:

BP_Systolic

BP_Diastolic

[ii] Missing Values
Identified and dropped rows with missing/invalid values using df.dropna()

[iii] Encoding Categorical Variables
Applied Label Encoding for:

Gender

Occupation

BMI Category

Sleep Disorder

[iv] Final Dataset
Rows: 352

Columns: 13

📈 Statistical Analysis
Descriptive Statistics
Metric	Value
Mean Sleep Duration	7.05 hours
Median Sleep Duration	7.0 hours
Standard Deviation of Stress Level	2.13
Correlation Insights
Positive correlation between Stress Level and Sleep Disorder

Positive correlation between Heart Rate and Sleep Disorder

Negative correlation between Physical Activity and Sleep Disorder

🤖 Model Implementation
[i] Data Splitting
Training Set: 80% of data

Testing Set: 20% of data

Used train_test_split with random_state=42

[ii] Model Selection - Decision Tree Classifier
Reasons for selection:

Handles both numerical and categorical data

Provides interpretability via feature importance

Models non-linear relationships effectively

[iii] Training Code
python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
📊 Model Evaluation
Performance Metrics
Metric	Score
Accuracy	0.87
Precision	0.85
Recall	0.84
F1-Score	0.84
Interpretation
The Decision Tree achieved an accuracy of 87%, demonstrating strong predictive performance for sleep disorder classification.

🔍 Feature Importance Analysis
Feature	Importance
Stress Level	0.24
Sleep Duration	0.21
Physical Activity Level	0.15
Quality of Sleep	0.13
BMI Category	0.10
Age	0.07
Heart Rate	0.05
Daily Steps	0.03
BP Systolic	0.02
Key Insights
Stress Level and Sleep Duration are the most influential predictors

Findings align with medical literature linking stress and poor sleep quality to sleep disorders

📊 Visualizations
Distribution Plot: Sleep Duration

Correlation Heatmap: Feature relationships

Confusion Matrix: Classification performance

Feature Importance Bar Chart: Key predictors

💡 Results and Discussion
Key Findings
✅ 87% accuracy achieved on test set

✅ High recall indicates effective identification of individuals with sleep disorders

✅ Stress level and sleep duration identified as key contributors

✅ Physical activity shown as protective factor

Limitations
📉 Small dataset size may affect generalization

📉 Slight class imbalance (fewer "Sleep Apnea" cases)

📉 Model prone to overfitting

Future Improvements
🔄 Ensemble methods (Random Forest, XGBoost) for better robustness

🔄 Larger, more balanced dataset

🔄 Hyperparameter tuning for optimization

📱 Source Code
https://colab.research.google.com/assets/colab-badge.svg

🎯 Conclusion
This project successfully implemented a Decision Tree Classifier to predict sleep disorders using health and lifestyle data. The model achieved 87% accuracy and identified stress level, sleep duration, and physical activity as key determinants of sleep health. The findings provide valuable insights for preventive healthcare and lifestyle interventions.

<div align="center">
University of Asia Pacific | Department of CSE | CSE 404
Artificial Intelligence and Expert Systems Lab Project

</div>
