# 🌟 Customer Churn Prediction  

## 📖 Overview  
This repository contains a project focused on predicting customer churn in the telecommunications industry. Leveraging advanced machine learning techniques and the CRISP-DM methodology, the project aims to identify key factors driving churn and provide actionable insights to enhance customer retention.  

## 🎯 Objectives  
### Business Objectives  
1. **Enhance customer satisfaction** by identifying key factors driving churn.  
2. **Minimize churn rates** by identifying high-risk customers and implementing targeted retention strategies.  

### Data Science Objectives  
1. Build and evaluate machine learning models to determine the best-performing model and analyze key churn-driving features.  
2. Develop algorithms to forecast churn and cluster customers to interpret behavior patterns.  

## 📋 Methodology  
This project follows the **CRISP-DM Framework**:  
1. **Business Understanding**: Define goals and objectives of churn prediction.  
2. **Data Understanding**: Analyze datasets to identify patterns and distributions.  
3. **Data Preparation**: Clean and preprocess data for machine learning.  
4. **Modeling**: Train and evaluate predictive models.  
5. **Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.  
6. **Deployment**: Deploy the best-performing model using Streamlit for real-world predictions.  

## 📊 Dataset  
The datasets (`churn-bigml-20.csv` and `churn-bigml-80.csv`) consist of customer information collected by a telecom company.  

### Key Features  
- **State**: Customer's state (categorical).  
- **Account length**: Duration of account in days (numeric).  
- **Area code**: Customer's area code (numeric).  
- **International plan**: Presence of an international calling plan (Yes/No).  
- **Voice mail plan**: Presence of a voice mail plan (Yes/No).  
- **Service usage metrics**: Details on calls, charges, and minutes for daytime, evening, nighttime, and international calls.  
- **Customer service calls**: Number of calls to customer service (numeric).  
- **Churn**: Target variable indicating whether the customer churned (Yes/No).  

## 🛠️ Machine Learning Models  
Several machine learning models were trained and evaluated. Below are the results:  

| Model               | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |  
|---------------------|----------|-----------|----------|----------|----------|  
| Logistic Regression | 0.795591 | 0.782027  | 0.819639 | 0.800391 | 0.871555 |  
| Gradient Boosting   | 0.950902 | 0.978723  | 0.921844 | 0.949432 | 0.985699 |  
| Random Forest       | 0.961924 | 0.983229  | 0.939880 | 0.961066 | 0.995153 |  
| SVM                 | 0.910822 | 0.921811  | 0.897796 | 0.909645 | 0.970346 |  
| XGBoost             | 0.976954 | 0.985714  | 0.967936 | 0.976744 | 0.994671 |  
| Decision Tree       | 0.916834 | 0.904669  | 0.931864 | 0.918065 | 0.916834 |  
| Neural Network      | 0.945892 | 0.937132  | 0.955912 | 0.946429 | 0.982876 |  
| AdaBoost            | 0.865731 | 0.902870  | 0.819639 | 0.859244 | 0.936283 |  

**Selected Model**: XGBoost provided the best overall performance and was selected for deployment.  

## 🚀 Deployment  
The model was deployed using **Streamlit** to provide an interactive interface for predicting customer churn.  



