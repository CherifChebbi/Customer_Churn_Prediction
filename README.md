# 🌟 Customer Churn Prediction  

## 📖 Overview  
This repository contains a project focused on predicting customer churn in the telecommunications industry. Leveraging advanced machine learning techniques and the CRISP-DM methodology, the project aims to identify key factors driving churn and provide actionable insights to enhance customer retention.  
![image_2024-12-18_210826740](https://github.com/user-attachments/assets/4adda434-5732-4847-9d38-4dd6bffe804c)

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
Several machine learning models were trained and evaluated.

| Model               |
|---------------------|
| Logistic Regression | 
| Gradient Boosting   | 
| Random Forest       |  
| SVM                 | 
| XGBoost             |
| Decision Tree       | 
| Neural Network      | 
| AdaBoost            | 

**Selected Model**: XGBoost provided the best overall performance and was selected for deployment.  



## 🚀 Deployment  
The model was deployed using **Streamlit** to provide an interactive interface for predicting customer churn.  



