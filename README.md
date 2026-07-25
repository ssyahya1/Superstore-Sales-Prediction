# 🛒 Superstore Sales Prediction using Machine Learning

## Project Overview

This project focuses on predicting **product sales** using the Kaggle Superstore dataset. The objective was to build a complete machine learning workflow, starting from data preprocessing and exploratory data analysis (EDA) to feature engineering, model training, evaluation, and comparison.

Multiple regression algorithms were implemented and compared to identify the best-performing model.



## Dataset

**Dataset:** Kaggle Superstore Sales Dataset

### Features

Order Date
Ship Date
Ship Mode
Segment
Country
City
State
Region
Category
Sub-Category
Customer Information
Product Information
Sales (Target Variable)



## Objective

Predict the **Sales** value for each order using historical order and product information.



## Technologies Used

Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib



## Exploratory Data Analysis (EDA)

The following analyses were performed:

Sales distribution
Sales by Category
Sales by Region
Sales by State
Sales by Segment
Sales by Month
Sales by Year
Top-selling Products
Top-selling Cities
Ship Mode Analysis

Visualizations were created using **Matplotlib** and **Seaborn** to understand sales trends and customer behaviour.



## Data Preprocessing

The following preprocessing steps were applied:

Removed missing values
Removed duplicate records
Converted date columns into datetime format
Created new date-based features
One-Hot Encoded categorical variables
Removed unnecessary columns
Standardized numerical features (for Linear Regression)



## Feature Engineering

Additional features were extracted from the order dates:

Order Year
Order Month
Order Day
Order Day of Week
Order Quarter
Shipping Days

These engineered features helped provide more meaningful information to the regression models.



## Machine Learning Models

The following regression algorithms were trained and evaluated:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor



## Evaluation Metrics

Models were evaluated using:

 R² Score
Mean Absolute Error (MAE)
 Root Mean Squared Error (RMSE)

A comparison table was created to compare the performance of each model.



## Results

Among the tested algorithms, **XGBoost Regressor** achieved the best overall performance.

Although the R² score was relatively low (approximately **0.19**), all models produced similar results. This indicates that the available features in the dataset have limited predictive power for forecasting sales. The project demonstrates that model performance depends not only on algorithm selection but also on the quality and relevance of the available features.



## Project Structure

```
Superstore-Sales-Prediction/
│
├── train.csv
├── Superstore_Sales_Prediction.ipynb
├── requirements.txt
├── README.md
└── sales_prediction_xgboost.pkl
```



## How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/Superstore-Sales-Prediction.git
```

2. Install the required libraries

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook

```bash
jupyter notebook
```

4. Run all notebook cells from top to bottom.



## Skills Demonstrated

Data Cleaning
 Exploratory Data Analysis
 Feature Engineering
 One-Hot Encoding
 Feature Scaling
 Regression Modelling
 Model Evaluation
 Model Comparison
 Machine Learning Workflow



## Future Improvements

 Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
 Feature selection techniques
 Cross-validation
 Deployment using Streamlit
 Experiment with CatBoost and LightGBM
 Use a richer sales dataset with additional business features



## Conclusion

This project demonstrates an end-to-end machine learning regression pipeline using real-world retail sales data. It covers the complete workflow from preprocessing and visualization to model comparison and evaluation. While the dataset limits predictive performance, the project highlights practical machine learning techniques and provides a strong foundation for future regression projects.



##  Author

**Syed Muhammad Yahya**

If you found this project helpful, consider giving it a ⭐ on GitHub!
