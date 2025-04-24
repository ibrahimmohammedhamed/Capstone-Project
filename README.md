# Capstone-Project
Enhancing E-Commerce Decision-Making with an Online Retail Prediction Model
 Overview
This project focuses on developing a machine learning-based prediction model to extract actionable insights from online retail transaction data. The goal is to help e-commerce businesses make data-driven decisions in areas like sales forecasting, inventory management, and customer behavior analysis.

 Dataset
Source: Online Retail Dataset from UCI Machine Learning Repository

Features: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

Size: ~500,000+ rows of real-world e-commerce transactions

🧹 Data Preprocessing
Removed missing CustomerID entries
Removed negative/zero quantities
Created TotalRevenue = Quantity * UnitPrice
Performed label encoding for categorical features
Extracted time-based features (Month, Hour, etc.)

📈 Exploratory Data Analysis (EDA)
Sales trends over time
Revenue by country
Top products by revenue
RFM (Recency, Frequency, Monetary) segmentation

🤖 Models Used
Linear Regression
Random Forest Regressor

Evaluation Metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

🧪 Results
Model	MAE	RMSE
Linear Regression	18.08	432.53
Random Forest	3.70	631.76
Random Forest had the best MAE, but Linear Regression offered more stable predictions.

✅ Key Findings
Clear seasonality and purchasing peaks were found
Certain products and countries significantly drive revenue
Predictive modeling can effectively support strategic decisions

🚀 Future Work
Use advanced models (XGBoost, LSTM)
Incorporate promotional or holiday data
Deploy as a dashboard or forecasting tool

Author
Ibrahim Mohammed Hamed
Master's in Data Science; Capstone Project.
