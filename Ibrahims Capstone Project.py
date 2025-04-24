#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("online_retail.csv")


# In[3]:


print(df.info())


# In[4]:


print(df.head())


# In[5]:


print("\nFirst 5 Rows:")
print(df.head())


# In[6]:


print("\nMissing Values:")
print(df.isnull().sum())


# In[7]:


print("\nSummary Statistics:")
print(df.describe())


# In[8]:


print("\nDuplicate Rows:", df.duplicated().sum())


# In[9]:


df = df.drop_duplicates()


# In[10]:


df = df.dropna(subset=['CustomerID'])


# In[11]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[12]:


df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']


# In[13]:


print("\nDuplicate Rows:", df.duplicated().sum())


# In[14]:


print(df.isnull().sum())


# In[15]:


#top-selling products

plt.figure(figsize=(10, 5))
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Description")
plt.title("Top 10 Best-Selling Products")
plt.show()


# In[16]:


#Sales trends over time

plt.figure(figsize=(12, 5))
df.set_index('InvoiceDate').resample('M')['TotalRevenue'].sum().plot(marker='o', color='b')
plt.xlabel("Date")
plt.ylabel("Total Revenue")
plt.title("Monthly Revenue Trend")
plt.grid()
plt.show()


# In[17]:


#sales distribution by country
plt.figure(figsize=(12, 6))
top_countries = df.groupby('Country')['TotalRevenue'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=top_countries.index, y=top_countries.values, palette="magma")
plt.xticks(rotation=45)
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.title("Top 10 Countries by Revenue")
plt.show()


# In[18]:


df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[19]:


df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # <- This was missing!
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[21]:


encoder = LabelEncoder()
df['StockCode'] = encoder.fit_transform(df['StockCode'])
df['CustomerID'] = encoder.fit_transform(df['CustomerID'])


# In[22]:


features = ['StockCode', 'Quantity', 'UnitPrice', 'CustomerID', 'Year', 'Month', 'Day', 'DayOfWeek']
target = 'TotalRevenue'


# In[23]:


X = df[features]
y = df[target]


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[26]:


y_pred_lr = lr_model.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"Linear Regression - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}")


# In[27]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[28]:


y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))


# In[29]:


print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")


# In[30]:


models = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [mae_lr, mae_rf],
    "RMSE": [rmse_lr, rmse_rf]
})
print(models)


# In[31]:


feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


# In[ ]:




