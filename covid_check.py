#!/usr/bin/env python
# coding: utf-8
import joblib
# In[4]:


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("C:\\Users\\ALAN M JOSHY\\Downloads\\Covid Dataset.csv")


# Encoding categorical variable 'outcome' (assuming it's already encoded)
y = df["COVID-19"]
x = df.drop(["COVID-19"], axis=1)
x = x.values
y = y.values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Data scaling (not necessary for XGBoost)

# Define the model
model = XGBClassifier()

# Train the model
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Single instance prediction
single_instance = np.array([[1,1,1,1,1,0,0,0,0,1,1,1,1,0,1,0,1,1,0,0]])
single_instance_pred = model.predict(single_instance)
print("\nPrediction for single instance:", single_instance_pred)

joblib.dump(model,"covid_check.pkl")
# In[ ]:





# In[ ]:




