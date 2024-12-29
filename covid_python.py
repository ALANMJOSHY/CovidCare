# %% [markdown]
# # Project Name: Covid Mortality Prediction using ML



# %% [markdown]
# ## Time Line of the Project:
# 
# - Data Analysis
# - Data Preprocessing
# - Model Building and Prediction using ML models
# 

# %% [markdown]
# ## Importing Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# %%
df= pd.read_csv("Covid_Mortality_Prediction.csv")

# %%
df.head()

# %% [markdown]
# ## Data Analysis

# %%
df.describe()

# %%
df.isnull().sum()

# %%
df.shape

# %%
df.info()

# %% [markdown]
# ### Handling NAN Values

# %% [markdown]
# #### 1) For float variables

# %%
from sklearn.impute import SimpleImputer  #to handle float values:   impute --->fill

si = SimpleImputer(missing_values=np.nan, strategy='mean')

# %%
float_col = df.select_dtypes(include='float64').columns

# %%
si.fit(df[float_col])  #transport all null values in to mean

# %%
float_col

# %%
df[float_col].shape

# %%
df[float_col] = si.transform(df[float_col])

# %%
df[float_col].isna()

# %%
df.isna().sum().value_counts()

# %%
df.info()

# %%
x = df.drop(columns='outcome')

y = df[['outcome']]

# %%
y.value_counts()

# %% [markdown]
# #### 2) For Dependent variable

# %%
SI =  SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# %%
SI.fit_transform(y)

# %%
y = pd.DataFrame(y, columns=['outcome'], dtype='int64')

# %%
y.dtypes

# %%


# %%
df_final = x.copy()

df_final['outcome'] = y

# %%
x.dtypes

# %%
df_final.isnull().sum()

# %% [markdown]
# ### Visualising our Dependent variable

# %%
fig, ax = plt.subplots(figsize=(8,5), dpi=100)

patches, texts, autotexts = ax.pie(df_final['outcome'].value_counts(), autopct= '%1.1f%%', shadow=True, 
                                   startangle=90, explode=(0.1, 0), labels=['Alive','Death'])

plt.setp(autotexts, size=12, color = 'black', weight='bold')
autotexts[1].set_color('white');

plt.title('Outcome Distribution', fontsize=14)
plt.show()

# %%
import plotly.express as px
fig = px.histogram(df, x="age", color="outcome", marginal="box", hover_data=df.columns)
fig.show()

# %%
fig = px.histogram(df, x="BMI", color="outcome", marginal="box", hover_data=df.columns)
fig.show()

# %%
fig = px.histogram(df, x="SP O2", color="outcome", marginal="box", hover_data=df.columns)
fig.show()

# %%
fig = px.histogram(df, x="heart rate", color="outcome", marginal="box", hover_data=df.columns)
fig.show()

# %%
df_final['gendera'].value_counts()

# %%
df_final['outcome'].value_counts()

# %%
plt.figure(figsize=(12,8))
plot = sns.countplot(df_final['gendera'], hue=df_final['outcome'])
plt.xlabel('Gender', fontsize=14, weight='bold')
plt.ylabel('Count', fontsize=14, weight='bold')
plt.xticks(np.arange(2), ['Male', 'Female'], rotation='vertical', weight='bold')

for i in plot.patches:
  plot.annotate(format(i.get_height()),
                (i.get_x() + i.get_width()/2,i.get_height()), ha='center', va='center',size=10, xytext=(0,8),textcoords='offset points') 

plt.show()

# %% [markdown]
# ### Correlation

# %%
col = ['group', 'gendera', 'hypertensive','atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
       'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'outcome']

# %%
corr = df_final[col].corr()

# %%
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='YlOrBr')

# %% [markdown]
# #### Distribution of Continuous var

# %%
plt.figure(figsize=(9,4))
df_final['age'].plot(kind='kde')

# %%
plt.figure(figsize=(10,5))
df_final['EF'].plot(kind='kde')

# %%
plt.figure(figsize=(10,5))
df_final['RBC'].plot(kind='kde')

# %%
plt.figure(figsize=(10,5))
df_final['Creatinine'].plot(kind='kde')

# %%
plt.figure(figsize=(10,5))
df_final['Blood calcium'].plot(kind='kde')

# %%
df_final.head()

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Splitting our data

# %%
x = df_final.drop(columns='outcome')
y = df_final[['outcome']]

# %% [markdown]
# ### Standardizing our data

# %%
from sklearn.preprocessing import StandardScaler

# %%
scale= StandardScaler()

# %%
scaled= scale.fit_transform(x)

# %%
final_x= pd.DataFrame(scaled,columns= x.columns)

# %%
final_x.head()

# %%
y.head()

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=123)

# %%
print(x_train.shape, x_test.shape)

# %%
x_train.drop(columns = 'ID', inplace=True)
x_test.drop(columns='ID', inplace=True)

# %%
x_train.head()

# %% [markdown]
# ## Model Development using ML

# %% [markdown]
# Model 1: Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression #logistic regression used for classification
lr_model=LogisticRegression()
lr_model.fit(x_train,y_train)
y_pred_lr=lr_model.predict(x_test)
y_pred_lr

# %%
from sklearn.metrics import accuracy_score
lr_score=accuracy_score(y_test,y_pred_lr)
lr_score

# %% [markdown]
# Model 2: RandomForestClassifier

# %%
from sklearn.ensemble import RandomForestClassifier
rf_model= RandomForestClassifier()
rf_model.fit(x_train,y_train)
y_pred_rf = rf_model.predict(x_test)
y_pred_rf

# %%
rf_score=accuracy_score(y_test,y_pred_rf)
rf_score

# %% [markdown]
# ### We will use the XG Boost Classifier model 

# %%
from xgboost import XGBClassifier, plot_tree, plot_importance

# %%
xgb = XGBClassifier(random_state=42)

# %%
xgb.fit(x_train, y_train)

# %%
pred_xgb = xgb.predict(x_test)

# %%
pred_xgb

# %%
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# %%
xgb_score=accuracy_score(y_test,pred_xgb)
xgb_score

# %%
cf = confusion_matrix(y_test, pred_xgb)

# %%
cf

# %%
print(classification_report(y_test, pred_xgb))

# %% [markdown]
# ### Comparing Values

# %%
combine = np.concatenate((y_test.values.reshape(len(y_test),1), pred_xgb.reshape(len(pred_xgb),1)),1)

# %%
combine_result = pd.DataFrame(combine,  columns=['y_test', 'pred_xgb'])

# %%
combine_result

# %% [markdown]
# #### Plotting ROC and Accuracy Curve

# %%
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

# %%
plot_roc_curve(xgb, x_test, y_test)
plt.plot([0,1], [0,1], color='magenta', ls='-')

# %% [markdown]
# 

# %%
x_train.shape

# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Assuming x_train is a NumPy array with shape (823, 49)
# Ensure the correct shape by reshaping it
x_train = np.reshape(x_train, (823, 49))

# Assuming y_train is your target variable

# Define the model
model = keras.Sequential([
    keras.layers.Dense(50, input_shape=(49,), activation='relu'),  # Update input_shape to (49,)
    keras.layers.Dense(25, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=100)


# %%
# Assuming you have a separate x_test and y_test for testing
# Make sure to preprocess your x_test similarly to x_train (reshape if needed)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the accuracy
print("Test Accuracy:", test_accuracy)


# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming x_train, y_train, x_test, y_test are your datasets
# Assuming x_train has shape (823, 49)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Define the model with dropout layers and adjusted learning rate
model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(49,), activation='relu'),
    keras.layers.Dropout(0.5),  
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with more epochs and validation data
model.fit(x_train_scaled, y_train, epochs=200, validation_data=(x_val_scaled, y_val))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)

# Print the test accuracy
print("Test Accuracy:", test_accuracy)


# %%


# %%
plt.bar(['Logistic','Random Forest','XGBoost','DeepNeuralNetwork'],[lr_score,rf_score,xgb_score,test_accuracy])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy_score")
plt.show()

# %%
df.sample(10)

# %%
row_index = 1164  # Replace this with the index of the row you want to access
selected_row = df.iloc[row_index]

 #Convert the values to integers if needed
formatted_values = [int(value) if isinstance(value, (int, float)) else value for value in selected_row.values]

# Now you can retrieve the values as a NumPy array
row_values = selected_row.values

#print(row_values)
print("\t".join(map(str, formatted_values)))

# %%
row_index = 1164  # Replace this with the index of the row you want to access
selected_row = df.iloc[row_index]

# Convert the values to integers if needed
formatted_values = [f"{value:,}" if isinstance(value, (int, float)) else value for value in selected_row.values]

# Now you can retrieve the values as a NumPy array
row_values = selected_row.values

# Print the values in the desired format
print("\t".join(map(str, formatted_values)))


# %%
# For scikit-learn versions 0.23 and later
from joblib import dump, load


# Save the model to a file
dump(xgb, 'Covid_Mortality_model.pkl')

# %%
x_test.head()

# %%
df.shape

# %%
x_test_scaled.shape

# %%

xgb.predict([[1,75,2,	30,	0,	0,	0,	0,	1,	0,	0,	0,	1,101,140,65,20,36,	96,	1425,30,3,31,31,98,14,12,246,80,0,12,17,1,2384,60,1,20,	147,4,138,8,98,11,1,7,33,0,78,55]])

# %%
xgb.predict([[1,78,2,37.85143414,1,0,0,1,0,0,0,0,0,76.38461538,95.44444444,60.25925926,21.75,36.12037037,94.38461538,1766,34.1625,4.2175,26.0125,32.125,81,19.0625,4.8375,172.25,70.9,0.5,17.9,14.2,1.2,24440,24,1.3,32.72727273,88,3.481818182,142.8181818,8.32,107.4545455,12.54545455,2.01,7.333333333,26.36363636,0.75,52,55]])

# %%
#model.predict([[1,78,2,37.85143414,1,0,0,1,0,0,0,0,0,76.38461538,95.44444444,60.25925926,21.75,36.12037037,94.38461538,1766,34.1625,4.2175,26.0125,32.125,81,19.0625,4.8375,172.25,70.9,0.5,17.9,14.2,1.2,24440,24,1.3,32.72727273,88,3.481818182,142.8181818,8.32,107.4545455,12.54545455,2.01,7.333333333,26.36363636,0.75,52,55]])

# %%



