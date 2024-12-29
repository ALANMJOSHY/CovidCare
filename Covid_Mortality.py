# %% [markdown]
# # Project Name: Covid Mortality Prediction using ML

# %% [markdown]


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


# %%
df= pd.read_csv("mortality.csv")

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
# plt.figure(figsize=(12,8))
# # plot = sns.countplot(df_final['gendera'], hue=df_final['outcome'])
# plt.xlabel('Gender', fontsize=14, weight='bold')
# plt.ylabel('Count', fontsize=14, weight='bold')
# plt.xticks(np.arange(2), ['Male', 'Female'], rotation='vertical', weight='bold')
#
# for i in plot.patches:
#   plot.annotate(format(i.get_height()),
#                 (i.get_x() + i.get_width()/2,i.get_height()), ha='center', va='center',size=10, xytext=(0,8),textcoords='offset points')
#
# plt.show()

# %% [markdown]
# ### Correlation

# %%
col = ['group', 'gendera', 'hypertensive','atrialfibrillation', 'CHD with no MI', 'diabetes', 'deficiencyanemias',
       'depression', 'Hyperlipemia', 'Renal failure', 'COPD', 'outcome']

# %%
corr = df_final[col].corr()

# # %%
# plt.figure(figsize=(12,8))
# sns.heatmap(corr, annot=True, cmap='YlOrBr')
#
# # %% [markdown]
# # #### Distribution of Continuous var
#
# # %%
# plt.figure(figsize=(9,4))
# df_final['age'].plot(kind='kde')
#
# # %%
# plt.figure(figsize=(10,5))
# df_final['EF'].plot(kind='kde')
#
# # %%
# plt.figure(figsize=(10,5))
# df_final['RBC'].plot(kind='kde')
#
# # %%
# plt.figure(figsize=(10,5))
# df_final['Creatinine'].plot(kind='kde')
#
# # %%
# plt.figure(figsize=(10,5))
# df_final['Blood calcium'].plot(kind='kde')

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
from joblib import dump, load
dump(lr_model, 'Covid_Mortality.pkl')
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
# For scikit-learn versions 0.23 and later



# Save the model to a file
dump(lr_model, 'Covid_Mortality_model.pkl')


