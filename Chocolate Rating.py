#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#     <h1 style="font-size: 48px;">Chocolate Rating - A Regression Problem</h1>
# </div>

# ![Project 6 - Chocolate Rating](Chocolate_Rating.jpg)

# In[1]:


import pandas as pd
import numpy as np
import math as mth
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 2000)


# # Load information

# In[2]:


chocolate = pd.read_csv("C:/Users/Usuario/Desktop/Machine Learning with Python/Repositorio de Projectos/Project 6 - Chocolate Rating/Raúl Reaño Araya - chocolate_ratings.csv")
chocolate.head()


# In[3]:


chocolate.shape


# In[4]:


chocolate.duplicated().sum() 


# In[5]:


chocolate.dtypes


# In[6]:


chocolate.info()


# # Data Cleaning

# ### Finding null values (values = NaN)

# In[7]:


chocolate.isnull().mean()


# In[8]:


chocolate = chocolate.dropna()


# In[9]:


chocolate.shape


# In[10]:


chocolate.describe().round(3)


# ### Finding missing values (values = 0)

# In[11]:


chocolate = chocolate.drop(['REF'], axis=1)


# In[12]:


(chocolate==0).sum()/(chocolate==0).count()*100


# In[13]:


cat_features = ["Company (Manufacturer)","Company Location","Country of Bean Origin","Specific Bean Origin or Bar Name",
                "Cocoa Percent","Ingredients","Most Memorable Characteristics"]


# # EDA

# In[14]:


plt.figure(figsize=(12, 7))
sns.countplot(x=chocolate["Rating"])
plt.title('Histrogram of Chocolate Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
# Show the visualization
plt.tight_layout()
plt.show();


# In[15]:


for col in chocolate.select_dtypes(include=['O']).columns:
    print(f"{col}: {chocolate[col].nunique()} valores únicos")


# In[16]:


plt.figure(figsize=(15, 5))
feature = cat_features[0]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[17]:


plt.figure(figsize=(15, 5))
feature = cat_features[1]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[18]:


plt.figure(figsize=(15, 5))
feature = cat_features[2]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[19]:


plt.figure(figsize=(15, 5))
feature = cat_features[3]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[20]:


plt.figure(figsize=(15, 5))
feature = cat_features[4]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[21]:


plt.figure(figsize=(15, 5))
feature = cat_features[5]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# In[22]:


plt.figure(figsize=(15, 5))
feature = cat_features[6]
count = chocolate[feature].value_counts()
percent = 100*chocolate[feature].value_counts(normalize=True)
df0 = pd.DataFrame({'count':count, 'percent':percent.round(1)})
print(df0)
count.plot(kind='bar', title=feature, );


# # Feature Engineering

# In[23]:


chocolate = chocolate.drop(['Review Date'], axis=1)


# In[24]:


def assign_continent_codes_to_country_location(country):
    asia = ['Japan', 'Vietnam', 'Singapore', 'Taiwan', 'Israel', 'South Korea', 'Malaysia', 'Philippines', 'Thailand', 'Indonesia', 'India']
    america = ['U.S.A.', 'Canada', 'Mexico', 'Ecuador', 'Venezuela', 'Colombia', 'Brazil', 'Peru', 'Guatemala', 'Nicaragua', 'Costa Rica', 'Dominican Republic', 'El Salvador', 'Honduras', 'Puerto Rico', 'St. Lucia', 'St. Vincent-Grenadines', 'Martinique', 'Grenada', 'St.Vincent-Grenadines', 'Suriname']
    europe = ['France', 'U.K.', 'Italy', 'Belgium', 'Switzerland', 'Germany', 'Spain', 'Denmark', 'Austria', 'Hungary', 'Netherlands', 'Lithuania', 'Poland', 'Sweden', 'Ireland', 'Scotland', 'Portugal', 'Norway', 'Finland', 'Iceland', 'Czech Republic', 'Russia']
    oceania = ['Australia', 'New Zealand', 'Fiji', 'Vanuatu']
    africa = ['South Africa', 'Sao Tome & Principe', 'Ghana']

    if country in asia:
        return 0
    elif country in america:
        return 1
    elif country in europe:
        return 2
    elif country in oceania:
        return 3
    elif country in africa:
        return 4
    else:
        return 5  # For another countries in the list


# In[25]:


chocolate['Company Location'] = chocolate['Company Location'].apply(assign_continent_codes_to_country_location)


# In[26]:


chocolate['Company Location'].value_counts()


# In[27]:


chocolate['Cocoa Percent'] = chocolate['Cocoa Percent'].str.rstrip('%').astype('float') / 100.0


# In[28]:


plt.figure(figsize=(10, 8))
sns.lineplot(x="Cocoa Percent", y="Rating",data=chocolate)
plt.title('Rating value for Cocoa Percent')
plt.xlabel('Cocoa percent')
plt.ylabel('Rating  Value')
plt.tight_layout()
plt.show();


# In[29]:


def assign_continent_codes_to_country_of_begin_origin(country):
    asia = ['Vietnam', 'Indonesia', 'Papua New Guinea', 'Philippines', 'India', 'Sri Lanka', 'Malaysia', 'Thailand', 'Taiwan']
    america = ['Venezuela', 'Peru', 'Dominican Republic', 'Ecuador', 'Nicaragua', 'Bolivia', 'Colombia', 'Brazil', 'Belize', 'Guatemala', 'Mexico', 'Costa Rica', 'U.S.A.', 'Haiti', 'Honduras', 'Jamaica', 'Grenada', 'Cuba', 'Panama', 'St. Lucia', 'Puerto Rico', 'El Salvador', 'Trinidad', 'Ivory Coast', 'Togo', 'Sao Tome', 'Tobago', 'Principe', 'Suriname']
    africa = ['Madagascar', 'Cameroon', 'Nigeria', 'Ghana', 'Gabon', 'Ivory Coast', 'Sierra Leone', 'Congo', 'Liberia', 'Sao Tome & Principe']
    europe = ['Blend']
    oceania = ['Fiji', 'Vanuatu', 'Solomon Islands', 'Australia', 'New Zealand']

    if country in asia:
        return 0
    elif country in america:
        return 1
    elif country in europe:
        return 2
    elif country in oceania:
        return 3
    elif country in africa:
        return 4
    else:
        return 5  # For countries not covered in the list


# In[30]:


chocolate['Country of Bean Origin'] = chocolate['Country of Bean Origin'].apply(assign_continent_codes_to_country_of_begin_origin)
chocolate.head()


# In[31]:


get_ipython().system('pip install category_encoders')


# In[32]:


from category_encoders import TargetEncoder


# In[33]:


encoder=TargetEncoder()
("")
chocolate["Company (Manufacturer)"]=encoder.fit_transform(chocolate["Company (Manufacturer)"],chocolate["Rating"])
chocolate["Specific Bean Origin or Bar Name"]=encoder.fit_transform(chocolate["Specific Bean Origin or Bar Name"],chocolate["Rating"])
chocolate["Ingredients"]=encoder.fit_transform(chocolate["Ingredients"],chocolate["Rating"])
chocolate["Most Memorable Characteristics"]=encoder.fit_transform(chocolate["Most Memorable Characteristics"],chocolate["Rating"])


# In[34]:


# Matriz de correlación
plt.figure(figsize=(20, 7))
heatmap = sns.heatmap(chocolate.corr(),cmap='viridis',annot=True)
heatmap.set_title('Correlation Matrix', fontdict={'fontsize':25}, pad=12);


# ## Finding outliers, Removing inconsistence records and delimit outliers

# In[35]:


chocolate.head()


# In[36]:


# A visualization to see the outliers in data
plt.figure(figsize=(10, 6))
chocolate.boxplot(rot=45)
plt.title('Features outliers')
plt.xlabel('Features')
plt.ylabel('Values')
plt.tight_layout()
plt.show();


# In[37]:


def detect_outliers(column):
    Q1 = np.percentile(column, 25)
    Q3 = np.percentile(column, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)


# In[38]:


fig, axs = plt.subplots(nrows=len(chocolate.columns), figsize=(8, 6 * len(chocolate.columns)))
# Iterate through the columns and create a chart for each one
for i, columna in enumerate(chocolate.columns):
    axs[i].boxplot(chocolate[columna])
    axs[i].set_title(f'Boxplot of {columna}')
    # Detect and highlight outliers
    outliers = detect_outliers(chocolate[columna])
    axs[i].plot(np.where(outliers)[0] + 1, chocolate[columna][outliers], 'ro', label='Outliers')
    axs[i].legend()

plt.tight_layout()
plt.show()


# In[39]:


def delimit_variables(df):
    percentiles = df.select_dtypes(include=np.float64).quantile([0.01, 0.95])
    
    def delimit(col):
        if col.dtype == np.float64:
            return col.clip(lower=percentiles.loc[0.01, col.name], upper=percentiles.loc[0.95, col.name])
        else:
            return col
    
    return df.apply(delimit)


# In[40]:


chocolate = delimit_variables(chocolate)


# In[41]:


chocolate.shape


# In[42]:


# A visualization to see the outliers in data
plt.figure(figsize=(10, 6))
chocolate.boxplot(rot=45)
plt.title('Features outliers')
plt.xlabel('Features')
plt.ylabel('Values')
plt.tight_layout()
plt.show();


# In[43]:


fig, axs = plt.subplots(nrows=len(chocolate.columns), figsize=(8, 6 * len(chocolate.columns)))
# Iterate through the columns and create a chart for each one
for i, columna in enumerate(chocolate.columns):
    axs[i].boxplot(chocolate[columna])
    axs[i].set_title(f'Boxplot of {columna}')
    # Detect and highlight outliers
    outliers = detect_outliers(chocolate[columna])
    axs[i].plot(np.where(outliers)[0] + 1, chocolate[columna][outliers], 'ro', label='Outliers')
    axs[i].legend()

plt.tight_layout()
plt.show()


# # Slicing training and test features

# In[44]:


X = chocolate.drop(["Rating"], axis=1)    # independent variables
y = chocolate["Rating"]   #Objective or dependent target


# In[45]:


X.shape


# In[46]:


y.head()


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)


# In[48]:


mask = np.triu(np.ones_like(chocolate.corr(), dtype=bool))
plt.figure(figsize=(10, 5))
plt.title("Correlation Graph", size=20)
sns.heatmap(chocolate.corr(), annot=True, fmt=".3f",
            vmin=-1, vmax=1, linewidth=1,
            center=0, mask=mask, cmap="summer")
plt.show();


# In[49]:


print("Original Data",X.shape)
print("Data for Train",X_train.shape)
print("Data for Test",X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[50]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()
#scaling the training set
X_train = std.fit_transform(X_train)
#scaling the test set
X_test = std.transform (X_test)


# # Modeling

# ## 1. XGBoost

# In[51]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
# Import XGBRegressor
from xgboost import XGBRegressor
# Calling the objetct
xgb = XGBRegressor()
# Training the model
xgb.fit(X_train, y_train)
# making the predictions
y_pred_xgb = xgb.predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_xgb = xgb.predict(X_train)


# In[52]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_xgb)}')


# ### 1.1 Tuning Hyperparameters for XGBoost

# In[53]:


from sklearn.model_selection import GridSearchCV
param_grid_for_XGBoost = {
    'n_estimators': [10, 20, 50],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 2, 4],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'alpha': [0, 0.1, 0.2]  # Regularización L1
}


# In[54]:


grid_search_for_XGBoost = GridSearchCV(estimator=xgb, 
                           param_grid=param_grid_for_XGBoost, 
                           scoring='neg_mean_squared_error', cv=5)


# In[55]:


grid_search_for_XGBoost.fit(X_train, y_train)


# In[56]:


best_params_to_XGBoost = grid_search_for_XGBoost.best_params_
best_params_to_XGBoost


# In[57]:


best_xgb = XGBRegressor(**best_params_to_XGBoost)
best_xgb.fit(X_train, y_train)
# making the predictions
y_pred_xgb = best_xgb.predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_xgb = best_xgb.predict(X_train)


# In[58]:


#Valor real valor pronóstico
print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_xgb)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_xgb)}')


# In[59]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_xgb, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ## 2. Random Forest

# In[60]:


# build the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
# Loading the function 
rf = RandomForestRegressor(n_estimators=10,  
                           max_depth=5, 
                           min_samples_split=10,
                           min_samples_leaf=4)# 100 trees for default
# Training the model
rf.fit(X_train, y_train)
# making the predictions
y_pred_rf = rf.predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_rf = rf.predict(X_train)


# In[61]:


#Valor real valor pronóstico
print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_rf)}')


# ### 2.1 Tuning Hyperparameters for XGBoost

# In[62]:


param_grid_for_rf = {
    'n_estimators': [10, 20, 30],
    'max_depth': [3, 5, 8 , 12],
    'min_samples_split': [2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 4, 5]
}


# In[63]:


from sklearn.metrics import make_scorer, mean_squared_error
# Define una función de puntuación personalizada para el RMSE
def rmse_score(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)
rmse_scorer = make_scorer(rmse_score, greater_is_better=False)


# In[64]:


grid_search_for_rf = GridSearchCV(estimator=rf, 
                           param_grid=param_grid_for_rf, 
                           scoring=rmse_scorer, 
                           cv=5)


# In[65]:


grid_search_for_rf.fit(X_train, y_train)


# In[66]:


best_params_to_rf = grid_search_for_rf.best_params_
if 'alpha' in best_params_to_rf:
    del best_params_to_rf['alpha']
best_params_to_rf = grid_search_for_rf.best_params_
best_params_to_rf


# In[67]:


best_rf = RandomForestRegressor(**best_params_to_rf)
best_rf.fit(X_train, y_train)


# In[68]:


y_pred_rf = best_rf.predict(X_test)
y_train_rf = best_rf.predict(X_train)


# In[69]:


#Valor real valor pronóstico
print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_rf)}')


# In[70]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="y", label="Actual", linewidth=2)
plt.plot(y_pred_rf, c="r", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ## 3. LightGBM

# In[71]:


# build the lightgbm model
import lightgbm as lgb
from lightgbm import LGBMRegressor
lgbm = lgb.LGBMRegressor()
lgbm.fit(X_train, y_train)
# predict the results
y_pred_lgbm=lgbm.fit(X_train, y_train).predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_lgbm = lgbm.fit(X_train, y_train).predict(X_train)


# In[72]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_lgbm)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_lgbm)}')


# In[73]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_lgbm, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ### 3.1 Tuning Hyperparameters for LigthGBM

# In[74]:


param_grid_for_lgbm = {
    'n_estimators': [10, 20, 30, 50],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}


# In[75]:


grid_search_for_lgbm = GridSearchCV(estimator=lgbm, 
                           param_grid=param_grid_for_lgbm, 
                           scoring='neg_mean_squared_error', 
                           cv=5)


# In[76]:


grid_search_for_lgbm.fit(X_train, y_train)


# In[77]:


best_params_to_lgbm = grid_search_for_lgbm.best_params_
best_params_to_lgbm


# In[78]:


best_lgbm = lgb.LGBMRegressor(**best_params_to_lgbm)
best_lgbm.fit(X_train, y_train)


# In[79]:


y_pred_lgbm = best_lgbm.predict(X_test)
y_train_lgbm = best_lgbm.predict(X_train)


# In[80]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_rf)}')


# In[81]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_lgbm, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ## 4. Support Vector Machine

# In[82]:


#Calling the object
from sklearn.svm import SVR
svm = SVR()
# Training our model
svm.fit(X_train, y_train)
# making the predictions
y_pred_svm = svm.fit(X_train, y_train).predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_svm = svm.fit(X_train, y_train).predict(X_train)


# In[83]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_svm)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_svm)}')


# In[84]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_svm, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ### 4.1 Tuning Hyperparameters for Support Vector Machine

# In[85]:


param_grid_for_svm = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'epsilon': [0.1, 0.2, 0.3]
}


# In[86]:


grid_search_for_svm = GridSearchCV(estimator=svm, 
                           param_grid=param_grid_for_svm, 
                           scoring='neg_mean_squared_error', 
                           cv=5)


# In[87]:


grid_search_for_svm.fit(X_train, y_train)


# In[88]:


best_params_to_svm = grid_search_for_svm.best_params_
best_params_to_svm


# In[89]:


best_svm = SVR(**best_params_to_svm)
best_svm.fit(X_train, y_train)


# In[90]:


y_pred_svm = best_svm.predict(X_test)
y_train_svm = best_svm.predict(X_train)


# In[91]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_svm)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_svm)}')


# In[92]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_svm, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ## 5. GradientBoosting

# In[93]:


from sklearn.ensemble import GradientBoostingRegressor
# Create and train the Gradient Boosting Regressor model
gboosting = GradientBoostingRegressor(n_estimators=50,
                                      learning_rate=0.1,
                                      max_depth=4,
                                      min_samples_split=10,
                                      min_samples_leaf=4,
                                      subsample=0.8,
                                      random_state=42)
gboosting.fit(X_train, y_train)
# predict the results
y_pred_gboosting=gboosting.fit(X_train, y_train).predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_gboosting = gboosting.fit(X_train, y_train).predict(X_train)


# In[94]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_gboosting)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_gboosting)}')


# In[95]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_gboosting, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ### 5.1 Tuning Hyperparameters for Support Vector Machine

# In[96]:


param_grid_for_gboosting = {
    'n_estimators': [10, 20, 30, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 8],
    'min_samples_split': [2, 5, 3, 10],
    'min_samples_leaf': [1, 2, 3, 4]
}


# In[97]:


grid_search_for_gboosting = GridSearchCV(estimator=gboosting, 
                           param_grid=param_grid_for_gboosting, 
                           scoring='neg_mean_squared_error', 
                           cv=5)


# In[98]:


grid_search_for_gboosting.fit(X_train, y_train)


# In[99]:


best_params_to_gboosting = grid_search_for_gboosting.best_params_
best_params_to_gboosting


# In[100]:


best_gboosting = GradientBoostingRegressor(**best_params_to_gboosting)
best_gboosting.fit(X_train, y_train)


# In[101]:


y_pred_gboosting = best_gboosting.predict(X_test)
y_train_gboosting = best_gboosting.predict(X_train)


# In[102]:


#Valor real valor pronóstico
print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_gboosting)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_gboosting)}')


# In[103]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_gboosting, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# # 6. HistGradientBoosting

# In[104]:


from sklearn.ensemble import HistGradientBoostingRegressor
hgb = HistGradientBoostingRegressor()
hgb.fit(X_train, y_train)
# predict the results
y_pred_hgb=hgb.fit(X_train, y_train).predict(X_test)
# We make predictions of the model with the training set (Train)
y_train_hgb=hgb.fit(X_train, y_train).predict(X_train)


# In[105]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_hgb)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_hgb)}')


# In[106]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_hgb, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# ### 6.1 Tuning Hyperparameters for HistGradientBoosting

# In[107]:


param_grid_for_hgb = {
    'learning_rate': [0.01, 0.1, 0.125, 0.2, 0.25],
    'max_depth': [3, 4, 5, 7],
    'min_samples_leaf': [1, 2, 4, 5]
}


# In[108]:


grid_search_for_hgb = GridSearchCV(estimator=hgb, 
                                   param_grid=param_grid_for_hgb, 
                                   scoring='neg_mean_squared_error', 
                                   cv=5)


# In[109]:


grid_search_for_hgb.fit(X_train, y_train)


# In[110]:


best_params_to_hgb = grid_search_for_hgb.best_params_


# In[111]:


best_hgb = HistGradientBoostingRegressor(**best_params_to_hgb)
best_hgb.fit(X_train, y_train)


# In[112]:


y_pred_hgb = best_hgb.predict(X_test)
y_train_hgb = best_hgb.predict(X_train)


# In[113]:


print ("Root Mean Squared Error:" , np.sqrt(metrics.mean_squared_error(y_test, y_pred_hgb)))
# The coefficient of determination: 1 is perfect prediction
print(f'Coeficiente de determinación: {r2_score(y_test, y_pred_hgb)}')


# In[114]:


plt.figure(figsize=(15, 6))
plt.plot(list(y_test),  c="lightblue", label="Actual", linewidth=2)
plt.plot(y_pred_hgb, c="lime", label="predicción", linewidth=2)
plt.legend(loc='best')
plt.show();


# # Model Comparison

# In[115]:


model_comparison = pd.DataFrame({
    'Model': ["XGBoost","Random Forest","LightGBM","Support Vector Machine","GradientBoosting","HistGrandientBoosting"],
     "R2_train":[r2_score(y_train, y_train_xgb), r2_score(y_train, y_train_rf) ,r2_score(y_train, y_train_lgbm), 
                 r2_score(y_train, y_train_svm), r2_score(y_train, y_train_gboosting), r2_score(y_train, y_train_hgb)],
     "RMSE_train": [np.sqrt(mean_squared_error(y_train, y_train_xgb)), np.sqrt(mean_squared_error(y_train, y_train_rf)), 
                    np.sqrt(mean_squared_error(y_train, y_train_lgbm)), np.sqrt(mean_squared_error(y_train, y_train_svm)), 
                    np.sqrt(mean_squared_error(y_train, y_train_gboosting)), np.sqrt(mean_squared_error(y_train, y_train_hgb))],
      "R2_test":[r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_rf) ,r2_score(y_test, y_pred_lgbm),
                 r2_score(y_test, y_pred_svm), r2_score(y_test, y_pred_gboosting), r2_score(y_test, y_pred_hgb)],
     "RMSE_test": [np.sqrt(mean_squared_error(y_test, y_pred_xgb)), np.sqrt(mean_squared_error(y_test, y_pred_rf)), 
                   np.sqrt(mean_squared_error(y_test, y_pred_lgbm)), np.sqrt(mean_squared_error(y_test, y_pred_svm)), 
                   np.sqrt(mean_squared_error(y_test, y_pred_gboosting)), np.sqrt(mean_squared_error(y_test, y_pred_hgb))]})

model_comparison.sort_values(by='RMSE_test', ascending=True)


# In[116]:


y_pred_for_each_model = {
    "XGBoost": y_pred_xgb,
    "Random Forest": y_pred_rf,
    "Lightgbm": y_pred_lgbm,
    "Support Vector Machine": y_pred_svm,
    "GradientBoosting": y_pred_gboosting,
    "HisGradientBoosting": y_pred_hgb
}


# In[117]:


num_models = len(y_pred_for_each_model)
num_cols = 3
num_rows = int(np.ceil(num_models / num_cols))


# In[118]:


fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

for i, (model, pred) in enumerate(y_pred_for_each_model.items()):
    row = i // num_cols
    col = i % num_cols
    
    ax = axes[row, col] if num_rows > 1 else axes[col]
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='dashed', color='blue')
    ax.set_xlabel('y_test')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Comparison - {model}')
    ax.grid(True)
# Ajustar diseño y mostrar gráficos
plt.tight_layout()
plt.show();


# # Predicting Rating

# In[119]:


best_model_to_predict = model_comparison.loc[model_comparison['R2_test'].idxmax()]
best_model_to_predict


# In[126]:


test = chocolate.sample()
test


# In[127]:


test= test.drop('Rating', axis=1)
test


# In[128]:


lgbm.predict(test)


# In[ ]:




