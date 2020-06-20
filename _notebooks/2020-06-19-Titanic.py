#!/usr/bin/env python
# coding: utf-8

# 
# <img src = 'titanic2.jpg' style= 'width:100; height:500px'/>

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set() # set seaborn as default plots
import sklearn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# # In Machine Learning, the data is mainly divided into two parts —
# * Training and Testing (the third split is validation, but you don’t have to care about that right now). 
# * Training data is for training our algorithm and Testing data is to check how well our algorithm performs

# In[101]:


# importing the training data set and naming it data_1 and also making a copy of it
data_1 = pd.read_csv('TIT_train.csv')
data_2 = data_1.copy()
data_3 = data_1.copy()


# # viewing some details of the train data set
# * train data set info
# * train data set columns
# * checking if there are missing data  and fill the isnull data
# * checking for strings and converting it to numeric data ( Data Engineering )

# In[102]:


data_1.head().T


# In[103]:


data_1.info()


# In[104]:


data_1.columns


# In[105]:


data_1.isnull()


# In[106]:


# checking all the string data type
for label, columns in data_1.items():
    if pd.api.types.is_string_dtype(columns):
        print(label)


# In[107]:


# checking all the numeric data type
for label, columns in data_1.items():
    if pd.api.types.is_numeric_dtype(columns):
        print(label)


# In[108]:


# turn string data type into category
for label, columns in data_1.items():
    if pd.api.types.is_string_dtype(columns):
        data_1[label] = columns.astype('category').cat.as_ordered()


# In[109]:


data_1.info()


# In[110]:


# checking all string data type that has null/missing value
for label, content in data_1.items():
    if pd.api.types.is_string_dtype(content):
        if pd.isna(content).sum():
            print(label)


# In[111]:


# Turn categorical variables into numbers and fill missing
for label, content in data_1.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Turn categories into numbers and add +1
        data_1[label] = pd.Categorical(content).codes+1


# In[112]:


# checking all numeric datatype that has missing valus
for label, columns in data_1.items():
    if pd.api.types.is_numeric_dtype(columns):
        if pd.isna(columns).sum():
            print(label)


# In[113]:


# filling all missing numeric valuw
for label, content in data_1.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            data_1[label] = content.fillna(content.median())


# In[114]:


data_1.isna().sum()


# # Data visualization
# * Create a function to visualize the Variables in relation to the Target column
# * Age in relation to Survived or not Survived
# * Gender in relation to SUrvived or not Survived 

# In[115]:


facet = sns.FacetGrid(data_2, hue ='Survived', aspect =4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim =(0, data_2['Age'].max()))
facet.add_legend()
plt.show()


# In[116]:


def bar_chart(feature):
    survived = data_2[data_2['Survived'] ==1][feature].value_counts()
    dead = data_2[data_2['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived','Dead']
    df.plot(kind ='bar',stacked =True, figsize =(10,5))


# In[117]:


bar_chart('Sex')


# In[118]:


bar_chart('Parch')


# In[119]:


fig, ax= plt.subplots(figsize =(8,5))
ax.scatter(data_1.Fare, data_1.Age)
ax.set(xlabel ='Pclass', ylabel ='Age');


# In[120]:


data_1.Fare.plot(kind ='hist');


# In[121]:


data_1.Pclass


# # creating a model
# * using RandomForestRegressor and XGBClassifier to create a model
# 

# In[122]:


x_train = data_1.drop('Survived',axis = 1)
y_train = data_1['Survived']
clf4 = RandomForestClassifier(n_jobs=-1, random_state =47)
clf4.fit(x_train,y_train)


# In[125]:


x_train = data_1.drop('Survived',axis = 1)
y_train = data_1['Survived']
clf5 = XGBClassifier(n_jobs=-1, random_state =47)
clf5.fit(x_train,y_train)


# In[124]:


x_train.info()


# # Importing the Training Data set
# * viewing the test data set
# * feature engineering
# * using the the test data set on the test set

# In[126]:


# Importing the test data
df_test = pd.read_csv('TIT_test.csv')


# In[127]:


df_test.head().T


# In[128]:


def preprocess_data(du):
    
    # Fill the numeric rows with median
    for label, content in du.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Fill missing numeric values with median
                du[label] = content.fillna(content.median())
    
        # Filled categorical missing data and turn categories into numbers
        if not pd.api.types.is_numeric_dtype(content):
            # We add +1 to the category code because pandas encodes missing categories as -1
            du[label] = pd.Categorical(content).codes+1
    
    return du


# In[51]:


test = preprocess_data(df_test)
test.head().T


# In[129]:


test_preds  = clf4.predict(test)


# In[130]:


test_preds


# In[55]:


submission = pd.DataFrame({'passengerId': df_test['PassengerId'],'survived':test_preds})


# In[57]:


submission


# In[65]:


submission.to_csv('submission.csv', index = False)


# In[131]:


preds2 = clf5.predict(test)


# In[132]:


xgb = pd.DataFrame({'passenger id': test['PassengerId'], 'survived': preds2})
xgb


# <img src = 'titanic.gif' style= 'width:500; height:400px'/>

# In[ ]:




