#!/usr/bin/env python
# coding: utf-8

# # IMPORTING RELEVENT LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading the dataset

# In[3]:


data = pd.read_csv(r"C:\Users\Admin\Downloads\archive (1)\winequality-red.csv")
data.head(10)


# In[5]:


#Descriptive statistics of dataset
data.describe()


# In[6]:


set(data['quality'])


# In[7]:


#checking null values
data.isnull().sum()


# # corelation matrix among all feature inside the dataset

# In[14]:


plt.figure(figsize=(20,10))
corr= data.corr()
sns.heatmap(corr, annot=True)
plt.show()


# In[16]:


# Detection of Outliers Using Z score
from scipy import stats
z = np.abs(stats.zscore(data))
print(z)


# In[17]:


print(np.where(z>3))


# # Remove those records where (z>3)(outliers)

# In[18]:


new_data = data[(z < 3).all(axis=1)]


# In[20]:


new_data.shape


# In[21]:


data.shape


# # Data Spliting into Features(x) and Target(y)

# In[31]:


from sklearn.model_selection import train_test_split 

x = new_data.drop(columns= 'quality')
y = new_data['quality']


# In[32]:


x.head()


# In[33]:


y


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# # Data Modeling Using Random Forest Classifier

# In[35]:


from sklearn.ensemble import RandomForestClassifier

rf_classificationModel = RandomForestClassifier(n_estimators=100)
rf_classificationModel.fit(x_train,y_train)


# # Prediction of Data

# In[37]:


y_pred = rf_classificationModel.predict(x_test)
y_pred


# # Data Evaluation

# In[38]:


# Data Evaluation

from sklearn import metrics
print('Accuracy Score', metrics.accuracy_score(y_pred, y_test))


# In[39]:


#Accuracy score of 73%


# In[40]:


# Hyperparameter Tuning:

from sklearn import tree
tree.plot_tree(rf_classificationModel.estimators_[0], filled=True)

plt.figure(figsize=(20,20))


# In[41]:


plt.figure(figsize=(20,20))
for i in range(len(rf_classificationModel.estimators_)):
               tree.plot_tree(rf_classificationModel.estimators_[i], filled=True)


# In[ ]:




