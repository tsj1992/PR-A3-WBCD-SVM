#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data_set = pd.read_csv("breast-cancer-wisconsin.data_MISSING_DATA_ROWS_REMOVEDx.csv")


# In[11]:


data_clean = data_set.dropna()


# In[12]:


print(data_clean.describe())


# In[13]:


X = data_clean.drop('target', axis=1)
y = data_clean['target']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)
print(X_train)


# In[15]:


svclassifier = SVC(kernel='linear')
print(svclassifier)
svclassifier.fit(X_train, y_train)


# In[16]:


y_pred = svclassifier.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

