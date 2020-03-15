#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_set = pd.read_csv(r"D:\MSc\Semester 1\CS5612 - Pattern Recognition\Assignment 3\WBCD\breast-cancer-wisconsin.data_MISSING_DATA_ROWS_REMOVEDx.csv")


# In[3]:


data_clean = data_set.dropna()


# In[4]:


print(data_clean.describe())


# In[5]:


X = data_clean.drop('target', axis=1)
y = data_clean['target']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)


# In[7]:


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)


# In[8]:


y_pred = svclassifier.predict(X_test)


# In[9]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

