#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[7]:



import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading .csv dataset
fnd_data=pd.read_csv("C:/Users/konda/Downloads/Data_Science_DataSets/news.csv")

#printing rows and columns
fnd_data.shape

#printing first 5 rows
fnd_data.head(5)


# In[8]:


x=fnd_data["text"]
y=fnd_data["label"]


# # Splitting data into training and testing sets

# In[9]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=7)


# # converting raw documents into matrix of TF_IDF features

# In[11]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[12]:


# Fit and transform train set, transform test set

# TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.

tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# # fitting data into PassiveAggressiveClassifier

# In[13]:


#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)


# # making predictions on the test data

# In[14]:


#DataPredict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)


# # calculating the accuracy

# In[16]:


score=accuracy_score(y_test,y_pred)
print(round(score*100,2))


# # printing confusion matrix

# In[20]:


confusion_matrix(y_test,y_pred)


# In[ ]:




