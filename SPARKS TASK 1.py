#!/usr/bin/env python
# coding: utf-8

# GRIP @ The Sparks Foundation
# 
# Task 1 : Prediction using Supervised Machine Learning
# 
# SUBMITTED BY : GAYATHRI N

# In[2]:


# Importing the required libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Reading the data from source

# In[3]:


# Dataset path
URL = "http://bit.ly/w-data"
data = pd.read_csv(URL)
print("Data imported successfully")
data.head()


# In[4]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('DISTRIBUTION OF SCORES')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[8]:


print(data.shape)


# In[7]:


data.dtypes


# In[6]:


data.describe()


# In[8]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# Prepare The Data For Machine Learning Algorithm

# In[9]:


#Splitting the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=0)


# In[10]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[14]:


m = regression_model.coef_
c = regression_model.intercept_
regression_line = m * X + c
plt.plot(X, regression_line, color='black')

plt.scatter(X, y)

plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.show()


# In[15]:


print("Training Accuracy:", regression_model.score(X_train, y_train))
print("Test Accuracy:", regression_model.score(X_test, y_test))

print(X_test)

y_predicted=regression_model.predict(X_test)
comp = pd.DataFrame({'Actual': y_test, 'Predicted': y_predicted})
comp


# In[19]:


# Plotting the Bar graph to depict the difference between the actual and predicted value

comp.plot(kind='bar',figsize=(7,7))
plt.grid(which='major', linewidth='0.5', color='blue')
plt.grid(which='minor', linewidth='0.5', color='green')
plt.show()


# In[23]:


hrs = 9.25
pred_score = regression_model.predict([[hrs]])

print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred_score[0]))


# Evaluating the Model

# In[26]:


from sklearn import metrics  

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_predicted)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
print('R-2:', metrics.r2_score(y_test, y_predicted))


# In[ ]:




