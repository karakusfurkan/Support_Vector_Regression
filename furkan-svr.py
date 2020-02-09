#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 02:40:01 2020

@author: furkan
"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#%%
dataset
#%%
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
#%%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1,1)
y = sc_y.fit_transform(y)


#%%
#fittin SVR to dataset
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf') #its default but you know xd
regressor.fit(X, y)

#%%
# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
print(y_pred)
#need to look at again
#%%
# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#%%
