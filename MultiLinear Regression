import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
df = pd.read_csv("C:/Users/ADMIN/Downloads/FuelConsumption.csv") 

df.head()
cdf  =  df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']] 
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS) 

plt.xlabel("ENGINESIZE") 

plt.ylabel("CO2EMISSIONS") 

plt.show()

X = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]
Y = df[['CO2EMISSIONS']]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.20,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
y_pred= regressor.predict(X_test)
from sklearn.metrics import mean_squared_error
print("Mean square error "mean_squared_error(Y_test,y_pred))
print("Regreesor score , accuracy",regressor.score(X_test,Y_test))
print('Coefficients of the model:', regressor.coef_) 
print('intercept of the model:', regressor.intercept_)

