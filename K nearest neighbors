import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('C:/Users/ADMIN/Downloads/iris.csv')
y=df.iloc[:,-1].values  
x=df.iloc[:,0:4].values  
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)
#print(x_train)
#print(x_test)

 
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=12)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix


c=confusion_matrix(y_test,y_pred)  
print(c)
