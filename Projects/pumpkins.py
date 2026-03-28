import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

pumpkins = pd.read_csv("../Regression_Models/Dataset/pumpkins.csv")
price=(pumpkins['Low Price']+pumpkins['High Price'])/2
month = pd.DatetimeIndex(pumpkins['Date']).month
# degree 2 testing for fitting
X= month.values.reshape(-1,1)
Y= price
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
pipeline=make_pipeline(PolynomialFeatures(2),LinearRegression())
#linear regression here finds the optimal values ofthe decisionv

pipeline.fit(X_train,Y_train)

Y_pred= pipeline.predict(X_test)

plt.scatter(X_train,Y_train,color='blue')
x_curve=np.linspace(X.min(),X.max(),200).reshape(-1,1)
y_curve=pipeline.predict(x_curve)

plt.plot(x_curve,y_curve)
plt.show()

