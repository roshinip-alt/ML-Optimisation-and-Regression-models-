import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

pumpkins=pd.read_csv("pumpkins.csv")
price=(pumpkins['Low Price']+pumpkins['High Price'])/2
month = pd.DatetimeIndex(pumpkins['Date']).month
# degree 2 testing for fitting
X= month.values.reshape(-1,1)
Y= price
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
pipeline=make_pipeline(PolynomialFeatures(2),LinearRegression())



