import numpy as np
import pandas as pd

def polynomial_regression_grad(X,Y,iterations,alpha):
    theta0=0
    theta1=0
    theta2=0
    
    n=len(X)
    for _ in range(iterations):
        Y_pred= theta0 + theta1*X + theta2*(X**2)
        error=Y_pred-Y
        d_theta0= (1/n) *np.sum(error)
        d_theta1= (1/n)*np.sum(error*X)
        d_theta2 = (1/n)*np.sum(error*(X**2))

        theta0 = theta0 - alpha*d_theta0
        theta1= theta1 - alpha*d_theta1
        theta2 = theta2 - alpha*d_theta2

    return theta0,theta1,theta2




