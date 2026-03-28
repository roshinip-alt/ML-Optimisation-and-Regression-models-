from Minimisation_Methods.Gradient_descent import graddes
import numpy as np

X=np.array([1,2,3,4])
Y=np.array([5,6,7,8])

iterations = 1000
alpha = 0.01
m,b=graddes(X,Y,iterations,alpha,0,0)

Y_pred=X*m+b
print(f"Best line of fit: y={m}x+{b}")

def Linear(X,Y):
    m,b=graddes(X,Y,0.01,0,0)
    return m,b

