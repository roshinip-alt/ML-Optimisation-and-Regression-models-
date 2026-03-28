import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error


full_pumpkins = pd.read_csv("../Regression_Models/Dataset/pumpkins.csv")

coloumn_to_select=['City Name','Package','Variety','Origin','Item Size','Color']

pumpkins=full_pumpkins.loc[:,coloumn_to_select]

pumpkins.dropna(inplace=True)

X=pumpkins[['City Name','Package','Variety','Origin','Item Size']]
Y=pumpkins['Color']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)

categorical_features=['City Name','Package','Variety','Origin','Item Size']
# create an ecnoder now
encoder= ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_features)
                   ]
    )
# now use the pipleine, for the main logisitic: 0 or 1 on linear
pipeline=Pipeline([
    ('encoder',encoder),
    ('Logistic',LogisticRegression(max_iter=1000))

])

pipeline.fit(X_train,Y_train)

Y_pred=pipeline.predict(X_test)

score =pipeline.score(X_test,Y_test)

print(score)