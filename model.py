import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('C:/Users/Windows/Downloads/heart.csv')

data = data.drop_duplicates()

X = data.drop(columns='target', axis=1)
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

log = LogisticRegression()
log.fit(X_train, Y_train)

# print(log.predict([[37,1,2,130,250,0,1,187,0,3.5,0,0,2]]))

import pickle
pickle.dump(log,open('model.pkl','wb'))

