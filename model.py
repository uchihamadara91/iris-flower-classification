# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

#-------------------------------------------------------
dataset = pd.read_csv('IRIS.csv')

print(dataset.columns)

print(dataset.head())

X = dataset[['sepal.length','sepal.width','petal.length','petal.width']]



y = dataset[['variety']]
print(y)
#print(type(y))

y = np.array(y)
y = y.ravel()

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#print(y.shape)
#print(type(y))
print(y)



#Since we have a very small dataset, we will train our model with all availabe data.

lr = LogisticRegression(max_iter=200)

#Fitting model with trainig data
lr.fit(X, y)


# Saving model to disk
pickle.dump(lr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2.5, 3.2, 3.5, 3.7]]))
