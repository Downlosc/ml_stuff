from sklearn import datasets
from sklearn.model_selection import train_test_split
from adaline import adaline
import matplotlib.pyplot as plt 
import numpy as np

iris_data = datasets.load_iris()
X = iris_data.data[:100]
y = iris_data.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)

ada = adaline.Adaline(X_train, y_train)
ada.fit()
score = ada.score(X_test, y_test)
print(score)
