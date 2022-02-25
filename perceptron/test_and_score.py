from sklearn import datasets
from sklearn.model_selection import train_test_split
import perceptron

iris_data = datasets.load_iris()
X = iris_data.data[:100]
y = iris_data.target[:100]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y) 

perc = perceptron.Perceptron()
perc.fit(X_train, y_train)

print(perc.score_compute(X_test, y_test))
