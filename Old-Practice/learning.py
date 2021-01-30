import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge

np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
print(X)
print(y)
indices = np.arange(y.shape[0])
print(indices)
np.random.shuffle(indices)
print(indices)
X, y = X[indices], y[indices]

train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha", np.logspace(-7, 3, 5))
print(train_scores)
print(valid_scores)

