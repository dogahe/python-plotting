from sklearn import linear_model
import matplotlib.pyplot as plot
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

reg = linear_model.LinearRegression(fit_intercept=True)
#reg.fit ([[0], [2], [4], [6]], [0, 1, 2, 2.9])
#reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
X_train = [[0], [2], [4], [6], [8.5]]
y_train = [1, 2, 3, 4, 6]
print(reg.fit(X_train, y_train))
print(reg.coef_)
print(reg.intercept_)

y_predict = reg.predict(X_train)
print(y_predict)
print('Variance score: %.2f' % r2_score(y_train, y_predict))

print("Here")
a = np.array([1, 2, 3, 4])
print(a.shape)
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(b.shape)

plot.scatter(X_train, y_train,  color='black')
plot.plot(X_train, y_predict, color='blue', linewidth=3)
plot.show()


