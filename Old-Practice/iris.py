from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm

iris = datasets.load_iris()
print(iris.data.shape, iris.target.shape)
print(iris)
print(iris.data)
print(iris.target)
print(iris.feature_names)
print(iris.target_names)
print(iris.DESCR)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(y_test)

clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf)
score = clf.score(X_test, y_test)
print(score)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
print(scores)
      
      
