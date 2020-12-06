from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

iris = load_iris()

knn = KNeighborsClassifier(n_neighbors=5)

X = iris.data
y = iris.target

print(knn)

knn.fit(X,y)

print("The first result is:", knn.predict([[3, 5, 4, 2]]))

X_new = [[3,5,4,2],[5.1, 3.5, 1.4, 0.2]]
print("The second result is:", knn.predict(X_new))

y_pred = knn.predict(X)
print ("The accuracy of the model is:", metrics.accuracy_score(y, y_pred))

print("AFTER TRAIN TEST SPLIT:")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("The accuracy after train test split is:", metrics.accuracy_score(y_test, y_pred))