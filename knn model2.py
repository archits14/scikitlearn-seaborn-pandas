from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

knn = KNeighborsClassifier(n_neighbors=5)

X = iris.data
y = iris.target

print(knn)

knn.fit(X,y)

print("The first result is:", knn.predict([[3, 5, 4, 2]]))

X_new = [[3,5,4,2],[5.1, 3.5, 1.4, 0.2]]
print("The first result is:", knn.predict(X_new))