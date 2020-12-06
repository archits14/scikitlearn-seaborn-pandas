from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()

logreg = LogisticRegression()

X = iris.data
y = iris.target

logreg.fit(X,y)

print("The first result is:", logreg.predict([[3, 5, 4, 2]]))

X_new = [[6.9, 3.2, 5.7, 2.3],[5, 4, 3, 2]]
print("The second result is:", logreg.predict(X_new))

y_pred = logreg.predict(X)
print ("The accuracy of the model is:", metrics.accuracy_score(y, y_pred))
