from sklearn.datasets import load_iris

iris = load_iris()

type(iris) #sklearn.utils.Bunch

print(iris.data)

print (iris.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print (iris.target)

print (iris.target_names) #['setosa' 'versicolor' 'virginica']

type(iris.data) #numpy.ndarray
type(iris.target) #numpy.ndarray

print(iris.data.shape) #(150,4)
print(iris.target.shape) #(150,)