from sklearn import datasets

# splitting data_set into training and tests

iris = datasets.load_iris()

X = iris.data
Y = iris.target

print(X.shape)
print(Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,teste_size=0.2)



