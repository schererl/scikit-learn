## support vector machine ##
'''
1- Effective hight dimensional spaces
2- Many Kernel function
3- used for classification and regression

It can create binary and multi-class classification on a dataset.

                   *   Mathematica Formulation  *

It creates a hyper-plane or a set of hyper-planes in a higher dimensional space for classification (splitting the data by its lines)

The line(s) drawn creates the largest margin as possible in both sides of the line.
This margin is made by the closest data points (in both sides of the line). 


- also called functional margin and the training data points are "support vectors"
- the larger the margin the lower the generalization error of the classifier.

The line can be not only a straight line, it could be curve or with any other
function aspect. (linear, polinomial, sigmoide...)

When the data overlaps two or more groups it can create a new dimenson hyperplan to diferenciate it.

'''

from sklearn import svm

# for linear SVC take as input two arrays x of shape (samples, features) holding training samples
X = [[0,0], [1,1]]

# also an array y of class labels (string or int), of shape (samples)
y = [0,1]

clf = svm.SVC()
clf.fit(X,y)

# after being fited the model can be used to predict new values:

clf.predict([[2.,2.]])

# use properties of the support vector
sv1=clf.support_vectors_ #get support vectors
sv2=clf.support_         # get indices of support vectors
sv3=clf.n_support_       # get number of support vectors for each class

from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target


classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = svm.SVC()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print("predictions: ", predictions)
print("actual:      ", y_test)
print("accuracy: ", acc)
