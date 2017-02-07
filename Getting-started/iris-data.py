from sklearn import datasets
from sklearn import neighbors, datasets, preprocessing 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Load Data
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target
#Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33) 
#Data Preprocessing 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Modelling
#Model 1 - knn
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Model 2 - SVC
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

#Predict
y_pred = svc.predict(X_test)

# Accuracy
accuracy_score(y_test, y_pred)