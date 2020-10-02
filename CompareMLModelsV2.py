# compare algorithms
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

array = dataset.values
X = array[:,0:4]
Y = array[:,4]
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
Y = label_encoder.fit_transform(Y)
# print(Y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.50, random_state=1, shuffle=True)

labels = np.concatenate((Y_validation, Y_train))

names = []
results = []
models = [
          ("NB",    GaussianNB(), GaussianNB()),
          ("LN",    LinearRegression(), LinearRegression()),
          ("P2",    make_pipeline(PolynomialFeatures(degree=2),LinearRegression()), make_pipeline(PolynomialFeatures(degree=2),LinearRegression())),
          ("P3",    make_pipeline(PolynomialFeatures(degree=3),LinearRegression()), make_pipeline(PolynomialFeatures(degree=3),LinearRegression())),
          ("KNN",   KNeighborsClassifier(), KNeighborsClassifier()),
          ("LDA",   LinearDiscriminantAnalysis(), LinearDiscriminantAnalysis()),
          ("QDA",   QuadraticDiscriminantAnalysis(), QuadraticDiscriminantAnalysis()),
          ("SVM",   LinearSVC(), LinearSVC()),
          ("DTree", DecisionTreeClassifier(),DecisionTreeClassifier()),
          ("RF",    RandomForestClassifier(), RandomForestClassifier()),
          ("ETree", ExtraTreesClassifier(), ExtraTreesClassifier()),
          ("NN",    MLPClassifier(), MLPClassifier())
         ]

for name, model1, model2 in models:
	# kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    model1.fit(X_train, Y_train)
    model2.fit(X_validation, Y_validation)
    predictions1 = model1.predict(X_validation)
    predictions1 = predictions1.astype(int)
    for x in range(75):
        if predictions1[x] > 2:
            predictions1[x] = 2
    predictions2 = model2.predict(X_train)
    predictions2 = predictions2.astype(int)
    for x in range(75):
        if predictions2[x] > 2:
            predictions2[x] = 2
    print("")
    print(name)
    predictions = np.concatenate((predictions1, predictions2))
    print(confusion_matrix(labels, predictions))
    print(accuracy_score(labels, predictions))
    print(classification_report(labels,predictions))
    # print("     +     ")
    # print(confusion_matrix(Y_train, predictions2))
    # print("     ||     ")
    # print(confusion_matrix(Y_train, predictions2) + confusion_matrix(Y_validation, predictions1))
