import pandas as pd
#import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# loading data 
data = pd.read_csv('iris.csv')
# changing the column names for convenience
data.columns = list(map(str.lower, data.columns.values.tolist()))
X_train, X_test, Y_train, Y_test = train_test_split(
        data.loc[:, ['petal_length','petal_width']], data.loc[:, 'species'])

knn = KNeighborsClassifier(n_neighbors=2, p=2, metric='minkowski')
knn.fit(X_train, Y_train)
# write training accuracy and test accuracy to file
file = open("trainingresults.txt","w") 
file.write("Training Accuracy {}".format(knn.score(X_train, Y_train)))
file.write("Testing Accuracy {}".format(knn.score(X_test, Y_test)))
