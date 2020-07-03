#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#get dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =1)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :-1] = sc.fit_transform(X_train[:, :-1])
X_test[:, :-1] = sc.transform(X_test[:, :-1])

#training
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear' , random_state =0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))