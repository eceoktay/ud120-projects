#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



###
### Modification #1: A Smaller Training Set
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
###

#########################################################
### your code goes here ###
from sklearn.svm import SVC
###
### Modification #0
#clf = SVC(kernel="linear")
### Modification #2
#clf = SVC(kernel="rbf")
### Modification #3
#clf = SVC(kernel="rbf", C=10.0)
#clf = SVC(kernel="rbf", C=100.0)
#clf = SVC(kernel="rbf", C=1000.0)
clf = SVC(kernel="rbf", C=10000.0)
###

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print "Accuracy: "
print accuracy

###
### Modification #4
#print "Predictions: "
#print "10: ", pred[10]
#print "26: ", pred[26]
#print "50: ", pred[50]
###

###
### Modification #5
count = 0
for x in pred:
    if x == 1:
        count = count + 1
print "Chris(1): ", count
###
#########################################################


