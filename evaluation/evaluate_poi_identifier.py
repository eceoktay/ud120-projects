#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier() 
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
prec = precision_score(labels_test, predictions)
rec = recall_score(labels_test, predictions)

print("Precision: ", prec)
print("Recall: ", rec)

count_positive = 0
count_negative = 0
count_prediction_positive = 0
count_prediction_negative = 0
count_true_positive = 0
count_true_negative = 0
count_false_positive = 0
count_false_negative = 0
for i in range(len(labels_test)):
    if labels_test[i] == 0:
        count_negative = count_negative + 1
    elif labels_test[i] == 1:
        count_positive = count_positive + 1
    if predictions[i] == 0:
        count_prediction_negative = count_prediction_negative + 1
    elif predictions[i] == 1:
        count_prediction_positive = count_prediction_positive + 1
    if labels_test[i] == 1 and predictions[i] == 1:
        count_true_positive = count_true_positive + 1
    elif labels_test[i] == 0 and predictions[i] == 0:
        count_true_negative = count_true_negative + 1
    elif labels_test[i] == 1 and predictions[i] == 0:
        count_false_positive = count_false_positive + 1
    elif labels_test[i] == 0 and predictions[i] == 1:
        count_false_negative = count_false_negative + 1
print("Negatives in test set: ", count_negative);
print("Positives in predictions: ", count_positive);
print("Negatives in test set: ", count_prediction_negative);
print("Positives in predictions: ", count_prediction_positive);
print("True Positives: ", count_true_positive);
print("False Positives: ", count_false_positive);
print("True Negatives: ", count_true_negative);
print("False Negatives: ", count_false_negative);


