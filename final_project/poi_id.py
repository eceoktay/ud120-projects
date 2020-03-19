#!/usr/bin/python

import sys
import pickle
import math
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments','total_payments', 
'loan_advances','bonus','restricted_stock_deferred','deferred_income', 
'total_stock_value','expenses','exercised_stock_options','other', 
'long_term_incentive','restricted_stock','director_fees','to_messages',
'from_poi_to_this_person','from_messages','from_this_person_to_poi',
'shared_receipt_with_poi','from_poi_to_this_person_percentage','from_this_person_to_poi_percentage'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
for person in data_dict.keys():
    if data_dict[person]['to_messages'] != 0:
        data_dict[person]['from_poi_to_this_person_percentage'] = float(data_dict[person]['from_poi_to_this_person']) / float(data_dict[person]['to_messages'])
    else:
        data_dict[person]['from_poi_to_this_person_percentage'] = 0.0
    if math.isnan(data_dict[person]['from_poi_to_this_person_percentage']):
        data_dict[person]['from_poi_to_this_person_percentage'] = 0
        
    if data_dict[person]['from_messages'] != 0:
        data_dict[person]['from_this_person_to_poi_percentage'] = float(data_dict[person]['from_this_person_to_poi']) / float(data_dict[person]['from_messages'])
    else:
        data_dict[person]['from_this_person_to_poi_percentage'] = 0.0
    if math.isnan(data_dict[person]['from_this_person_to_poi_percentage']):
        data_dict[person]['from_this_person_to_poi_percentage'] = 0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clfNB = GaussianNB()
from sklearn.svm import SVC
clfSVM = SVC()
from sklearn.tree import DecisionTreeClassifier
clfDT = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
target_names = ['Not PoI', 'PoI']

### Applying feature selection with span of all possible number of features
### Using naive bayes classifier with default parameters
for k_value in range(1, len(features_list)):
    pipeNB = Pipeline([('featureSelection', SelectKBest(f_classif, k = k_value)),
                ('classification', clfNB)])
    pipeNB.fit(features_train, labels_train)
    predNB = pipeNB.predict(features_test)
    ### Generating classification report
    print("NB with ", k_value, " features: ")
    print(classification_report(labels_test, predNB, target_names = target_names))

### Applying feature selection with span of all possible number of features
### Using SVM classifier incorporating GridSearchCV to find best parameters
for k_value in range(1, len(features_list)):
    param_grid = {
        'C': [100, 500, 1000, 5000, 10000],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        }
    clfSearchSVM = GridSearchCV(clfSVM, param_grid)
    pipeSVM = Pipeline([('featureSelection', SelectKBest(f_classif, k = k_value)),
                ('classification', clfSearchSVM)])
    pipeSVM.fit(features_train, labels_train)
    predSVM = pipeSVM.predict(features_test)
    print("SVM with ", k_value, " features: ", clfSearchSVM.best_params_)
    print(classification_report(labels_test, predSVM, target_names = target_names))

### Applying feature selection with span of all possible number of features
### Using decision tree classifier incorporating GridSearchCV to find best parameters
for k_value in range(1, len(features_list)):
    param_grid = {
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    clfSearchDT = GridSearchCV(clfDT, param_grid)
    pipeDT = Pipeline([('featureSelection', SelectKBest(f_classif, k = k_value)),
                ('classification', clfSearchDT)])
    pipeDT.fit(features_train, labels_train)
    predDT = pipeDT.predict(features_test)
    print("DT with ", k_value, " features: ", clfSearchDT.best_params_)
    print(classification_report(labels_test, predDT, target_names = target_names))

"""
### Examination by plot
import matplotlib.pyplot as plt
colors = ["r", "g"]
for i in range(len(labels)):
    f1 = features[i][0]
    f2 = features[i][1]
    plt.scatter(f1, f2, color=colors[(int)(labels[i])])
plt.show()
"""

### The best classifier
clf = Pipeline([('featureSelection', SelectKBest(f_classif, k = 6)),
                ('classification', clfNB)])
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("Selected Classifier: ", "GaussianNB")
### Listing selected features
selected_features = clf.named_steps.featureSelection.get_support()
selected_features_list = []
for i in range(len(selected_features)):
    if (selected_features[i]):
        selected_features_list.append(features_list[i])
features_list = selected_features_list
print("Selected Features: ", features_list)
### Generating classification report of the best classfier
print("Classification Report: ")
print(classification_report(labels_test, pred, target_names = target_names))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)