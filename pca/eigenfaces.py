"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  .. _LFW: http://vis-www.cs.umass.edu/lfw/

  original source: http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html

"""



print __doc__

from time import time
import logging
import pylab as pl
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
## min_faces_per_person: The extracted dataset will only retain pictures of people that have at least min_faces_per_person different pictures.
## resize: Ratio used to resize the each face picture.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
## images: numpy array of shape (13233, 62, 47) - 13233 samples of 62 x 47 pixels
##    Each row is a face image corresponding to one of the 5749 people in the dataset. 
##    Changing the slice_ or resize parameters will change the shape of the output.
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

# for machine learning we use the data directly (as relative pixel
# position info is ignored by this model)
## data: numpy array of shape (13233, 2914)
##    Each row corresponds to a ravelled face image of original size 62 x 47 pixels. 
##    Changing the slice_ or resize parameters will change the shape of the output.
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
## target: numpy array of shape (13233,)
##    Labels associated to each face image. Those labels range from 0-5748 and correspond to the person IDs.
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print "Total dataset size:"
print "n_samples: %d" % n_samples
print "n_features: %d" % n_features
print "n_classes: %d" % n_classes


###############################################################################
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150  # tried with the 10, 15, 25, 50, 100, 250

print "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print "done in %0.3fs" % (time() - t0)

print "Giving the amount of variance explained by each of the selected components"
print pca.explained_variance_ratio_

eigenfaces = pca.components_.reshape((n_components, h, w))

print "Projecting the input data on the eigenfaces orthonormal basis"
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print "done in %0.3fs" % (time() - t0)


###############################################################################
# Train a SVM classification model

print "Fitting the classifier to the training set"
t0 = time()
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
## The GridSearchCV implements the usual estimator API: when fitting it on a dataset 
##    all the possible combinations of parameter values are evaluated and the best combination is retained.
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print "done in %0.3fs" % (time() - t0)
print "Best estimator found by grid search:"
print clf.best_estimator_


###############################################################################
# Quantitative evaluation of the model quality on the test set

print "Predicting the people names on the testing set"
t0 = time()
y_pred = clf.predict(X_test_pca)
print "done in %0.3fs" % (time() - t0)

## target_names: optional display names matching the labels (same order)
## Text summary of the precision, recall, F1 score for each class.
## The reported averages include 
##    macro average (averaging the unweighted mean per label), 
##    weighted average (averaging the support-weighted mean per label), and 
##    sample average (only for multilabel classification). 
##    Micro average (averaging the total true positives, false negatives and false positives) 
##      is only shown for multi-label or multi-class with a subset of classes, because it corresponds to accuracy otherwise.
## Note that in binary classification, recall of the positive class is also known as 'sensitivity'; recall of the negative class is 'specificity'.
print classification_report(y_test, y_pred, target_names=target_names)
## labels: list of labels to index the matrix. 
##    This may be used to reorder or select a subset of labels. 
##    If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
## Confusion matrix to evaluate the accuracy of a classification. 
## By definition a confusion matrix C is such that C_ij is equal to the number of observations known to be in group i and predicted to be in group j.
## In binary classifiction: 
##    count of true negatives: C_00
##    count of false negatives: C_10
##    count of true positives: C_11
##    count of false positives: C_01
print confusion_matrix(y_test, y_pred, labels=range(n_classes))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        pl.subplot(n_row, n_col, i + 1)
        pl.imshow(images[i].reshape((h, w)), cmap=pl.cm.gray)
        pl.title(titles[i], size=12)
        pl.xticks(())
        pl.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

pl.show()
