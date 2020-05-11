import sys
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize,precision=3)
pd.options.display.max_columns = None
from sklearn.metrics import accuracy_score as accuracy
import warnings
warnings.filterwarnings('ignore')

# Import Different Machine Learning Models and metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Initialize all the models
tree_classifier = DecisionTreeClassifier()
logistic_classifier = LogisticRegression()
svc_classifier = SVC(kernel= 'rbf')
nb_classifier = GaussianNB()
knn_classifier = KNeighborsClassifier(n_neighbors= 2)

# Create Utility Functions

def PrintAccuracy(classifier_name, y_test, y_pred):
    '''
    Calculates the accuracy of the classification model
    
    Parameters :
        classifier_name (string) : The name of the classifier algorithm
        y_test (series) : actual label
        y_pred (series) : predicted label
    '''
    print(f'Accuracy using {classifier_name} is :- ', (accuracy(y_test,y_pred)*100))
    
def ApplyCrossValidation(regressor, x_train, y_train):
    '''
    Applies 10-Fold Cross Validation and returns the average accuracy across all the validations
    
    Parameters:
        regressor (model) : ML Model
        x_train (series) : Indpendent parameters from training set
        y_train (series) : Dependent Label from training set
    '''
    accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)
    return accuracies.mean()*100

def get_classification_model(algo_name):
    '''
    Returns the model based on the input algorithm name
    
    Parameters :
        algo_name (string) : The name of algorithm, 'DT','LR','SVM','NB','KNN'
    '''
    if algo_name == 'DT':
        return tree_classifier
    elif algo_name == 'LR':
        return logistic_classifier
    elif algo_name == 'SVM':
        return svc_classifier
    elif algo_name == 'NB':
        return nb_classifier
    elif algo_name == 'KNN':
        return knn_classifier

def FitAndEvaluateMLModel(x_train, y_train, x_test, y_test):
    '''
    Fits the models in the training dataset and evaluates on the test dataset.
    
    Parameters :
        x_train (series) : Independent variables from Training Set
        y_train (series) : Dependent label from Training Set
        x_test (series) : Independent variables from Test Set
        y_test (series) : Dependent label from Test Set
    '''
    # Fit the decision tree classifier in the data set
    tree_classifier.fit(x_train,y_train)

    # Make predictions
    tree_pred = tree_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Decision Tree', y_pred= tree_pred, y_test= y_test)
    
    print(classification_report(y_test, tree_pred))
    
    logistic_classifier.fit(x_train,y_train)

    # Make predictions
    logistic_pred = logistic_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Logistic Regression', y_pred= logistic_pred, y_test= y_test)
    
    print(classification_report(y_test, logistic_pred))
    
    # Fit SVM into the dataset
    svc_classifier.fit(x_train, y_train)

    # Make predictions
    svc_pred = svc_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Support Vector Machine', y_pred= svc_pred, y_test= y_test)
    
    print(classification_report(y_test, svc_pred))
    
    # Fit Naive Bayes to the dataset

    nb_classifier.fit(x_train, y_train)

    # Make predictions
    nb_pred = nb_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'Naive Bayes', y_pred= nb_pred, y_test= y_test)
    
    print(classification_report(y_test, nb_pred))
    
    # Fit KNN to the dataset
    knn_classifier.fit(x_train, y_train)

    # Make predictions
    knn_pred = knn_classifier.predict(x_test)
    PrintAccuracy(classifier_name= 'KNN', y_pred= knn_pred, y_test= y_test)
    
    print(classification_report(y_test, knn_pred))
    
    # N- Fold Cross Validations for the different model used

    # 1. Decision Tree Classifier
    print(f'Average accuracy of Decision Tree Classifier after applying 10- Fold CV is {ApplyCrossValidation(tree_classifier, x_train, y_train)}')

    # 2. Logistic Regression
    print(f'Average accuracy of Logistic Regression after applying 10- Fold CV is {ApplyCrossValidation(logistic_classifier, x_train, y_train)}')
    
    # 3. SVM
    print(f'Average accuracy of SVM after applying 10- Fold CV is {ApplyCrossValidation(svc_classifier, x_train, y_train)}')

    # 4. Naive Bayes
    print(f'Average accuracy of Naive Bayes after applying 10- Fold CV is {ApplyCrossValidation(nb_classifier, x_train, y_train)}')

    # 5. KNN
    print(f'Average accuracy of KNN after applying 10- Fold CV is {ApplyCrossValidation(knn_classifier, x_train, y_train)}')

