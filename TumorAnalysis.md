---
layout: default
---

[BACK](./)

## I. Background

1. To classify tumor is benign or malignant.

1. Data source: UCI

1. Variables
```
Sample code number: id number 
Clump Thickness: 1 - 10 
Uniformity of Cell Size: 1 - 10 
Uniformity of Cell Shape: 1 - 10 
Marginal Adhesion: 1 - 10 
Single Epithelial Cell Size: 1 - 10 
Bare Nuclei: 1 - 10 
Bland Chromatin: 1 - 10 
Normal Nucleoli: 1 - 10 
Mitoses: 1 - 10 
Class: (2 for benign, 4 for malignant)
```

1. Objects and measurement:
```
(1) To classify tumor is benign or malignant.
(2) The machine learning techniques utilized are:
Logistic Regression
K-Nearest Neighbours
Support Vector Machine
Kernel SVM
Naive Bayes
Decision Tree
Random Forests
XGBoost
```

## II. Data Processing

### 1. Importing the libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Importing the dataset
```
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
dataset.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 683 entries, 0 to 682
Data columns (total 11 columns):
 #   Column                       Non-Null Count  Dtype
---  ------                       --------------  -----
 0   Sample code number           683 non-null    int64
 1   Clump Thickness              683 non-null    int64
 2   Uniformity of Cell Size      683 non-null    int64
 3   Uniformity of Cell Shape     683 non-null    int64
 4   Marginal Adhesion            683 non-null    int64
 5   Single Epithelial Cell Size  683 non-null    int64
 6   Bare Nuclei                  683 non-null    int64
 7   Bland Chromatin              683 non-null    int64
 8   Normal Nucleoli              683 non-null    int64
 9   Mitoses                      683 non-null    int64
 10  Class                        683 non-null    int64
dtypes: int64(11)
memory usage: 58.8 KB
```
```
print(X[:5])
```
```
[[ 5  1  1  1  2  1  3  1  1]
 [ 5  4  4  5  7 10  3  2  1]
 [ 3  1  1  1  2  2  3  1  1]
 [ 6  8  8  1  3  4  3  7  1]
 [ 4  1  1  3  2  1  3  1  1]]
```

### 3. Splitting the dataset into the Training set and Test set
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
```
<!--
### 4. Feature Scaling
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 5. View data
```
```
-->
### 4. Define a Dataframe
```
df = pd.DataFrame(columns=['Classifier', 'CV_AUC'], dtype=float)
```

## III. Build predicting model
```
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
```

### 1. Logistic Regression

#### 1.1 Training the Logistic Regression model using k-fold cross-validation
```
from sklearn.linear_model import LogisticRegression

param_grid = { 
              'C': np.arange(0.1, 1, 0.1),
              'max_iter': np.arange(80, 100, 2),
              }
classifier = LogisticRegression()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.995315
best parameters: {'C': 0.6, 'max_iter': 80}
best estimator: LogisticRegression(C=0.6, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=80,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
```
#### 1.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
</table>

### 2. K nearest neighbors
#### 2.1 Train K-Nearest Neighbors using k-fold cross-validation
```
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors':  np.arange(1, 15, 1),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'leaf_size': np.arange(20, 50, 2),
             }
classifier = KNeighborsClassifier()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.992034
best parameters: {'algorithm': 'auto', 'leaf_size': 20, 'n_neighbors': 7, 'weights': 'distance'}
best estimator: KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=7, p=2,
                     weights='distance')
```
#### 2.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
   <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
</table>

### 3. Support Vector Machine
#### 3.1 Train Support Vector Machine using k-fold cross-validation
```
from sklearn.svm import SVC

param_grid = {'kernel':  ['linear', 'poly', 'rbf', 'sigmoid'],
             }
classifier = SVC()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.994657
best parameters: {'kernel': 'linear'}
best estimator: SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```
#### 3.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
</table>

### 4. Kernel SVM
#### 4.1 Train Kernel SVM using k-fold cross-validation
```
from sklearn.svm import SVC

param_grid = {'kernel':  ['rbf'],
             }
classifier = SVC()             
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.991537
best parameters: {'kernel': 'rbf'}
best estimator: SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
```
#### 4.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
  <tr>
    <td>Kernal SVM</td> <td>0.991537</td>
  </tr>
</table>

### 5. Naive Bayes
#### 5.1 Train Naive Bayes using k-fold cross-validation
```
from sklearn.naive_bayes import GaussianNB

param_grid = {
             }
classifier = GaussianNB()             
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.983938
best parameters: {}
best estimator: GaussianNB(priors=None, var_smoothing=1e-09)
```
#### 5.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
  <tr>
    <td>Kernal SVM</td> <td>0.991537</td>
  </tr>
  <tr>
    <td>Naive Bayes</td> <td>0.983938</td>
  </tr>
</table>

### 6. Decision Tree classification

#### 6.1 Train Decision Tree using k-fold cross-validation
```
from sklearn.tree import DecisionTreeClassifier

param_grid = {'criterion': ['entropy', 'gini'],
              'splitter': ['best', 'random'],
             }
classifier = DecisionTreeClassifier()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.950706
best parameters: {'criterion': 'entropy', 'splitter': 'random'}
best estimator: DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
```
#### 6.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
   <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
  <tr>
    <td>Kernal SVM</td> <td>0.991537</td>
  </tr>
  <tr>
    <td>Naive Bayes</td> <td>0.983938</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.950706</td>
  </tr>
</table>

### 7. Random Forest classification

#### 7.1 Train Random Forest using k-fold cross-validation
```
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': np.arange(50, 60, 2),
              'criterion': ['entropy', 'gini'],
              'max_depth': np.arange(10, 20, 5),
              'max_features': ['auto', 'sqrt', 'log2'],
              'class_weight': ['balanced', 'balanced_subsample'],
              'random_state': [0],
             }
classifier = RandomForestClassifier()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.989630
best parameters: {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto', 'n_estimators': 50, 'random_state': 0}
best estimator: RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                       class_weight='balanced_subsample', criterion='entropy',
                       max_depth=10, max_features='auto', max_leaf_nodes=None,
                       max_samples=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=50, n_jobs=None, oob_score=False,
                       random_state=0, verbose=0, warm_start=False)
```
#### 7.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
   <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
  <tr>
    <td>Kernal SVM</td> <td>0.991537</td>
  </tr>
  <tr>
    <td>Naive Bayes</td> <td>0.983938</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.950706</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.989630</td>
  </tr> 
</table>

### 8. XGBoost
#### 8.1 Train XGBoost using k-fold cross-validation
```
from xgboost import XGBClassifier

param_grid = {'booster': ['gbtree', 'gblinear', 'dart'],
             }
classifier = XGBClassifier()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print("best cross-validation score: {:.6f}".format(grid.best_score_))
print("best parameters:", grid.best_params_)
print("best estimator:", grid.best_estimator_)
```
```
best cross-validation score: 0.995046
best parameters: {'booster': 'gblinear'}
best estimator: XGBClassifier(base_score=0.5, booster='gblinear', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
```
#### 8.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.995315</td>
  </tr>
   <tr>
    <td>K-NN</td> <td>0.992034</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.994657</td>
  </tr>
  <tr>
    <td>Kernal SVM</td> <td>0.991537</td>
  </tr>
  <tr>
    <td>Naive Bayes</td> <td>0.983938</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.950706</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.989630</td>
  </tr> 
  <tr>
    <td>XGBoost</td> <td>0.995046</td>
  </tr> 
</table>

## IV. Compare and Apply
### 1. Compare models, choose the final model
We will choose the final model only based on the cross-validated AUC score. The Logistic Regression model had the highest AUC score of 0.995315, so Logistic Regression model is the final model.

### 2. Using the best parameters to make predictions
```
from sklearn.metrics import roc_auc_score

classifier = LogisticRegression(C=0.6, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=80,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```
```
print("y_test:\n", y_test)
print("y_pred:\n", y_pred)
```
```
y_test:
 [2 2 4 4 2 2 2 4 2 2 4 2 4 2 2 2 4 4 4 2 2 2 4 2 4 4 2 2 2 4 2 4 4 2 2 2 4
 4 2 4 2 2 2 2 2 2 2 4 2 2 4 2 4 2 2 2 4 2 2 4 2 2 2 2 2 2 2 2 4 4 2 2 2 2
 2 2 4 2 2 2 4 2 4 2 2 4 2 2 4 2 4 2 4 4 4 2 4 4 4 2 2 2 4]
y_pred:
 [2 2 4 4 2 2 2 4 2 2 4 2 4 2 2 2 4 4 4 2 2 2 4 2 4 4 2 2 2 4 2 4 4 2 2 2 4
 4 2 4 2 2 2 2 2 2 2 4 2 2 4 2 4 2 2 2 4 4 2 4 2 2 2 2 2 2 2 2 4 4 2 2 2 2
 2 2 4 2 2 2 4 2 4 2 2 4 2 4 4 2 4 2 4 4 2 4 4 4 4 2 2 2 4]
```
```
auc = roc_auc_score(y_test, y_pred)
print("test-set auc score: {:.6f}".format(auc))
```
```
test-set auc score: 0.963759
```
### 3. Conclusion
The AUC score for the Logistic Regression is 0.995315 on the training set. The AUC score on the test set is 0.963759.

<!-- Version 2.0
### 1. Compare models, choose the final model
We will choose the final model based on the AUC score. The K-NN model and Random Forest model had the highest AUC score of 0.973.
Version 2.0 -->

<!-- Version 1.0
### 1. Compare models, choose the final model
We will choose the final model only based on the cross-validated AUC score. The Logistic Regression model had the highest AUC score of 99.53%, so Logistic Regression model is the final model.

### 2. Hyperparameter Optimization
After we chose the Logistic Regression classifier as the final model to implement on our binary classification problem, we wished to optimize its performance. We implemented hyperparameter optimization by using grid search to find parameters to improve the performance of the Logistic Regression classifier.

```
from sklearn.model_selection import GridSearchCV
classifier = LogisticRegression(random_state = 0)

parameters = [{'C': [0.25, 0.5, 0.75, 1], 
               'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'multi_class':['auto', 'ovr', 'multinomial'],
               'max_iter':[35,40,50,60,70,90,100,110]}
              ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_auc = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best AUC: {:.2f} %".format(best_auc*100))
print("Best Parameters:", best_parameters)
```
```
Best AUC: 99.60 %
Best Parameters: {'C': 0.25, 'max_iter': 40, 'multi_class': 'auto', 'penalty': 'l1', 'solver': 'saga'}
```

### 3. Predict Test set
```
from sklearn.metrics import roc_auc_score

# Logistic Regression
classifier = LogisticRegression(random_state = 0, C = 0.25, max_iter = 40, multi_class='auto', penalty='l1', solver='saga')
classifier.fit(X_train, y_train)

probs = classifier.predict_proba(X_test)

# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate scores
auc = roc_auc_score(y_test, probs)
print("Area under curve: {:.2f} %".format(auc*100))
```
```
Area under curve: 99.45 %
```
### 4. Conclusion

After optimizing, the AUC score for the Logistic Regression classifier improved slightly from 99.54% to 99.60% on the training set. The AUC score on the test set was 99.45%.
Version 1.0 -->

References:

1. Predictive Models of Student College Commitment Decisions Using Machine Learning. [https://www.mdpi.com/2306-5729/4/2/65/htm](https://www.mdpi.com/2306-5729/4/2/65/htm)
2. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
3. Cross-Validation. [https://www.textbook.ds100.org/ch/15/bias_cv.html](https://www.textbook.ds100.org/ch/15/bias_cv.html)
4. Efficiently searching for optimal tuning parameters. [https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb](https://github.com/justmarkham/scikit-learn-videos/blob/master/08_grid_search.ipynb)
5. Cross Validation and Grid Search. [https://amueller.github.io/ml-training-intro/slides/03-cross-validation-grid-search.html#1](https://amueller.github.io/ml-training-intro/slides/03-cross-validation-grid-search.html#1)
6. roc_auc_score method and auc method. [https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc](https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc)
* * *
[BACK](./)
