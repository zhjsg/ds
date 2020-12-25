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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
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
df = pd.DataFrame(columns=['Classifier', 'CV_Accuracy', 'CV_AUC'], dtype=float)
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

param_grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
              }
classifier = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.996
best parameters: {'penalty': 'l2'}
test-set accuracy score: 0.956
test-set roc_auc score: 0.953
```
#### 1.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
</table>

### 2. K nearest neighbors
#### 2.1 Train K-Nearest Neighbors using k-fold cross-validation
```
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors':  np.arange(1, 15, 1),
             }
classifier = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.992
best parameters: {'n_neighbors': 5}
test-set accuracy score: 0.971
test-set roc_auc score: 0.973
```
#### 2.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
</table>

### 3. Support Vector Machine
#### 3.1 Train Support Vector Machine using k-fold cross-validation
```
from sklearn.svm import SVC

param_grid = {'kernel':  ['linear', 'poly', 'rbf', 'sigmoid'],
             }
classifier = GridSearchCV(SVC(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.996
best parameters: {'kernel': 'linear'}
test-set accuracy score: 0.956
test-set roc_auc score: 0.957
```
#### 3.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
</table>

### 4. Kernel SVM
#### 4.1 Train Kernel SVM using k-fold cross-validation
```
from sklearn.svm import SVC

param_grid = {'kernel':  ['rbf'],
             }
classifier = GridSearchCV(SVC(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.991
best parameters: {'kernel': 'rbf'}
test-set accuracy score: 0.964
test-set roc_auc score: 0.967
```
#### 4.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.964</td> <td>0.967</td>
  </tr>
</table>

### 5. Naive Bayes
#### 5.1 Train Naive Bayes using k-fold cross-validation
```
from sklearn.naive_bayes import GaussianNB

param_grid = {
             }
classifier = GridSearchCV(GaussianNB(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.984
best parameters: {}
test-set accuracy score: 0.949
test-set roc_auc score: 0.960
```
#### 5.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.964</td> <td>0.967</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.949</td> <td>0.960</td>
  </tr>
</table>

### 6. Decision Tree classification

#### 6.1 Train Decision Tree using k-fold cross-validation
```
from sklearn.tree import DecisionTreeClassifier

param_grid = {'criterion': ['entropy', 'gini'],
              'splitter': ['best', 'random'],
             }
classifier = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.945
best parameters: {'criterion': 'entropy', 'splitter': 'best'}
test-set accuracy score: 0.964
test-set roc_auc score: 0.959
```
#### 6.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.964</td> <td>0.967</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.949</td> <td>0.960</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.964</td> <td>0.959</td>
  </tr>
</table>

### 7. Random Forest classification

#### 7.1 Train Random Forest using k-fold cross-validation
```
from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': np.arange(10, 110, 5),
              'criterion': ['entropy', 'gini'],
             }
classifier = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.993
best parameters: {'criterion': 'entropy', 'n_estimators': 60}
test-set accuracy score: 0.971
test-set roc_auc score: 0.973
```
#### 7.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.964</td> <td>0.967</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.949</td> <td>0.960</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.964</td> <td>0.959</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.971</td> <td>0.973</td>
  </tr>
</table>

### 8. XGBoost
#### 8.1 Train XGBoost using k-fold cross-validation
```
from xgboost import XGBClassifier

param_grid = {'booster': ['gbtree', 'gblinear', 'dart'],
             }
classifier = GridSearchCV(XGBClassifier(), param_grid=param_grid, cv=10, scoring='roc_auc')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("best cross-validation score: {:.3f}".format(classifier.best_score_))
print("best parameters:", classifier.best_params_)
print("test-set accuracy score: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("test-set roc_auc score: {:.3f}".format(roc_auc_score(y_test, y_pred)))
```
```
best cross-validation score: 0.996
best parameters: {'booster': 'gblinear'}
test-set accuracy score: 0.934
test-set roc_auc score: 0.923
```
#### 8.2 Results
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.956</td> <td>0.953</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.956</td> <td>0.957</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.964</td> <td>0.967</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.949</td> <td>0.960</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.964</td> <td>0.959</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.971</td> <td>0.973</td>
  </tr>
  <tr>
    <td>XGBoost</td> <td>0.934</td> <td>0.923</td>
  </tr>
</table>

## IV. Compare and Apply
### 1. Compare models, choose the final model
We will choose the final model based on the AUC score. The K-NN model and Random Forest model had the highest AUC score of 0.973.

<!--
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
-->

References:

1. Predictive Models of Student College Commitment Decisions Using Machine Learning. [https://www.mdpi.com/2306-5729/4/2/65/htm](https://www.mdpi.com/2306-5729/4/2/65/htm)
2. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)

* * *
[BACK](./)
