---
layout: default
---

[BACK](./)

## I. Background

1. To classify tumor is benign or malignant.

1. Data source: UCI

1. Variables
```
1. Sample code number: id number 
2. Clump Thickness: 1 - 10 
3. Uniformity of Cell Size: 1 - 10 
4. Uniformity of Cell Shape: 1 - 10 
5. Marginal Adhesion: 1 - 10 
6. Single Epithelial Cell Size: 1 - 10 
7. Bare Nuclei: 1 - 10 
8. Bland Chromatin: 1 - 10 
9. Normal Nucleoli: 1 - 10 
10. Mitoses: 1 - 10 
11. Class: (2 for benign, 4 for malignant)
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

## II. Data Analysis

### 1. Importing the libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Importing the dataset
```
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### 3. Splitting the dataset into the Training set and Test set
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### 4. Feature Scaling
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
<!--
### 5. View data
```
```
-->
### 5. Define a Dataframe
```
df = pd.DataFrame(columns=['Classifier', 'CV_Accuracy', 'CV_AUC'], dtype=float)
```

## III. Build predicting model

### 1. Logistic Regression

#### 1.1 Training the Logistic Regression model using k-fold cross-validation
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#Classifier
classifier = LogisticRegression(random_state = 0)

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[0] = ['Logistic Regression', accuracies.mean(), auc.mean()]
```
#### 1.2 Results
```
Classifier	CV_Accuracy	CV_AUC
0	Logistic Regression	0.967003	0.995386

```
### 2. K nearest neighbors
#### 2.1 Train K-Nearest Neighbors using k-fold cross-validation
```
from sklearn.neighbors import KNeighborsClassifier

#Classifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[1] = ['K-NN', accuracies.mean(), auc.mean()]
```
#### 2.2 Results
```

```

### 3. Support Vector Machine
#### 3.1 Train Support Vector Machine using k-fold cross-validation
```
from sklearn.svm import SVC

#Classifier
classifier = SVC(kernel = 'linear', random_state = 0)

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[2] = ['SVM', accuracies.mean(), auc.mean()]
```

#### 3.2 Results
```

```
### 4. Kernel SVM
#### 4.1 Train Kernel SVM using k-fold cross-validation
```
from sklearn.svm import SVC

#Classifier
classifier = SVC(kernel = 'rbf', random_state = 0

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[3] = ['Kernel SVM', accuracies.mean(), auc.mean()]
```

#### 4.2 Results
```

```

### 5. Naive Bayes
#### 5.1 Train Naive Bayes using k-fold cross-validation
```
from sklearn.naive_bayes import GaussianNB

#Classifier
classifier = GaussianNB()

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[4] = ['Naive Bayer', accuracies.mean(), auc.mean()]
```

#### 5.2 Results
```

```

### 6. Decision Tree classification

#### 6.1 Train Decision Tree using k-fold cross-validation
```
from sklearn.tree import DecisionTreeClassifier

#Classifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[5] = ['Decision Tree', accuracies.mean(), auc.mean()]
```

#### 6.2 Results
```

```

### 7. Random Forest classification

#### 7.1 Train Random Forest using k-fold cross-validation
```
from sklearn.ensemble import RandomForestClassifier

#Classifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[6] = ['Random Forest', accuracies.mean(), auc.mean()]
```

#### 7.2 Results
```

```

### 8. XGBoost
#### 8.1 Train XGBoost using k-fold cross-validation
```
from xgboost import XGBClassifier

#Classifier
classifier = XGBClassifier()

#Accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='accuracy')

#ROC_AUC
auc = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring='roc_auc')
df.loc[7] = ['XGBoost', accuracies.mean(), auc.mean()]
```

#### 8.2 Results
```

```

## IV. Compare and Apply
### 1. Compare models, choose the most accurate model
```

```


Conclusion:
```
(1) .
```
### 2. Apply model
```

```

### 3. Conclusion
```
1. .
```
References:

1. Predictive Models of Student College Commitment Decisions Using Machine Learning. [https://www.mdpi.com/2306-5729/4/2/65/htm](https://www.mdpi.com/2306-5729/4/2/65/htm)
2. [UCI Machine Learning Repository] (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
[BACK](./)
