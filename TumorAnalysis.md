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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.965253</td> <td>0.989058</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.965253</td> <td>0.989058</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.961582</td> <td>0.981354</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.965253</td> <td>0.989058</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.961582</td> <td>0.981354</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.952424</td> <td>0.946226</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.965253</td> <td>0.989058</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.961582</td> <td>0.981354</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.952424</td> <td>0.946226</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.961515</td> <td>0.984498</td>
  </tr>
</table>

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
<table>
  <tr>
    <th>Classifier</th> <th>CV_Accuracy</th> <th>CV_AUC</th>
  </tr>
  <tr>
    <td>Logistic Regression</td> <td>0.967003</td> <td>0.995386</td>
  </tr>
  <tr>
    <td>K-NN</td> <td>0.967003</td> <td>0.981935</td>
  </tr>
  <tr>
    <td>SVM</td> <td>0.970707</td> <td>0.994927</td>
  </tr>
  <tr>
    <td>Kernel SVM</td> <td>0.965253</td> <td>0.989058</td>
  </tr>
  <tr>
    <td>Naive Bayer</td> <td>0.961582</td> <td>0.981354</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>0.952424</td> <td>0.946226</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>0.961515</td> <td>0.984498</td>
  </tr>
  <tr>
    <td>XGBoost</td> <td>0.965253</td> <td>0.991410</td>
  </tr>
</table>

## IV. Compare and Apply
### 1. Compare models, choose the final model
We will choose the final model only based on the cross-validated AUC score. The Logistic Regression model had the highest AUC score of 99.54%, so Logistic Regression model is the final model.

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

References:

1. Predictive Models of Student College Commitment Decisions Using Machine Learning. [https://www.mdpi.com/2306-5729/4/2/65/htm](https://www.mdpi.com/2306-5729/4/2/65/htm)
2. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)

* * *
[BACK](./)
