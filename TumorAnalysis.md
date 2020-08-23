---
layout: default
---

[BACK](./)

## I. Background

1. To classify tumor is good or bad.

1. Data source: kaggle

1. Variables
```
size: the size of tumor
```

1. Objects and measurement:
```
(1) To classify tumor is good or bad.
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

### 5. View data
```
```

## III. Build predicting model

### 1. Logistic regression

#### 1.1 Training the Logistic Regression model on the Training set
```
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
```

#### 1.2 Predicting the Test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 1.3 Making the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[103   4]
 [  5  59]]
0.9473684210526315
```

### 2. K nearest neighbors
#### 2.1 Train K nearest neighbors on the training set
```
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
```
#### 2.2 Predicting the Test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```
#### 2.3 Making the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[103   4]
 [  5  59]]
0.9473684210526315
```

### 3. Support Vector Machine
#### 3.1 Train Support Vector Machine on the training set
```
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
```
#### 3.2 Predicting the Test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 3.3 Making the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[102   5]
 [  5  59]]
0.9415204678362573
```
### 4. Kernel SVM
#### 4.1 Train Kernel SVM on the training set
```
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
```

#### 4.2 Predicting the Test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 4.3 Making the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[102   5]
 [  3  61]]
0.9532163742690059
```

### 5. Naive Bayes
#### 5.1 Train Naive Bayes on the training set
```
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

#### 5.2 Predicting the test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 5.3 Make the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[99  8]
 [ 2 62]]
0.9415204678362573
```

### 6. Decision Tree classification

#### 6.1 Train Decision Tree on the training set
```
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

#### 6.2 Predicting the test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 6.3 Make the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[103   4]
 [  3  61]]
0.9590643274853801
```

### 7. Random Forest classification

#### 7.1 Train Random Forest on the training set
```
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
```

#### 7.2 Predicting the test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```
#### 7.3 Make the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[102   5]
 [  6  58]]
0.935672514619883
```

### 8. XGBoost
#### 8.1 Train XGBoost on the training set
```
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

#### 8.2 Predicting the test set results
```
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```

#### 8.3 Make the Confusion Matrix
```
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```
```
[[84  3]
 [ 0 50]]
0.9781021897810219
```
#### 8.4 Applying k-Fold Cross Validation
```
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```
```
Accuracy: 96.53 %
Standard Deviation: 2.07 %
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
Reference:

1. [https://blog.csdn.net/qq_20408903/article/details/80628331](https://blog.csdn.net/qq_20408903/article/details/80628331)

[BACK](./)
