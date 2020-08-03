---
layout: default
---

[BACK](./)

Reference:

1. [https://blog.csdn.net/qq_20408903/article/details/80628331](https://blog.csdn.net/qq_20408903/article/details/80628331)

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

#### 1.1 Build logistic regresson and verify
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

### 2. Decision Tree
#### 2.1 Build decision tree
```

```


#### 2.2 Evaluate model, draw ROC/AUC


### 3. Random forest
#### 3.1 Build random forest



#### 3.2 Evaluate model, draw ROC/AUC


### 4. SVM
#### 4.1 Build SVM


#### 4.2 Evaluate model, draw ROC/AUC


### 5. Compare models, choose the most accurate model
```

```


Conclusion:
```
(1) .
```
### 6. Apply model
```

```

### 7. Conclusion
```
1. .
```

[BACK](./)
