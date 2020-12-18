---
layout: default
---

[BACK](./)

## I. Background

1. To analysis and predict PM2.5.

1. Data source: UCI

1. Variables
```
No: row number 
year: year of data in this row 
month: month of data in this row 
day: day of data in this row 
hour: hour of data in this row 
pm2.5: PM2.5 concentration (ug/m^3) 
DEWP: Dew Point (â„ƒ) 
TEMP: Temperature (â„ƒ) 
PRES: Pressure (hPa) 
cbwd: Combined wind direction 
Iws: Cumulated wind speed (m/s) 
Is: Cumulated hours of snow 
Ir: Cumulated hours of rain 
```

1. Objects and measurement:
```
(1) To build the PM2.5 prediction model.
(2) The machine learning techniques utilized are:
Multiple Linear Regression
Support Vector Regression
Decision Tree
Random Forest
Artificial Neural Network
XGBoost
```

## II. Data Processing

### 1. Importing the libraries
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
```

### 2. Importing the dataset
```
dataset = pd.read_csv('PRSA_data.csv')

# View Dataset
dataset.head()
```

# View Dataset Info
```
dataset.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 43824 entries, 0 to 43823
Data columns (total 13 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   No      43824 non-null  int64  
 1   year    43824 non-null  int64  
 2   month   43824 non-null  int64  
 3   day     43824 non-null  int64  
 4   hour    43824 non-null  int64  
 5   pm2.5   41757 non-null  float64
 6   DEWP    43824 non-null  int64  
 7   TEMP    43824 non-null  float64
 8   PRES    43824 non-null  float64
 9   cbwd    43824 non-null  object 
 10  Iws     43824 non-null  float64
 11  Is      43824 non-null  int64  
 12  Ir      43824 non-null  int64  
dtypes: float64(4), int64(8), object(1)
memory usage: 4.3+ MB
None
```
# Move target to the last column
```
dataset = dataset[ [ col for col in dataset.columns 
                    if col != 'pm2.5'] + ['pm2.5'] ]

# Drop NULL target
dataset = dataset[dataset['pm2.5'].notnull()]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

### 3. Encoding categorical data
```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

### 4. Splitting the dataset into the Training set and Test set
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### 5. Feature Scaling
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### 6. Define a Dataframe
```
df = pd.DataFrame(columns=['Regression', 'Mean Squared Error'], dtype=float)
```

## III. Build predicting model

### 1. Multiple Linear Regression

#### 1.1 Training the Multiple Linear Regression model
```
# Build the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```
#### 1.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
</table>

### 2. Support Vector Regression
#### 2.1 Train SVR
```
# Build the model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```
#### 2.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
  <tr>
    <td>SVR</td> <td>6269</td>
  </tr>
</table>

### 3. Decision Tree
#### 3.1 Train Decision Tree
```
# Build the model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```

#### 3.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
  <tr>
    <td>SVR</td> <td>6269</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>2410</td>
  </tr>
</table>

### 4. Random Forest
#### 4.1 Train Random Forest
```
# Build the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```

#### 4.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
  <tr>
    <td>SVR</td> <td>6269</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>2410</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>1383</td>
  </tr>
</table>

### 5. Artificial Neural Network
#### 5.1 Train ANN
```
X_train = np.asarray(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# Build the model
import tensorflow as tf

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=32, activation='relu',input_dim = 14))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Adding the third hidden layer
#ann.add(tf.keras.layers.Dense(units=32, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the test set results
y_pred = ann.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```

#### 5.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
  <tr>
    <td>SVR</td> <td>6269</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>2410</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>1383</td>
  </tr>
  <tr>
    <td>ANN</td> <td>2276</td>
  </tr>
</table>
### 6. XGBoost

#### 6.1 Train XGBoost
```
# Build the model
import xgboost as xgb
regressor = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.9, 
                          learning_rate = 0.4, max_depth = 5, alpha = 10, n_estimators = 120)
regressor.fit(X_train,y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Calculate MSE
mean_squared_error(y_test, y_pred)
```

#### 6.2 Results
<table>
  <tr>
    <th>Regression</th> <th>Mean Squared Error</th>
  </tr>
  <tr>
    <td>Multiple Linear Regression</td> <td>5996</td>
  </tr>
  <tr>
    <td>SVR</td> <td>6269</td>
  </tr>
  <tr>
    <td>Decision Tree</td> <td>2410</td>
  </tr>
  <tr>
    <td>Random Forest</td> <td>1383</td>
  </tr>
  <tr>
    <td>ANN</td> <td>2276</td>
  </tr>
  <tr>
    <td>XGBoost</td> <td>2074</td>
  </tr>
</table>

## IV. Compare and Apply
### 1. Compare models, choose the final model
We will choose the final model only based on the mean squared error score. The Random Forest model had the lowest MSE score of 1383, so Random Forest model is the best among all models.

References:

1. Using XGBoost in Python. [https://www.datacamp.com/community/tutorials/xgboost-in-python](https://www.datacamp.com/community/tutorials/xgboost-in-python)
2. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)

* * *
[BACK](./)
