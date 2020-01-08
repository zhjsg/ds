---
layout: default
---

[BACK](./)

Reference:

1. [https://chf2012.github.io/2017/01/30/数据分析/20_数据分析/专题分析/优秀员工离职原因分析与预测/](https://chf2012.github.io/2017/01/30/数据分析/20_数据分析/专题分析/优秀员工离职原因分析与预测/)
2. [https://blog.csdn.net/qq_20408903/article/details/80628331](https://blog.csdn.net/qq_20408903/article/details/80628331)
3. [https://blog.csdn.net/wanglingli95/article/details/79435976](https://blog.csdn.net/wanglingli95/article/details/79435976)
4. [https://blog.csdn.net/Cocaine_bai/article/details/80749588](https://blog.csdn.net/Cocaine_bai/article/details/80749588)
5. [https://blog.csdn.net/Cocaine_bai/article/details/80758636](https://blog.csdn.net/Cocaine_bai/article/details/80758636)

## I. Background

1. Plenty of excellent and experienced employee resign ahead of expected.

1. Data source: kaggle

1. Variables
```
satisfaction: employee satisfaction level
evaluation: last evaluation
project: number of projects
hours: average monthly hours
years: time spent at the company
accident: whether they have had a work accident
promotion: whether they have had a promotion in the last 5 years
sales: department
salary: salary
left: whether the employee has left
```

1. Objects and measurement:
```
(1) Analyze what is the possible reasons the employee resign
(2) Build the predicting model and predict which excellent employee will be the next to resign
```

## II. Data Analysis

Libraries
```
library(readr)
library(dplyr)
library(ggplot2)
library(gmodels)
```

### 1. Import Data and view

1.1 Import data
```
hr <- read_csv("HR_comma_sep.csv")
hr <- tbl_df(hr)
View(hr)
str(hr)
```
![view_hr_file](./images/view_hr_file.png)

1.2 Rename variables
```
colnames(hr) <- c("satisfaction", "evaluation", "project", "hours", "years", "accident", "left", "promotion", "sales", "salary")
```

1.3 Factor
```
hr$sales <- factor(hr$sales)
hr$salary <- factor(hr$salary, levels=c("low", "medium", "high"))
```

1.4 View data
```
sum(is.na(hr))
```
#[1] 0

```
summary(hr)
```
![view_hr_summary](./images/view_hr_summary.png)

### 2. Based on assumption, choose the subset of excellent employees
Excellent employee:
```
(1) evaludation >= 0.75
(2) project >= 4
(3) year >= 4
```

2.1 filter subset based on assumption
```
hr_good <- filter(hr, evaluation>=0.75 & years>=4 & project>=4)
```

2.2 Compare left employee from subset with overall
```
CrossTable(hr$left)
```
![hr_left](./images/hr_left.png)

```
CrossTable(hr_good$left)
```
![hr_good_left](./images/hr_good_left.png)

Conclusion:
```
(1) from overall left employees, excellent employee percentage = 1778/3571 = 50%
(2) from excellent employees subset, the left percentage = 1778/2763 = 64%
```

2.3 view excellent employee subset
```
summary(hr_good)
```
![view_hr_good_summary](./images/view_hr_good_summary.png)

2.4 View the correlationship among the variables in excellent employees subset
```
library("corrplot")
hr_good_corr <- select(hr, -sales, -salary) %>% cor()
corrplot(hr_good_corr, method="circle", tl.col="black", title="Left and Satisfaction", mar=c(1,1,3,1))
```
![satisfaction](./images/satisfaction.png)

Conclusion:
```
(1) Left has negative correlationship with satisfaction, and high correlated.
```
### 3. View the relationship among employee left, satisfaction and other variables 

3.1 View satisfaction distribution
```

```
![xxx](./images/xxx.png)
3.2 View the relationship among salary, working hours, satisfaction and left
```

```
![xxx](./images/xxx.png)
Conclusion:
```
(1)
(2)
(3)
```
3.3 View the relationship among promotion, satisfaction and left
```

```
![xxx](./images/xxx.png)
Conclusion:
```
(1)
```
3.4 View the relationship among working years, satisfaction and left
```

```
![xxx](./images/xxx.png)
Conclusion:
```
(1)
(2)
(3)
```
3.5 View the relationship among department, projects and left
```

```
![xxx](./images/xxx.png)
Conclusion:
```
(1)
```
3.6 View the relationship between department and left
```

```
![xxx](./images/xxx.png)
Conclusion:
```
(1)
```

## III. Build predicting model
Data Partition
```

```
### 1. Logistic regression

1.1 Build logistic regresson and verify
```

```
Confusion Matrix and Statistics
![xxx](./images/xxx.png)

1.2 Evaluate model, draw ROC/AUC
```

```
![xxx](./images/xxx.png)

### 2. Decision Tree
2.1 Build decision tree
```

```
![xxx](./images/xxx.png)
```

```
![xxx](./images/xxx.png)
```

```
![xxx](./images/xxx.png)
```

```
Confusion Matrix and Statistics
![xxx](./images/xxx.png)
2.2 Evaluate model, draw ROC/AUC
```

```
![xxx](./images/xxx.png)

### 3. Random forest
3.1 Build random forest
```

```
![xxx](./images/xxx.png)
```

```
![xxx](./images/xxx.png)
```

```
![xxx](./images/xxx.png)
3.2 Evaluate model, draw ROC/AUC
```

```
![xxx](./images/xxx.png)

### 4. SVM
4.1 Build SVM
```

```
Confusion Matrix and Statistics
![xxx](./images/xxx.png)
4.2 Evaluate model, draw ROC/AUC
```

```
![xxx](./images/xxx.png)

### 5. Compare models, choose the most accurate model
```

```
![xxx](./images/xxx.png)
Conclusion:
```

```
### 6. Apply model
```

```
MeanDecreaseGini
![xxx](./images/xxx.png)
6.1 Remove unimportant factors, rebuild the model
```

```
![xxx](./images/xxx.png)
```

```
Confusion Matrix and Statistics
![xxx](./images/xxx.png)
6.2 Evaluate model, draw ROC/AUC
```

```
![xxx](./images/xxx.png)

### 7. Conclusion
```
1.
2.
3.
```


* * *
JS
<script>document.write(5 + 6);</script>


[BACK](./)
