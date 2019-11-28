---
layout: default
---

Reference:

1. [https://chf2012.github.io/2017/01/30/数据分析/20_数据分析/专题分析/优秀员工离职原因分析与预测/](https://chf2012.github.io/2017/01/30/数据分析/20_数据分析/专题分析/优秀员工离职原因分析与预测/)
2. [https://blog.csdn.net/qq_20408903/article/details/80628331](https://blog.csdn.net/qq_20408903/article/details/80628331)
3. [https://blog.csdn.net/wanglingli95/article/details/79435976](https://blog.csdn.net/wanglingli95/article/details/79435976)
4. [https://blog.csdn.net/Cocaine_bai/article/details/80749588](https://blog.csdn.net/Cocaine_bai/article/details/80749588)
5. [https://blog.csdn.net/Cocaine_bai/article/details/80758636](https://blog.csdn.net/Cocaine_bai/article/details/80758636)

## I. Background

1. Empolyee attritioin analysis

2. Data source: kaggle

3. Variable

satisfaction: Employee satisfaction level
evaluation: Last evaluation
project: Number of projects
hours: Average monthly hours
years: Time spent at the company
accident: Whether they have had a work accident
promotion: Whether they have had a promotion in the last 5 years
sales: Department
salary: Salary
left: Whether the employee has left

4. Objects and measurement

## II. Data Analysis

Libraries

```
library(readr)
library(dplyr)
library(ggplot2)
library(gmodels)
```

### 1. Import Data and view

![satisfaction](./images/satisfaction.jpg)


```js
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```

[BACK](./)
