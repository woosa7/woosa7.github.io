---
layout: post
title:  Decision Tree & Ensemble
date:   2017-01-18
categories: python
img: randomforest.jpg
---

#### Decision Tree, Random Forest, Gradient Boosting 모델을 이용한 예측, 범주형 변수를 dummy 변수로 변환하여 모든 변수 사용하기.

----------------------------------

### Decision Tree의 장점

* 이해하기 쉽다.
* 전처리가 단순하고 학습 속도가 빠르다.
* 다양한 종류의 변수를 다룰 수 있다.
* 모델의 시각화가 쉽다.
* 통계적 가정이 적다.

### Decision Tree의 단점

* 과적합(overfitting)의 가능성이 높다.
* 결과가 불안정하다 (실행할 때마다 다른 결과 도출)
* 최적화가 어렵다.
* 학습시키기 어려운 문제들이 있다. (예, XOR 문제 등)
* 불균형 데이터에 취약하다.

### 앙상블 기법

* 하나의 모델은 underfitting, overfitting 될 수 있기 때문에 여러개의 모델을 만들어 다수결 또는 평균을 결과로 사용한다.

### Boosting

1. 모든 데이터에 동일한 가중치
2. 데이터로 모형 1을 학습
3. 모형 1이 틀린 데이터의 가중치 높임
4. 데이터로 모형 2를 학습
5. 3-4의 과정을 반복

### Gradient Boosting

1. 데이터로 모형 1을 학습
2. 모형 1의 예측과 실제의 오차
3. 위의 오차로 모형 2를 학습
4. 3-4의 과정을 반복

* 실제값 = 모형 1의 예측 + 모형 1의 오차
* 모형 1의 오차 = 모형 2의 예측 + 모형 2의 오차
* 모형 2의 오차 = 모형 3의 예측 + 모형 3의 오차
* 실제값 = 모형 1의 예측 + 모형 2의 예측 + … + 아주 작은 오차

### 데이터 준비


```python
import pandas as pd
from sklearn import metrics
```


```python
cars = pd.read_csv('data/automobile.csv')
```


```python
variables = ['bore', 'city_mpg', 'compression_ratio', 'curb_weight', 'engine_size',
             'horsepower', 'peak_rpm', 'city_mpg', 'price']
X = cars[variables]
y = cars['doors']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
```

### 모형 평가 출력 함수


```python
def model_performance(y_test, y_pred):    
    print('confusion matrix')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred)))
    print('precision : {}'.format(metrics.precision_score(y_test, y_pred, pos_label='four')))
    print('recall : {}'.format(metrics.recall_score(y_test, y_pred, pos_label='four')))
    print('F1 : {}'.format(metrics.f1_score(y_test, y_pred, pos_label='four')))
```

## 1. Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
y_pred = tree.predict(X_test)
```


```python
model_performance(y_test, y_pred)
```

    confusion matrix
    [[29 12]
     [ 7 16]]
    accuracy : 0.703125
    precision : 0.8055555555555556
    recall : 0.7073170731707317
    F1 : 0.7532467532467532
    


```python
# 모델에서 각 변수의 중요도
```


```python
varDic = {'var':variables, 'importance':tree.feature_importances_}
importance = pd.DataFrame(varDic)
importance
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
      <th>var</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.053534</td>
      <td>bore</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.108305</td>
      <td>city_mpg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>compression_ratio</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.190495</td>
      <td>curb_weight</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.275549</td>
      <td>engine_size</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.048502</td>
      <td>horsepower</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.021454</td>
      <td>peak_rpm</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.036779</td>
      <td>city_mpg</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.265382</td>
      <td>price</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rf = RandomForestClassifier(n_estimators=10, random_state=0)
rf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                n_estimators=10, n_jobs=1, oob_score=False, random_state=0,
                verbose=0, warm_start=False)




```python
y_pred_rf = rf.predict(X_test)
```


```python
model_performance(y_test, y_pred_rf)
```

    confusion matrix
    [[31 10]
     [12 11]]
    accuracy : 0.65625
    precision : 0.7209302325581395
    recall : 0.7560975609756098
    F1 : 0.7380952380952381
    

## 3. Gradient Boosting Tree


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
gb = GradientBoostingClassifier(n_estimators=10, random_state=0)
gb.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=1,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=10, presort='auto', random_state=0,
                  subsample=1.0, verbose=0, warm_start=False)




```python
y_pred_gb = gb.predict(X_test)
```


```python
model_performance(y_test, y_pred_gb)
```

    confusion matrix
    [[40  1]
     [15  8]]
    accuracy : 0.75
    precision : 0.7272727272727273
    recall : 0.975609756097561
    F1 : 0.8333333333333334
    

# 범주형 변수를 dummy 변수로 변환하여 모든 변수 사용


```python
# target인 door를 제외한 모든 범주형 변수
```


```python
cate_var = cars.columns[cars.dtypes == 'object'].difference(['doors'])
cate_var
```




    Index(['aspiration', 'body', 'cylinders', 'engine_location', 'engine_type',
           'fuel', 'fuel_system', 'maker', 'wheels'],
          dtype='object')




```python
# 범주형 변수를 dummy 변수로 변환
```


```python
dummyVar = pd.get_dummies(cars[cate_var])
```


```python
X_all = pd.concat([X, dummyVar], axis=1)  # 연속형 변수와 범주형 변수 합치기
```


```python
X_all.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bore</th>
      <th>city_mpg</th>
      <th>compression_ratio</th>
      <th>curb_weight</th>
      <th>engine_size</th>
      <th>horsepower</th>
      <th>peak_rpm</th>
      <th>city_mpg</th>
      <th>price</th>
      <th>aspiration_std</th>
      <th>...</th>
      <th>maker_plymouth</th>
      <th>maker_porsche</th>
      <th>maker_saab</th>
      <th>maker_subaru</th>
      <th>maker_toyota</th>
      <th>maker_volkswagen</th>
      <th>maker_volvo</th>
      <th>wheels_4wd</th>
      <th>wheels_fwd</th>
      <th>wheels_rwd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.19</td>
      <td>24</td>
      <td>10.0</td>
      <td>2337</td>
      <td>109</td>
      <td>102</td>
      <td>5500</td>
      <td>24</td>
      <td>13950</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.19</td>
      <td>18</td>
      <td>8.0</td>
      <td>2824</td>
      <td>136</td>
      <td>115</td>
      <td>5500</td>
      <td>18</td>
      <td>17450</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.19</td>
      <td>19</td>
      <td>8.5</td>
      <td>2844</td>
      <td>136</td>
      <td>110</td>
      <td>5500</td>
      <td>19</td>
      <td>17710</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.13</td>
      <td>17</td>
      <td>8.3</td>
      <td>3086</td>
      <td>131</td>
      <td>140</td>
      <td>5500</td>
      <td>17</td>
      <td>23875</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.50</td>
      <td>23</td>
      <td>8.8</td>
      <td>2395</td>
      <td>108</td>
      <td>101</td>
      <td>5800</td>
      <td>23</td>
      <td>16430</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 56 columns</p>
</div>




```python
X_all_train, X_all_test, y_train, y_test = train_test_split(X_all, y, test_size=0.4)
```

### SVC


```python
from sklearn.svm import SVC
```


```python
model = SVC(kernel='rbf')
model.fit(X_all_train, y_train)
y_pred = model.predict(X_all_test)
```


```python
model_performance(y_test, y_pred)
```

    confusion matrix
    [[41  0]
     [23  0]]
    accuracy : 0.640625
    precision : 0.640625
    recall : 1.0
    F1 : 0.780952380952381
    

### DecisionTree


```python
model = DecisionTreeClassifier()
model.fit(X_all_train, y_train)
y_pred = model.predict(X_all_test)
```


```python
model_performance(y_test, y_pred)
```

    confusion matrix
    [[33  8]
     [11 12]]
    accuracy : 0.703125
    precision : 0.75
    recall : 0.8048780487804879
    F1 : 0.7764705882352942
    

### RandomForest


```python
model = RandomForestClassifier()
model.fit(X_all_train, y_train)
y_pred = model.predict(X_all_test)
```


```python
model_performance(y_test, y_pred)
```

    confusion matrix
    [[36  5]
     [ 9 14]]
    accuracy : 0.78125
    precision : 0.8
    recall : 0.8780487804878049
    F1 : 0.8372093023255814
    

### GradientBoosting


```python
model = GradientBoostingClassifier(random_state=0)
model.fit(X_all_train, y_train)
y_pred = model.predict(X_all_test)
```


```python
model_performance(y_test, y_pred)
```

    confusion matrix
    [[38  3]
     [ 6 17]]
    accuracy : 0.859375
    precision : 0.8636363636363636
    recall : 0.926829268292683
    F1 : 0.8941176470588236
    
