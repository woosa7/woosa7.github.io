---
layout: post
title:  개인의 income을 예측하는 머신러닝 모델링
date:   2017-01-21
categories: python
img: multilayerperceptron_network.png
---

#### Logistic Regression, Decision Tree, Random Forest, Gradient Boosting Tree, SVM, k-NN, Neural Network

----------------------------------

### 데이터 불러오기


```python
import pandas as pd
from sklearn import metrics
```


```python
train = pd.read_csv('data/train.csv')
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>marital</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>under50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>under50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>under50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>under50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>under50k</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.shape
```




    (24999, 12)




```python
train.columns
```




    Index(['age', 'workclass', 'education', 'marital', 'occupation',
           'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
           'hours_per_week', 'income'],
          dtype='object')



#### 종속변수


```python
y = train['income']
y.head()
```




    0    under50k
    1    under50k
    2    under50k
    3    under50k
    4    under50k
    Name: income, dtype: object



#### 독립변수


```python
# 연속형 변수
conti_var = train.columns[train.dtypes != 'object']
conti_var
```




    Index(['age', 'education', 'capital_gain', 'capital_loss', 'hours_per_week'], dtype='object')




```python
# 범주형 변수
cate_var = train.columns[train.dtypes == 'object'].difference(['income'])
cate_var
```




    Index(['marital', 'occupation', 'race', 'relationship', 'sex', 'workclass'], dtype='object')




```python
# 범주형 변수를 dummy 변수로 변환
```


```python
dummy_var = pd.get_dummies(train[cate_var])
```


```python
X = pd.concat([train[conti_var], dummy_var], axis=1)
X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>marital_Divorced</th>
      <th>marital_Married-AF-spouse</th>
      <th>marital_Married-civ-spouse</th>
      <th>marital_Married-spouse-absent</th>
      <th>marital_Never-married</th>
      <th>...</th>
      <th>relationship_Wife</th>
      <th>sex_Female</th>
      <th>sex_Male</th>
      <th>workclass_Federal-gov</th>
      <th>workclass_Local-gov</th>
      <th>workclass_Private</th>
      <th>workclass_Self-emp-inc</th>
      <th>workclass_Self-emp-not-inc</th>
      <th>workclass_State-gov</th>
      <th>workclass_Without-pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>13</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>50</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
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
      <td>38</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>



### 데이터 분할


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
```


```python
X_train.shape
```




    (22499, 46)




```python
X_test.shape
```




    (2500, 46)



### 모형 평가 출력 함수


```python
# 년수입이 $ 50,000 를 넘는 사람에 대한 분류/예측 결과
```


```python
def model_performance(y_test, y_pred):    
    print('confusion matrix')
    print(metrics.confusion_matrix(y_test, y_pred))
    print('accuracy : {}'.format(metrics.accuracy_score(y_test, y_pred).round(3)))
    print('precision : {}'.format(metrics.precision_score(y_test, y_pred, pos_label='over50k').round(3)))
    print('recall : {}'.format(metrics.recall_score(y_test, y_pred, pos_label='over50k').round(3)))
    print('F1 : {}'.format(metrics.f1_score(y_test, y_pred, pos_label='over50k').round(3)))
```

## 1. Logistic Regression (Lasso / Ridge)


```python
from sklearn.linear_model import LogisticRegression
```


```python
def run_lr_model(penalties, Clist):
    for p, c in zip(penalties, Clist):
        print('---------- penalty : {}, C : {} ----------'.format(p, c))
        model = LogisticRegression(penalty=p, C=c)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance(y_test, y_pred)
        print('\n')
```

* penalty : l1 = Lasso / l2 = Ridge
* Lasso (L1 regularization) : 특정 변수의 coef를 0으로 만듬. automatic feature selection.
* Ridge (L2 regularization) : coef를 조절하지만 0으로 만들지는 않음.
* C > 1 : 오차를 주로 줄임. 가능한 training set에 맞춤 / C < 1 : coef를 주로 줄임


```python
plist = ['l1','l1','l1','l1','l1','l2','l2','l2','l2','l2',]
```


```python
clist = [0.001, 0.01, 0.1, 1, 100, 0.001, 0.01, 0.1, 1, 100]
```


```python
run_lr_model(plist, clist)
```

    ---------- penalty : l1, C : 0.001 ----------
    confusion matrix
    [[ 168  461]
     [  55 1816]]
    accuracy : 0.794
    precision : 0.753
    recall : 0.267
    F1 : 0.394
    
    
    ---------- penalty : l1, C : 0.01 ----------
    confusion matrix
    [[ 338  291]
     [ 107 1764]]
    accuracy : 0.841
    precision : 0.76
    recall : 0.537
    F1 : 0.629
    
    
    ---------- penalty : l1, C : 0.1 ----------
    confusion matrix
    [[ 374  255]
     [ 124 1747]]
    accuracy : 0.848
    precision : 0.751
    recall : 0.595
    F1 : 0.664
    
    
    ---------- penalty : l1, C : 1 ----------
    confusion matrix
    [[ 378  251]
     [ 129 1742]]
    accuracy : 0.848
    precision : 0.746
    recall : 0.601
    F1 : 0.665
    
    
    ---------- penalty : l1, C : 100 ----------
    confusion matrix
    [[ 379  250]
     [ 129 1742]]
    accuracy : 0.848
    precision : 0.746
    recall : 0.603
    F1 : 0.667
    
    
    ---------- penalty : l2, C : 0.001 ----------
    confusion matrix
    [[ 190  439]
     [  59 1812]]
    accuracy : 0.801
    precision : 0.763
    recall : 0.302
    F1 : 0.433
    
    
    ---------- penalty : l2, C : 0.01 ----------
    confusion matrix
    [[ 338  291]
     [  99 1772]]
    accuracy : 0.844
    precision : 0.773
    recall : 0.537
    F1 : 0.634
    
    
    ---------- penalty : l2, C : 0.1 ----------
    confusion matrix
    [[ 369  260]
     [ 121 1750]]
    accuracy : 0.848
    precision : 0.753
    recall : 0.587
    F1 : 0.66
    
    
    ---------- penalty : l2, C : 1 ----------
    confusion matrix
    [[ 374  255]
     [ 124 1747]]
    accuracy : 0.848
    precision : 0.751
    recall : 0.595
    F1 : 0.664
    
    
    ---------- penalty : l2, C : 100 ----------
    confusion matrix
    [[ 373  256]
     [ 124 1747]]
    accuracy : 0.848
    precision : 0.751
    recall : 0.593
    F1 : 0.663
    
    
    


```python
# Lasso & C = 100 인 모델의 F1 score가 가장 높다.
```

## 2. Decision Tree


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')




```python
y_pred = model.predict(X_test)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 410  219]
     [ 258 1613]]
    accuracy : 0.809
    precision : 0.614
    recall : 0.652
    F1 : 0.632
    

## 3. Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
def run_rf_model(n_estimators, n_jobs):
    for ne, nj in zip(n_estimators, n_jobs):
        print('---------- n_estimators : {}, n_jobs : {} ----------'.format(ne, nj))
        model = RandomForestClassifier(n_estimators=ne, n_jobs=nj)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance(y_test, y_pred)
        print('\n')
```

* n_estimators = number of trees

* n_jobs = number of jobs to run in parallel for both fit and predict.


```python
n_estimators = [10, 10, 10, 100, 100, 100, 1000, 1000, 1000]
```


```python
n_jobs = [1, 10, 100, 1, 10, 100, 1, 10, 100]
```


```python
run_rf_model(n_estimators, n_jobs)
```

    ---------- n_estimators : 10, n_jobs : 1 ----------
    confusion matrix
    [[ 409  220]
     [ 195 1676]]
    accuracy : 0.834
    precision : 0.677
    recall : 0.65
    F1 : 0.663
    
    
    ---------- n_estimators : 10, n_jobs : 10 ----------
    confusion matrix
    [[ 420  209]
     [ 194 1677]]
    accuracy : 0.839
    precision : 0.684
    recall : 0.668
    F1 : 0.676
    
    
    ---------- n_estimators : 10, n_jobs : 100 ----------
    confusion matrix
    [[ 412  217]
     [ 189 1682]]
    accuracy : 0.838
    precision : 0.686
    recall : 0.655
    F1 : 0.67
    
    
    ---------- n_estimators : 100, n_jobs : 1 ----------
    confusion matrix
    [[ 397  232]
     [ 153 1718]]
    accuracy : 0.846
    precision : 0.722
    recall : 0.631
    F1 : 0.673
    
    
    ---------- n_estimators : 100, n_jobs : 10 ----------
    confusion matrix
    [[ 397  232]
     [ 156 1715]]
    accuracy : 0.845
    precision : 0.718
    recall : 0.631
    F1 : 0.672
    
    
    ---------- n_estimators : 100, n_jobs : 100 ----------
    confusion matrix
    [[ 398  231]
     [ 154 1717]]
    accuracy : 0.846
    precision : 0.721
    recall : 0.633
    F1 : 0.674
    
    
    ---------- n_estimators : 1000, n_jobs : 1 ----------
    confusion matrix
    [[ 392  237]
     [ 159 1712]]
    accuracy : 0.842
    precision : 0.711
    recall : 0.623
    F1 : 0.664
    
    
    ---------- n_estimators : 1000, n_jobs : 10 ----------
    confusion matrix
    [[ 398  231]
     [ 149 1722]]
    accuracy : 0.848
    precision : 0.728
    recall : 0.633
    F1 : 0.677
    
    
    ---------- n_estimators : 1000, n_jobs : 100 ----------
    confusion matrix
    [[ 394  235]
     [ 152 1719]]
    accuracy : 0.845
    precision : 0.722
    recall : 0.626
    F1 : 0.671
    
    
    


```python
# n_estimators : 1000, n_jobs : 100 인 random forest 모델의  F1 score가 가장 높으며, Lasso Logistic Regression 결과보다 높다.
```

## 4. Gradient Boosting Tree


```python
from sklearn.ensemble import GradientBoostingClassifier
```

### (1) loss function : ‘deviance’ = logistic regression (default)


```python
def run_gbt_model(n_estimators, l_rate):
    for ne, lr in zip(n_estimators, l_rate):
        print('---------- n_estimators : {}, learning_rate : {} ----------'.format(ne, lr))
        model = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance(y_test, y_pred)
        print('\n')
```

* n_estimators = number of boosting stages to perform


```python
n_estimators = [100, 100, 100, 1000, 1000, 1000]
```


```python
l_rate = [0.1, 0.3, 0.5, 0.1, 0.3, 0.5]
```


```python
run_gbt_model(n_estimators, l_rate)
```

    ---------- n_estimators : 100, learning_rate : 0.1 ----------
    confusion matrix
    [[ 379  250]
     [ 102 1769]]
    accuracy : 0.859
    precision : 0.788
    recall : 0.603
    F1 : 0.683
    
    
    ---------- n_estimators : 100, learning_rate : 0.3 ----------
    confusion matrix
    [[ 407  222]
     [ 111 1760]]
    accuracy : 0.867
    precision : 0.786
    recall : 0.647
    F1 : 0.71
    
    
    ---------- n_estimators : 100, learning_rate : 0.5 ----------
    confusion matrix
    [[ 414  215]
     [ 119 1752]]
    accuracy : 0.866
    precision : 0.777
    recall : 0.658
    F1 : 0.713
    
    
    ---------- n_estimators : 1000, learning_rate : 0.1 ----------
    confusion matrix
    [[ 420  209]
     [ 121 1750]]
    accuracy : 0.868
    precision : 0.776
    recall : 0.668
    F1 : 0.718
    
    
    ---------- n_estimators : 1000, learning_rate : 0.3 ----------
    confusion matrix
    [[ 423  206]
     [ 133 1738]]
    accuracy : 0.864
    precision : 0.761
    recall : 0.672
    F1 : 0.714
    
    
    ---------- n_estimators : 1000, learning_rate : 0.5 ----------
    confusion matrix
    [[ 415  214]
     [ 135 1736]]
    accuracy : 0.86
    precision : 0.755
    recall : 0.66
    F1 : 0.704
    
    
    

### (2) loss function : ‘exponential’ = AdaBoost algorithm.


```python
def run_gbtExp_model(n_estimators, l_rate):
    for ne, lr in zip(n_estimators, l_rate):
        print('---------- n_estimators : {}, learning_rate : {} ----------'.format(ne, lr))
        model = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, loss='exponential')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance(y_test, y_pred)
        print('\n')
```


```python
run_gbt_model(n_estimators, l_rate)
```

    ---------- n_estimators : 100, learning_rate : 0.1 ----------
    confusion matrix
    [[ 379  250]
     [ 102 1769]]
    accuracy : 0.859
    precision : 0.788
    recall : 0.603
    F1 : 0.683
    
    
    ---------- n_estimators : 100, learning_rate : 0.3 ----------
    confusion matrix
    [[ 407  222]
     [ 111 1760]]
    accuracy : 0.867
    precision : 0.786
    recall : 0.647
    F1 : 0.71
    
    
    ---------- n_estimators : 100, learning_rate : 0.5 ----------
    confusion matrix
    [[ 414  215]
     [ 119 1752]]
    accuracy : 0.866
    precision : 0.777
    recall : 0.658
    F1 : 0.713
    
    
    ---------- n_estimators : 1000, learning_rate : 0.1 ----------
    confusion matrix
    [[ 419  210]
     [ 121 1750]]
    accuracy : 0.868
    precision : 0.776
    recall : 0.666
    F1 : 0.717
    
    
    ---------- n_estimators : 1000, learning_rate : 0.3 ----------
    confusion matrix
    [[ 423  206]
     [ 132 1739]]
    accuracy : 0.865
    precision : 0.762
    recall : 0.672
    F1 : 0.715
    
    
    ---------- n_estimators : 1000, learning_rate : 0.5 ----------
    confusion matrix
    [[ 415  214]
     [ 135 1736]]
    accuracy : 0.86
    precision : 0.755
    recall : 0.66
    F1 : 0.704
    
    
    

* F1 score와 Accuracy가 가장 높은 Tree에서 변수의 중요도 확인


```python
# feature importance
```


```python
model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1)
model.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.1, loss='deviance', max_depth=3,
                  max_features=None, max_leaf_nodes=None,
                  min_impurity_split=1e-07, min_samples_leaf=1,
                  min_samples_split=2, min_weight_fraction_leaf=0.0,
                  n_estimators=1000, presort='auto', random_state=None,
                  subsample=1.0, verbose=0, warm_start=False)




```python
varDic = {'var':X_train.columns, 'importance':model.feature_importances_}
impVar = pd.DataFrame(varDic)
impVar.sort_values(by='importance', ascending=False)[1:11]
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
      <th>4</th>
      <td>0.135903</td>
      <td>hours_per_week</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.108185</td>
      <td>education</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.104246</td>
      <td>capital_loss</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.089625</td>
      <td>capital_gain</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.031849</td>
      <td>marital_Married-civ-spouse</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.021935</td>
      <td>workclass_Self-emp-not-inc</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.020932</td>
      <td>workclass_Self-emp-inc</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.019298</td>
      <td>occupation_Exec-managerial</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.017739</td>
      <td>relationship_Wife</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.017046</td>
      <td>occupation_Sales</td>
    </tr>
  </tbody>
</table>
</div>



## 5. SVM

* SVM Classification 의 경우 kernel='linear'로 설정시 22499 건의 데이터를 훈련시키는데 엄청난 시간이 걸림.
* 범주형 변수를 제외하고 연속형 변수만으로 Classification을 하는 경우도 시간이 많이 소요됨.


```python
from sklearn.svm import SVC
```

### RBF kernel


```python
model = SVC(C=10)
```


```python
model.fit(X_train, y_train)
```




    SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
y_pred = model.predict(X_test)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 397  232]
     [ 119 1752]]
    accuracy : 0.86
    precision : 0.769
    recall : 0.631
    F1 : 0.693
    

### 변수 Scaling


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
scaler.fit(X_train)
```




    MinMaxScaler(copy=True, feature_range=(0, 1))




```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
model = SVC(C=10)
model.fit(X_train_scaled, y_train)
```




    SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)




```python
y_pred = model.predict(X_test_scaled)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 369  260]
     [ 122 1749]]
    accuracy : 0.847
    precision : 0.752
    recall : 0.587
    F1 : 0.659
    

## 6. k-NN


```python
from sklearn.neighbors import KNeighborsClassifier
```


```python
neighbors = range(1,17,2)  # 최근접이웃 갯수.
```


```python
def run_knn_model(n_neighbors):
    for nn in n_neighbors:
        print('---------- knn : ' + str(nn) + ' ----------')
        model = KNeighborsClassifier(n_neighbors=nn)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance(y_test, y_pred)
        print('\n')
```


```python
run_knn_model(neighbors)
```

    ---------- knn : 1 ----------
    confusion matrix
    [[ 411  218]
     [ 245 1626]]
    accuracy : 0.815
    precision : 0.627
    recall : 0.653
    F1 : 0.64
    
    
    ---------- knn : 3 ----------
    confusion matrix
    [[ 414  215]
     [ 189 1682]]
    accuracy : 0.838
    precision : 0.687
    recall : 0.658
    F1 : 0.672
    
    
    ---------- knn : 5 ----------
    confusion matrix
    [[ 406  223]
     [ 160 1711]]
    accuracy : 0.847
    precision : 0.717
    recall : 0.645
    F1 : 0.679
    
    
    ---------- knn : 7 ----------
    confusion matrix
    [[ 392  237]
     [ 152 1719]]
    accuracy : 0.844
    precision : 0.721
    recall : 0.623
    F1 : 0.668
    
    
    ---------- knn : 9 ----------
    confusion matrix
    [[ 394  235]
     [ 155 1716]]
    accuracy : 0.844
    precision : 0.718
    recall : 0.626
    F1 : 0.669
    
    
    ---------- knn : 11 ----------
    confusion matrix
    [[ 400  229]
     [ 148 1723]]
    accuracy : 0.849
    precision : 0.73
    recall : 0.636
    F1 : 0.68
    
    
    ---------- knn : 13 ----------
    confusion matrix
    [[ 393  236]
     [ 151 1720]]
    accuracy : 0.845
    precision : 0.722
    recall : 0.625
    F1 : 0.67
    
    
    ---------- knn : 15 ----------
    confusion matrix
    [[ 383  246]
     [ 156 1715]]
    accuracy : 0.839
    precision : 0.711
    recall : 0.609
    F1 : 0.656
    
    
    

## 7. Neural Network


```python
from sklearn.neural_network import MLPClassifier
```

### (1) adam : stochastic gradient-based optimizer


```python
model = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(100,), max_iter=2000)
```


```python
model.fit(X_train, y_train)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
y_pred = model.predict(X_test)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 403  226]
     [ 171 1700]]
    accuracy : 0.841
    precision : 0.702
    recall : 0.641
    F1 : 0.67
    

### (2) sgd : stochastic gradient descent


```python
model = MLPClassifier(solver='sgd', activation='logistic', hidden_layer_sizes=(200,), max_iter=2000)
```


```python
model.fit(X_train, y_train)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(200,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='sgd', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
y_pred = model.predict(X_test)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 383  246]
     [ 215 1656]]
    accuracy : 0.816
    precision : 0.64
    recall : 0.609
    F1 : 0.624
    

### 변수 Scaling + adam


```python
from sklearn.preprocessing import MinMaxScaler
```


```python
scaler = MinMaxScaler()
scaler.fit(X_train)
```




    MinMaxScaler(copy=True, feature_range=(0, 1))




```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


```python
model = MLPClassifier(solver='adam', activation='logistic', hidden_layer_sizes=(100,), max_iter=2000)
model.fit(X_train_scaled, y_train)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=2000, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
y_pred = model.predict(X_test_scaled)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 411  218]
     [ 159 1712]]
    accuracy : 0.849
    precision : 0.721
    recall : 0.653
    F1 : 0.686
    

# 최종 예측


```python
newdata = pd.read_csv('data/newpeople.csv')
newdata.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education</th>
      <th>marital</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>Private</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>Local-gov</td>
      <td>11</td>
      <td>Divorced</td>
      <td>Protective-serv</td>
      <td>Own-child</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51</td>
      <td>Private</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>7298</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>Private</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63</td>
      <td>Private</td>
      <td>14</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
dummy_final = pd.get_dummies(newdata[cate_var])
```


```python
X_final = pd.concat([newdata[conti_var], dummy_final], axis=1)
X_final.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>marital_Divorced</th>
      <th>marital_Married-AF-spouse</th>
      <th>marital_Married-civ-spouse</th>
      <th>marital_Married-spouse-absent</th>
      <th>marital_Never-married</th>
      <th>...</th>
      <th>relationship_Wife</th>
      <th>sex_Female</th>
      <th>sex_Male</th>
      <th>workclass_Federal-gov</th>
      <th>workclass_Local-gov</th>
      <th>workclass_Private</th>
      <th>workclass_Self-emp-inc</th>
      <th>workclass_Self-emp-not-inc</th>
      <th>workclass_State-gov</th>
      <th>workclass_Without-pay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>38</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51</td>
      <td>9</td>
      <td>7298</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 46 columns</p>
</div>



* 최종 모델 선택 - F1 score와 accuracy가 가장 높은 모델

* Gradient Boosting Tree. n_estimators : 1000, learning_rate : 0.1 적용 (loss function = default 사용)


```python
# 학습 및 검증
```


```python
selected_model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1)
selected_model.fit(X_train, y_train)
y_pred = selected_model.predict(X_test)
model_performance(y_test, y_pred)
```

    confusion matrix
    [[ 420  209]
     [ 121 1750]]
    accuracy : 0.868
    precision : 0.776
    recall : 0.668
    F1 : 0.718
    


```python
# 예측
```


```python
y_final = selected_model.predict(X_final)
```


```python
y_final
```




    array(['under50k', 'under50k', 'over50k', ..., 'under50k', 'under50k',
           'over50k'], dtype=object)




```python
# 데이터 저장
```


```python
import numpy
numpy.savetxt('final.csv', y_final, fmt='%s')
```
