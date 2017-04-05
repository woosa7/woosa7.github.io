---
layout: post
title: cloud9에 MySQL 환경 설정
date:   2017-02-05
categories: python
img: c9_mysql_50_1.png
---

#### [FinDA] Cloud9에 MySQL DB 설정 및 기초 데이터 수집

----------------------------------

[참조 블로그](https://nbviewer.jupyter.org/gist/FinanceData/01623efe62e9175da1869bf9901a942c)

## 1. MySQL 설정

### 1.1. MySQL 시작

```bash
$ mysql-ctl start
```

### 1.2. admin 계정과 패스워드 설정
MySQL cli 실행 하여 admin 계정과 패스워드 설정

```bash
$ mysql-ctl cli

mysql> GRANT ALL PRIVILEGES ON *.* to 'admin'@'%' IDENTIFIED BY 'finda***';
mysql> exit
```

### 1.3. utf-8 인코딩 설정
* utf-8이 기본이 되도록 만들어 놓은 설정 파일 my.cnf [링크](https://www.dropbox.com/s/uzuhb28u363i597/my.cnf?dl=0)
* my.cnf 을 다운로드 하여, /etc/mysql/ 에 복사하고, mysql을 재시동

```bash
$ wget https://www.dropbox.com/s/uzuhb28u363i597/my.cnf?dl=0 -O my.cnf
$ sudo cp my.cnf /etc/mysql/my.cnf
$ mysql-ctl restart

```

## 2. sqlalchemy 와 mysql-connector 설치

### 2.1 sqlalchemy 설치

```bash
$ sudo pip3 install sqlalchemy
```

### 2.2 mysql-connector 설치

```bash
$ git clone https://github.com/mysql/mysql-connector-python.git
$ cd mysql-connector-python
$ python3 ./setup.py build
$ sudo python3 ./setup.py install
```

설치 파일들 삭제
```bash
$ cd ..
$ rm -rf mysql-connector-python/
```

## 3. 초기 데이터 구축

### 3.1 작업 디렉토리 확인
```bash
$ cd ~/workspace/
$ pwd
/home/ubuntu/workspace
```

## 3.2 git 저장소 클론 (.py 프로그램들)
```bash
$ git clone https://gist.github.com/d103b7677376030fd24de8f87ca59de7 findb 
$ cd findb
```
(참고) 만일 이미 git clone 했다면 아래와 같이 업데이트.

```bash
$ cd ~/workspace/findb
$ git pull
```

### 3.3 MySQL 데이터 다운로드 및 restore

```bash
$ wget -O findb_dump.sql.gz "https://www.dropbox.com/s/1f44sylm9b32jqi/findb_dump.sql.gz?dl=0"
$ gunzip < findb_dump.sql.gz | mysql -u admin -p
Enter password:  (MySQL에 설정한 admin 패스워드 입력)  
(약 2~3분 소요)
```

### 3.4 데이터 저장 확인
MySQL DB 데이터 확인

```bash
$ mysql -u admin -p
Enter password: 
mysql> use findb;
mysql> select count(*) from stock_dart;     # (180만개 보고서 링크)
```

## 4. 추가 데이터 채우기

### 4.1 작업 디렉토리 확인
```bash
$ cd ~/workspace/findb
$ pwd
/home/ubuntu/workspace/findb
```

### 4.2 추가 데이터 수집
저장된 데이터 이후 추가 데이터 수집
```bash
# 전자공시 보고서 데이터 추가 (약 5분 소요)
$ ./stock_dart.py 

# (전종목) 가격 데이터 수집  (약 2.5~3시간 소요)
$ ./stock_price.py 
```

## 5. 서버에서 jupyter notebook 실행

c9.io 터미널에서 jupyter notebook 를 실행. 

```bash
$ jupyter notebook --ip=0.0.0.0 --port=8080 --no-browser
```

## 6. 기본적인 분석


```python
import pandas as pd
import numpy as np
from pandas_datareader import data, wb
from datetime import datetime
```

### 6.1 MySQL에 연결


```python
import mysql.connector
from sqlalchemy import create_engine
import getpass
```


```python
pwd = getpass.getpass()
engine = create_engine('mysql+mysqlconnector://admin:'+pwd+'@localhost/findb', echo=False)
```

    ········
    

### 6.2 테이블 목록과 스키마 확인


```python
sql = 'show tables'
df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tables_in_findb</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stock_dart</td>
    </tr>
    <tr>
      <th>1</th>
      <td>stock_desc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stock_finstate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>stock_master</td>
    </tr>
    <tr>
      <th>4</th>
      <td>stock_price</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql = 'desc stock_master'
#sql = 'desc stock_desc'
#sql = 'desc stock_finstate'
#sql = 'desc stock_price'
#sql = 'desc stock_dart'

df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Field</th>
      <th>Type</th>
      <th>Null</th>
      <th>Key</th>
      <th>Default</th>
      <th>Extra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>code</td>
      <td>varchar(20)</td>
      <td>NO</td>
      <td>PRI</td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>name</td>
      <td>varchar(50)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>sector_code</td>
      <td>varchar(30)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>sector</td>
      <td>varchar(80)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



### 6.3 간단한 조회 예제


```python
# 전체 종목수
sql = 'select count(*) from stock_master'
df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2072</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql = "select * from stock_master where code='005930'"
df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>name</th>
      <th>sector_code</th>
      <th>sector</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>005930</td>
      <td>삼성전자</td>
      <td>032604</td>
      <td>통신 및 방송 장비 제조업</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql = "select `name`,`wics`,`address`,`desc` from stock_desc where code='005930'"
df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>wics</th>
      <th>address</th>
      <th>desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>삼성전자</td>
      <td>반도체와반도체장비</td>
      <td>경기도 수원시 영통구 삼성로 129(매탄동)</td>
      <td>1969년 설립되어, 본사를 거점으로 한국 및 CE, IM 부문 산하 해외 9개 지...</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql = """
    select m.code, m.name, m.sector, d.wics, d.desc
    from stock_master m, stock_desc d
    where m.code=d.code
"""
df = pd.read_sql(sql, con=engine)
print(len(df))
df.head()
```

    2040
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>name</th>
      <th>sector</th>
      <th>wics</th>
      <th>desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000020</td>
      <td>동화약품</td>
      <td>의약품 제조업</td>
      <td>제약</td>
      <td>동사는 1897년 설립된 국내 최초 제약기업으로, 주 사업내용으로는 의약품, 원료의...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000030</td>
      <td>우리은행</td>
      <td>은행 및 저축기관</td>
      <td>은행</td>
      <td>1899년 설립되어 2002년 우리은행으로 사명을 변경하였으며, 은행법에 의한 은행...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000040</td>
      <td>KR모터스</td>
      <td>그외 기타 운송장비 제조업</td>
      <td>자동차</td>
      <td>동사는 2014년 03월 19일자로 코라오그룹으로 편입되었으며, 이륜차 제조사업을 ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000050</td>
      <td>경방</td>
      <td>방적 및 가공사 제조업</td>
      <td>섬유,의류,신발,호화품</td>
      <td>동사는 1919년 10월 5일 설립하여 면사, 생사, 화섬사 등 각종사 및 면직물,...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000060</td>
      <td>메리츠화재</td>
      <td>보험업</td>
      <td>손해보험</td>
      <td>1922년 국내 최초로 설립된 종합손해보험회사로, 메리츠금융그룹 기업집단에 속해있으...</td>
    </tr>
  </tbody>
</table>
</div>



## 7. matplotlib 기본 설정

### 나눔 글꼴 설치
```bash
$ sudo apt-get install fonts-nanum*
$ sudo fc-cache -f -v
```

### matplotlib 나눔 글꼴을 추가
나눔 글꼴을 matplotlib 에 복사하고, matplotlib의 폰트 cache를 clear 
```bash
$ sudo cp /usr/share/fonts/truetype/nanum/Nanum* /usr/local/lib/python3.4/dist-packages/matplotlib/mpl-data/fonts/ttf/
$ rm -rf /home/ubuntu/.cache/matplotlib/*
```


```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'nanummyeongjo'
plt.rcParams["font.size"] = 8
plt.rcParams["figure.figsize"] = (14,4)
```


```python
# 한글 표시 확인
plt.ylabel('가격', size=20)
plt.xlabel('범위', size=20)
plt.show()
```


![png](../images/c9_mysql_33_0.png)


## 8. 수집한 데이터 활용 예

업종 분류 (거래소)


```python
sectors = df.groupby('sector')['code'].count()
sectors = sectors.sort_values(ascending=True)
sectors[-20:].plot(kind='barh', figsize=(14,10))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16cf48f9e8>




![png](../images/c9_mysql_36_1.png)


업종 분류 (wics)


```python
sectors = df.groupby('wics')['code'].count()
sectors = sectors.sort_values(ascending=True)
sectors[-20:-1].plot(kind='barh', figsize=(14,8))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16cb700710>




![png](../images/c9_mysql_38_1.png)


전자 공시 최근 10개


```python
sql = "select corp_name, title, link from stock_dart order by date desc limit 10"
df = pd.read_sql(sql, con=engine)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corp_name</th>
      <th>title</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>나라케이아이씨</td>
      <td>감사보고서제출</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>에이티테크놀러지</td>
      <td>감사보고서제출</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>화승엔터프라이즈</td>
      <td>감사보고서제출</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>나이스정보통신</td>
      <td>감사보고서제출</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서연</td>
      <td>감사보고서제출(자회사의 주요경영사항)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NICE</td>
      <td>감사보고서제출(자회사의 주요경영사항)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>신양오라컴</td>
      <td>주주총회소집결의</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>이디</td>
      <td>최대주주변경을수반하는주식담보제공계약체결</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>바른손이앤에이</td>
      <td>[기재정정]회사합병결정(종속회사의주요경영사항)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>크로넥스</td>
      <td>[기재정정]주주총회소집결의</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
  </tbody>
</table>
</div>



2016년 1월 이후 삼성전자 주별 공시 건수 카운드


```python
sql = "SELECT date FROM stock_dart WHERE corp_name='삼성전자' and date >= '2016-01'"

df = pd.read_sql(sql, con=engine)
df.set_index(pd.to_datetime(df['date']), inplace=True)

df['n']  = 1
df['n'].resample('W').sum().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16cb7b4e48>




![png](../images/c9_mysql_42_1.png)


제목에 "보고서" (각종 정기 사업 보고서등)를 포함하는 삼성전자 공시


```python
sql = "SELECT date, title, link FROM stock_dart WHERE corp_name='삼성전자' and title like '%보고서%'"

df = pd.read_sql(sql, con=engine)
print ('count = ', len(df))
df.tail()
```

    count =  184
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>title</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>2016-11-04 16:10:00</td>
      <td>합병등종료보고서 (분할)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2016-11-14 16:12:00</td>
      <td>분기보고서 (2016.09)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2017-01-24 08:53:00</td>
      <td>주요사항보고서(자기주식취득결정)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2017-01-24 17:25:00</td>
      <td>[기재정정]주요사항보고서(자기주식취득결정)</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
    <tr>
      <th>183</th>
      <td>2017-02-28 16:36:00</td>
      <td>감사보고서제출</td>
      <td>http://dart.fss.or.kr/dsaf001/main.do?rcpNo=20...</td>
    </tr>
  </tbody>
</table>
</div>



제무재표


```python
# 삼성전자 현금배당수익률, 현금배당성향(%)

sql = "select * from stock_finstate where code = '005930' and freq_type='Y'"
df = pd.read_sql(sql, con=engine, index_col='date')
df_dividend = df[['현금배당수익률', '현금배당성향(%)']].dropna()
df_dividend 
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>현금배당수익률</th>
      <th>현금배당성향(%)</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-12</th>
      <td>0.53</td>
      <td>5.20</td>
    </tr>
    <tr>
      <th>2012-12</th>
      <td>1.04</td>
      <td>7.23</td>
    </tr>
    <tr>
      <th>2013-12</th>
      <td>1.51</td>
      <td>13.00</td>
    </tr>
    <tr>
      <th>2014-12</th>
      <td>1.67</td>
      <td>16.42</td>
    </tr>
    <tr>
      <th>2015-12</th>
      <td>1.60</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2016-12</th>
      <td>1.69</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2017-12</th>
      <td>1.77</td>
      <td>0.17</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['code', 'fin_type', 'freq_type', '매출액', '영업이익', '세전계속사업이익', '당기순이익',
           '당기순이익(지배)', '당기순이익(비지배)', '자산총계', '부채총계', '자본총계', '자본총계(지배)',
           '자본총계(비지배)', '자본금', '영업활동현금흐름', '투자활동현금흐름', '재무활동현금흐름', 'CAPEX', 'FCF',
           '이자발생부채', '영업이익률', '순이익률', 'ROE(%)', 'ROA(%)', '부채비율', '자본유보율',
           'EPS(원)', 'PER(배)', 'BPS(원)', 'PBR(배)', '현금DPS(원)', '현금배당수익률',
           '현금배당성향(%)', '발행주식수(보통주)'],
          dtype='object')




```python
# 삼성전자 분기별 매출액, 당기순이익
```


```python
sql = "select * from stock_finstate where code = '005930' and freq_type = 'Y'"

df = pd.read_sql(sql, con=engine, index_col='date')
df = df[['매출액', '당기순이익']].dropna()
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>매출액</th>
      <th>당기순이익</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-12</th>
      <td>2011036.0</td>
      <td>238453.0</td>
    </tr>
    <tr>
      <th>2012-12</th>
      <td>2286927.0</td>
      <td>304748.0</td>
    </tr>
    <tr>
      <th>2013-12</th>
      <td>2062060.0</td>
      <td>233944.0</td>
    </tr>
    <tr>
      <th>2014-12</th>
      <td>2006535.0</td>
      <td>190601.0</td>
    </tr>
    <tr>
      <th>2015-12</th>
      <td>2064953.0</td>
      <td>233667.0</td>
    </tr>
    <tr>
      <th>2016-12</th>
      <td>2127383.0</td>
      <td>249384.0</td>
    </tr>
    <tr>
      <th>2017-12</th>
      <td>2204236.0</td>
      <td>263336.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['매출액'].plot()
df['당기순이익'].plot(secondary_y=True, color='c')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16cb6a2828>




![png](../images/c9_mysql_50_1.png)

