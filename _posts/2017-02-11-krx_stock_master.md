---
layout: post
title: 거래소 상회법인목록 크롤링
date:   2017-02-11
categories: python
img: 04-1-2.jpg
---

#### [FinDA] 거래소(KRX) 상장법인목록 크롤링 및 MySQL DB update

----------------------------------

### 1. 상장법인목록

http://kind.krx.co.kr/corpgeneral/corpList.do?method=loadInitPage

다운로드한 파일의 확장자가 xls 이지만 열어보면 html 이다.

![img](../images/04-1-1.jpg)

크롬 개발자도구를 통해 EXCEL 버튼 선택시 웹브라우저의 동작을 확인한다.

검사 - Network - Name (corpList.do) - Headers

* request url : http://kind.krx.co.kr/corpgeneral/corpList.do
* Form Data 항목에서 요청에 사용된 키와 값을 확인

![img](../images/04-1-2.jpg)


```python
import pandas as pd
import numpy as np
import requests
```


```python
from bs4 import BeautifulSoup
```


```python
from io import BytesIO
```


```python
def stock_master():
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    data = {
        'method':'download',
        'orderMode':'1',           # 정렬컬럼
        'orderStat':'D',           # 정렬 내림차순
        'searchType':'13',         # 검색유형: 상장법인
        'fiscalYearEnd':'all',     # 결산월: 전체
        'location':'all',          # 지역: 전체
    }

    r = requests.post(url, data=data)
    f = BytesIO(r.content)
    dfs = pd.read_html(f, header=0, parse_dates=['상장일'])
    df = dfs[0].copy()

    # 숫자를 앞자리가 0인 6자리 문자열로 변환
    df['종목코드'] = df['종목코드'].astype(np.str)   
    df['종목코드'] = df['종목코드'].str.zfill(6)
    return df
```


```python
df_master = stock_master()
df_master.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>회사명</th>
      <th>종목코드</th>
      <th>업종</th>
      <th>주요제품</th>
      <th>상장일</th>
      <th>결산월</th>
      <th>대표자명</th>
      <th>홈페이지</th>
      <th>지역</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BYC</td>
      <td>001460</td>
      <td>봉제의복 제조업</td>
      <td>메리야스,란제리 제조,도매/건축공사/부동산 임대,분양,공급</td>
      <td>1975-06-02</td>
      <td>12월</td>
      <td>유 중 화</td>
      <td>http://www.byc.co.kr</td>
      <td>서울특별시</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CJ씨푸드</td>
      <td>011150</td>
      <td>기타 식품 제조업</td>
      <td>수산물(어묵,맛살)가공품 도매,원양수산업,수출입</td>
      <td>1988-11-26</td>
      <td>12월</td>
      <td>유병철, 강신호(각자대표)</td>
      <td>http://www.cjseafood.net</td>
      <td>경기도</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DRB동일</td>
      <td>004840</td>
      <td>회사본부, 지주회사 및 경영컨설팅 서비스업</td>
      <td>고무벨트(V벨트,콘베이어벨트,평벨트),프라스틱제품 제조,판매</td>
      <td>1976-05-21</td>
      <td>12월</td>
      <td>박진삼</td>
      <td>http://drbworld.com</td>
      <td>부산광역시</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DSR제강</td>
      <td>069730</td>
      <td>1차 철강 제조업</td>
      <td>와이어로프,각종 경강선,철선제품,PC강선,아연도 강연선 제조</td>
      <td>2003-01-28</td>
      <td>12월</td>
      <td>홍하종</td>
      <td>http://www.dsrcorp.com</td>
      <td>전라남도</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GS건설</td>
      <td>006360</td>
      <td>건물 건설업</td>
      <td>토목공사,건축공사,주택공사,산업플랜트공사,리모델링공사,시설물유지관리공사/산업설비설계...</td>
      <td>1981-08-03</td>
      <td>12월</td>
      <td>허창수, 임병용(각자 대표이사)</td>
      <td>http://www.gsconst.co.kr/</td>
      <td>서울특별시</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df_master.loc[:, ['회사명', '종목코드', '업종', '상장일', '결산월']]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>회사명</th>
      <th>종목코드</th>
      <th>업종</th>
      <th>상장일</th>
      <th>결산월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BYC</td>
      <td>001460</td>
      <td>봉제의복 제조업</td>
      <td>1975-06-02</td>
      <td>12월</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CJ씨푸드</td>
      <td>011150</td>
      <td>기타 식품 제조업</td>
      <td>1988-11-26</td>
      <td>12월</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DRB동일</td>
      <td>004840</td>
      <td>회사본부, 지주회사 및 경영컨설팅 서비스업</td>
      <td>1976-05-21</td>
      <td>12월</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DSR제강</td>
      <td>069730</td>
      <td>1차 철강 제조업</td>
      <td>2003-01-28</td>
      <td>12월</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GS건설</td>
      <td>006360</td>
      <td>건물 건설업</td>
      <td>1981-08-03</td>
      <td>12월</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# 2. 거래소(KRX) 전체 종목코드 크롤링 (시가총액순위)

### 트래픽 분석

시장정보 - 주식 - 순위정보 - 시가총액 상하위 - CSV 선택

시가총액순 종목리스트는 다음과 같이 두 단계에 걸쳐 진행된다.

1. http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx
1. http://file.krx.co.kr/download.jspx  :  GenerateOTP.jspx를 통해 리턴받은 값을 입력으로 사용한다.

### 데이터 가져오기


```python
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from datetime import datetime
```


```python
def stock_master_price(date=None):
    if date == None:
        date = datetime.today().strftime('%Y%m%d')   # 오늘 날짜

    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name':'fileDown',
        'filetype':'xls',
        'url':'MKD/04/0404/04040200/mkd04040200_01',
        'market_gubun':'ALL',                          #시장구분: ALL=전체
        'indx_ind_cd':'',
        'sect_tp_cd':'',
        'schdate': date,
        'pagePath':'/contents/MKD/04/0404/04040200/MKD04040200.jsp',
    }
    
    r = requests.post(gen_otp_url, gen_otp_data)
    code = r.content  # 리턴받은 값을 아래 요청의 입력으로 사용.
    
    # STEP 02: download
    down_url = 'http://file.krx.co.kr/download.jspx'
    down_data = {
        'code': code,
    }
    
    r = requests.post(down_url, down_data)
    df = pd.read_excel(BytesIO(r.content), header=0, thousands=',')
    return df
```


```python
df = stock_master_price()
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>종목코드</th>
      <th>종목명</th>
      <th>현재가</th>
      <th>대비</th>
      <th>등락률</th>
      <th>거래량</th>
      <th>거래대금</th>
      <th>시가총액</th>
      <th>시가총액비중(%)</th>
      <th>상장주식수(천주)</th>
      <th>외국인 보유주식수</th>
      <th>외국인 지분율(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>005930</td>
      <td>삼성전자</td>
      <td>2090000</td>
      <td>-17000</td>
      <td>-0.8</td>
      <td>69044</td>
      <td>144463049000</td>
      <td>294019814330000</td>
      <td>18.35</td>
      <td>140679337</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000660</td>
      <td>SK하이닉스</td>
      <td>49300</td>
      <td>-1000</td>
      <td>-2.0</td>
      <td>1449682</td>
      <td>71990011200</td>
      <td>35890516594500</td>
      <td>2.24</td>
      <td>728002365</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>005935</td>
      <td>삼성전자우</td>
      <td>1615000</td>
      <td>-25000</td>
      <td>-1.5</td>
      <td>17531</td>
      <td>28297084000</td>
      <td>33129184605000</td>
      <td>2.07</td>
      <td>20513427</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>005380</td>
      <td>현대차</td>
      <td>149000</td>
      <td>-1000</td>
      <td>-0.7</td>
      <td>187065</td>
      <td>27902233000</td>
      <td>32821195371000</td>
      <td>2.05</td>
      <td>220276479</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>015760</td>
      <td>한국전력</td>
      <td>45650</td>
      <td>-450</td>
      <td>-1.0</td>
      <td>439486</td>
      <td>20161042650</td>
      <td>29305660115050</td>
      <td>1.83</td>
      <td>641964077</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
date = datetime(2017, 3, 3).strftime('%Y%m%d')
df = stock_master_price(date)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>종목코드</th>
      <th>종목명</th>
      <th>현재가</th>
      <th>대비</th>
      <th>등락률</th>
      <th>거래량</th>
      <th>거래대금</th>
      <th>시가총액</th>
      <th>시가총액비중(%)</th>
      <th>상장주식수(천주)</th>
      <th>외국인 보유주식수</th>
      <th>외국인 지분율(%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>005930</td>
      <td>삼성전자</td>
      <td>1981000</td>
      <td>-5000</td>
      <td>-0.2</td>
      <td>254174</td>
      <td>500689858000</td>
      <td>278685766597000</td>
      <td>18.07</td>
      <td>140679337</td>
      <td>71175695</td>
      <td>50.59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000660</td>
      <td>SK하이닉스</td>
      <td>47100</td>
      <td>-600</td>
      <td>-1.3</td>
      <td>3937032</td>
      <td>186640393450</td>
      <td>34288911391500</td>
      <td>2.22</td>
      <td>728002365</td>
      <td>358720683</td>
      <td>49.27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>005935</td>
      <td>삼성전자우</td>
      <td>1545000</td>
      <td>-22000</td>
      <td>-1.4</td>
      <td>60571</td>
      <td>93097539000</td>
      <td>31693244715000</td>
      <td>2.05</td>
      <td>20513427</td>
      <td>15857741</td>
      <td>77.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>005380</td>
      <td>현대차</td>
      <td>142000</td>
      <td>-6500</td>
      <td>-4.4</td>
      <td>774281</td>
      <td>110458431000</td>
      <td>31279260018000</td>
      <td>2.03</td>
      <td>220276479</td>
      <td>98462189</td>
      <td>44.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>015760</td>
      <td>한국전력</td>
      <td>42950</td>
      <td>-200</td>
      <td>-0.5</td>
      <td>1166862</td>
      <td>49900115200</td>
      <td>27572357107150</td>
      <td>1.79</td>
      <td>641964077</td>
      <td>199597043</td>
      <td>31.09</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    2253



# 3. 거래소(KRX) 상장회사목록 크롤링 & DB update

### 트래픽 분석

시장정보 - 주식 - 상장현황 - 상장회사검색 - CSV 선택

두 단계에 걸쳐 진행됨.

1. http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx
1. http://file.krx.co.kr/download.jspx  :  GenerateOTP.jspx를 통해 리턴받은 값을 입력으로 사용한다.

### 데이터 가져오기


```python
import pandas as pd
import numpy as np
import requests
import io
```


```python
def get_krx_stock_master():
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name':'fileDown',
        'filetype':'xls',
        'url':'MKD/04/0406/04060100/mkd04060100_01',
        'market_gubun':'ALL',                          # ALL:전체
        'isu_cdnm':'전체',
        'sort_type':'A',                               # 정렬 : A 기업명
        'std_ind_cd':'01',
        'cpt':'1',
        'in_cpt':'',
        'in_cpt2':'',
        'pagePath':'/contents/MKD/04/0406/04060100/MKD04060100.jsp',
    }  # Query String Parameters

    r = requests.post(gen_otp_url, gen_otp_data)
    code = r.content

    # STEP 02: download
    down_url = 'http://file.krx.co.kr/download.jspx'
    down_data = {
        'code': code,
    }

    r = requests.post(down_url, down_data)
    f = io.BytesIO(r.content)
    
    usecols = ['종목코드', '기업명', '업종코드', '업종']
    df = pd.read_excel(f, converters={'종목코드': str, '업종코드': str}, usecols=usecols)
    df.columns = ['code', 'name', 'sector_code', 'sector']
    return df
```


```python
df_master = get_krx_stock_master()
df_master.head()
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
      <td>060310</td>
      <td>3S</td>
      <td>032602</td>
      <td>전자부품 제조업</td>
    </tr>
    <tr>
      <th>1</th>
      <td>095570</td>
      <td>AJ네트웍스</td>
      <td>126903</td>
      <td>산업용 기계 및 장비 임대업</td>
    </tr>
    <tr>
      <th>2</th>
      <td>068400</td>
      <td>AJ렌터카</td>
      <td>126901</td>
      <td>운송장비 임대업</td>
    </tr>
    <tr>
      <th>3</th>
      <td>006840</td>
      <td>AK홀딩스</td>
      <td>137105</td>
      <td>회사본부, 지주회사 및 경영컨설팅 서비스업</td>
    </tr>
    <tr>
      <th>4</th>
      <td>054620</td>
      <td>AP시스템</td>
      <td>032902</td>
      <td>특수 목적용 기계 제조업</td>
    </tr>
  </tbody>
</table>
</div>




```python
import mysql.connector
from sqlalchemy import create_engine
```


```python
con_str = 'mysql+mysqlconnector://woosa7:finda888@localhost/findb'
engine = create_engine(con_str, echo=False)
```


```python
insert_update_sql = """
    insert into stock_master (code, name, sector_code, sector) 
    values (%s,%s,%s,%s)
    on duplicate key update
        code=values(code),
        name=values(name),
        sector_code=values(sector_code),
        sector=values(sector)
"""
```


```python
for ix, r in df_master.iterrows():
    engine.execute(insert_update_sql, (r['code'], r['name'], r['sector_code'], r['sector']))
    print(r['code'], r['name'])
```

    060310 3S
    095570 AJ네트웍스
    068400 AJ렌터카
    006840 AK홀딩스
    054620 AP시스템
    211270 AP위성
    027410 BGF리테일
    138930 BNK금융지주
    001460 BYC
    032040 C&S자산관리
    001040 CJ
    ......
    010240 흥국
    189980 흥국에프엔비
    000540 흥국화재
    003280 흥아해운
    037440 희림
    
