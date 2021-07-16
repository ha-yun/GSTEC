# 2주차
## <1>
* 박준영 강사님  
1. Quick,Draw  
2. Teachable machine
<hr>

## <2>
* 간단한 토너먼트 코딩
```python
import random
t=[1,2,3,4,5,6,7,8,9]
a=[]
while len(t)>0:
  one=random.choice(t)
  t.remove(one)
  if len(t)==0:
    a.append(one)
    break
  two=random.choice(t)
  t.remove(two)
  a.append((one,two))
print(a)
```
```python
teams=list(range(1,10))
random.shuffle(teams)
half=len(teams)//2
print(teams[:half], teams[half:])
print(teams)
print(list(zip(teams[:half],teams[half:])))
```
* html 배포 : https://app.netlify.com/teams/yun-aha/overview  
  or 깃헙 연동  
  web을 배포하면 default로 찾는 폴더 = index.html
  
<hr>

## <3>
### 머신러닝(Machine Learning) 종류
<b>
1. 지도학습 (Supervised Learning)<br>
2. 비지도학습 (Unsupervised Learning)<br>
3. 강화학습 (Reinforcement Learning)
</b>


1. 지도학습 (Supervised Learning)
- 데이터에 대한 Label(명시적인 답)이 주어진 상태에서 컴퓨터를 학습시키는 방법.
- 분류(Classification)와 회귀(Regression)로 나뉘어진다.
> (ex. 스팸 메일 분류, 집 가격 예측, 손글씨 숫자 판별, 신용카드 의심거래 감지, 의료영상 이미지기반 종양판단)

2. 비지도학습(Unsupervised Learning)  
- 데이터에 대한 Label(명시적인 답)이 없는 상태에서 컴퓨터를 학습시키는 방법.
- 데이터의 숨겨진 특징, 구조, 패턴 파악.
- 데이터를 비슷한 특성끼리 묶는 클러스터링(Clustering)과 차원축소(Dimensionality Reduction)등이 있다.
> (ex. 블로그 글 주제구분, 고객 취향별 그룹화, 웹사이트 비정상 접근 탐지, 이미지 감색 처리, 소비자 그룹 마케팅)

{좋은 입력 데이터를 만들어내는 방법 -> 특성추출(특성공학) }

3. 강화학습(Reinforcement Learning)  
- 지도학습과 비슷하지만 완전한 답(Label)을 제공하지 않는 특징이 있다.
- 기계는 더 많은 보상을 얻을 수 있는 방향으로 행동을 학습
> (ex. 게임이나 로봇 학습)


머신러닝 vs 딥러닝

|구분| Machine Learning| Deep Learning|
|---|---|---|
|훈련 데이터 크기| 작음| 큼|
|시스템 성능| 저 사양| 고 사양|
|feature 선택| 전문가 (사람) |알고리즘|
|feature 수| 많음 |적음|
|문제 해결 접근법| 문제를 분리 -> 각각 답을 얻음 -> 결과 통합| end-to-end (결과를 바로 얻음)|
|실행 시간| 짧음 |김|
|해석력 |해석 가능| 해석 어려움|

### [scikit-learn](https://scikit-learn.org/stable/index.html)
- 파이썬에 머신러닝 프레임워크 라이브러리
- 회귀, 분류, 군집, 차원축소, 특성공학, 전처리, 교차검증, 파이프라인 등 머신러닝에 필요한 기능 제공
- 학습을 위한 샘플 데이터 제공  

#### scikit-learn으로 XOR 연산 학습해보기
XOR연산?
- 두값이 서로 같으면 0, 다르면 1  (배타적 논리 합)

|P(입력)| Q(입력)| R(출력)|
|---|---|---|
| 0| 0| 0|
| 0| 1| 1|
| 1| 0| 1|
| 1| 1| 0|

```python
from sklearn import svm
# XOR의 계산 결과 데이터
xor_input=[
           # P,Q,REsult
           [0,0,0],
           [0,1,1],
           [1,0,1],
           [1,1,0]
]

# 1. 학습을 위해 데이터와 레이블 분리하기
xor_data=[]
xor_label=[]

for i in xor_input:
  xor_data.append([i[0],i[1]])
  xor_label.append(i[-1])
print(xor_data)
print(xor_label)

# or
for [p,q,r] in xor_input:
  xor_data.append([p,q])
  xor_label.append(r)
  
# 2. 데이터 학습시키기
model = svm.SVC()  #svm.SVC() -> support vector machine은 클래스가 두 개있는 이항의 분류 문제를 위해 만들어졌다.
model.fit(xor_data,xor_label)
   
# 3. 데이터 예측하기
pre=model.predict(xor_data)
print(pre)
   
# 4. 결과 확인하기
ok = 0
for idx,answer in enumerate(xor_label):
  p=pre[idx]
  if p==answer:ok+=1
print('정답률 : ',ok,'/',4, '=',ok/4)
   ```
   
```python
### pandas 라이브러리를 사용하여 코드 간략화
import pandas as pd
from sklearn import svm, metrics

# XOR연산
xor_input=[
           # P,Q,REsult
           [0,0,0],
           [0,1,1],
           [1,0,1],
           [1,1,0]
]

# 1. 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류하기
xor_df = pd.DataFrame(xor_input)
xor_data = xor_df[[0,1]]
xor_label = xor_df[2]

# 2. 데이터 학습과 예측하기
model = svm.SVC()
model.fit(xor_data, xor_label)
pre = model.predict(xor_data)

#3. 정답률 구하기
ac_score = metrics.accuracy_score(xor_label, pre)
print(ac_score)
```

```python
### KNN 분류 모델을 이용
import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier  ##분류 모델 추가

# XOR연산
xor_input=[
           # P,Q,REsult
           [0,0,0],
           [0,1,1],
           [1,0,1],
           [1,1,0]
]

# 1. 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류하기
xor_df = pd.DataFrame(xor_input)
xor_data = xor_df[[0,1]]
xor_label = xor_df[2]

# 2. 데이터 학습과 예측하기
model = KNeighborsClassifier(n_neighbors=1)
model.fit(xor_data, xor_label)
pre = model.predict(xor_data)

# 3. 정답률 구하기
ac_score = metrics.accuracy_score(xor_label, pre)
print(ac_score)
```
   
진행순서

>1. clf = 머신러닝모델 생성  # svm.SVC() or KNeighborsClassifier(n_neighbors=1)
>2. clf.fit(문제 , 답)
>3. 예측결과 = clf.predict(값을 얻고 싶은 데이터 )
>4. ac_score = metrics.accuracy_score(실제답, 예측결과)

clf (classifier) - scikit-learn 에서 [Estimator](https://en.wikipedia.org/wiki/Estimator) 인스턴스인 분류기를 지칭  

[머신러닝 용어집](https://developers.google.com/machine-learning/glossary)

### 모델 저장과 불러오기
1. pickle --> load(불러오기)와 dump(입력)  
객체를 파일로 피클링, 파일을 객체로 언피클링
```python
import pickle  
with open('xor_model.pkl','wb') as f:   #wb는 write binary
  pickle.dump(model,f)  
import pickle
with open('xor_model.pkl','rb') as f:   #rb는 read binary
  model = pickle.load(f)
pre = model.predict([[1,1],[1,0]])
print(pre)
pre[0],pre[1]
   ```
2. joblib -->load와 dump  
```python
from sklearn.externals import joblib
joblib.dump(model,'xor_model_2.pkl')
model = joblib.load('xor_model_2.pkl')
```
   

#### scikit-learn 연습 01
AND 연산 모델 작성  
AND연산?
- 두값이 서로 참이면 1, 아니면 0 

|P(입력)| Q(입력)| R(출력)|
|---|---|---|
| 0| 0| 0|
| 0| 1| 0|
| 1| 0| 0|
| 1| 1| 1|

```python
### KNN 분류 모델을 이용
import pandas as pd
from sklearn import svm, metrics
from sklearn.neighbors import KNeighborsClassifier  ##분류 모델 추가

# and연산
and_input=[
           # P,Q,REsult
           [0,0,0],
           [0,1,0],
           [1,0,0],
           [1,1,1]
]

# 1. 입력을 학습 전용 데이터와 테스트 전용 데이터로 분류하기
and_df = pd.DataFrame(and_input)
and_data = and_df[[0,1]]
and_label = and_df[2]

# 2. 데이터 학습과 예측하기
model = KNeighborsClassifier(n_neighbors=1)
model.fit(and_data, and_label)
pre = model.predict(and_data)

# 3. 정답률 구하기
ac_score = metrics.accuracy_score(and_label, pre)
print(ac_score)
```
   
### 구글 드라이브 연동
```python
from google.colab import drive
drive.mount('/gdrive', force_remount=True)
# 구글 드라이브 파일 확인
!ls '/gdrive/My Drive/temp/'
# 반복되는 드라이브 경로 변수화
drive_path = '/gdrive/My Drive/temp/'
```
  
<hr>

# 3주차
## <1>
#### scikit-learn 연습 02

비만도 데이터 학습
- 500명의 키와 몸무게, 비만도 라벨을 이용해 비만을 판단하는 모델을 만들어보자.
```python
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics # 평가를 위한 모듈
df = pd.read_csv(drive_path + 'bmi_500.csv', index_col='Label')
df.head()
df.info()
df.index.unique()
df.loc['Normal']
```
```python
def easy_scatter(label,color):
  t = df.loc[label]
  plt.scatter(t['Weight'],t['Height'],color=color,label=label)

plt.figure(figsize=(5, 5) )
easy_scatter('Extreme Obesity','black')
easy_scatter('Weak','blue')
easy_scatter('Normal','green')
easy_scatter('Overweight','pink')
easy_scatter('Obesity','purple')
easy_scatter('Extremely Weak','red')

plt.legend()
plt.show()
```

### 모델링
1. 문제와 답으로 분리
2. 훈련셋과 평가셋으로 분리
3. 모델생성 및 하이퍼파라미터 조정
4. 학습 및 평가

```python
data=pd.read_csv(drive_path + 'bmi_500.csv')

X=data.loc[:,'Height':'Weight']  #숫자로 가져오고 싶으면 iloc, 문자면 loc
y=data.loc[:,'Label']
print(X.shape)
print(y.shape)

X_train = X.iloc[:350, :]
X_test = X.iloc[350:, :]
y_train = y.iloc[:350]
y_test = y.iloc[350:]

bmi_model = KNeighborsClassifier(n_neighbors=10)
bmi_model.fit(X_train, y_train)
pre=bmi_model.predict(X_test)
metrics.accuracy_score(y_test,pre)

bmi_model.predict([[185, 43], [100,20]])
```

### 머신러닝(Machine Learning) 진행 과정
<b>
1. ProblemIdentification (문제정의) <br>
2. Data Collect(데이터 수집) <br>
3. Data Preprocessing(데이터 전처리) <br>
4. EDA(탐색적 데이터분석) <br>
5. Model 선택, Hyper Parameter 조정 <br>
6. 학습 <br>
7. 모델 Evaluation(평가)
</b>

1. 문제정의
 - 지도학습 : 분류, 회귀
 - 비지도학습 : 군집, 차원축소
 - 강화학습

2. 데이터 수집
 - File Data, Database, 공공데이터, kaggle
 - Web Crawler (뉴스, SNS, 블로그)
 - IoT 센서를 통한 수집

3. 데이터 전처리
 - 결측치, 이상치 수정
 - Encoding : Categorical Data를 수치 데이터로 변경, 원핫인코딩
 - Feature Engineering (특성공학) : 단위 변환, 새로운 속성 추가 (MinMaxScaler, StandardScaler, RobustScaler)

4. EDA
 - 시각화를 통해 특성 선택 : (scatterplot, pairplot, boxplot, heatmap)
 - 사용할 Feature 선택 : 전처리 전략수립

5. Model 선택, Hyper Parameter 조정
 - 목적에 맞는 적절한 모델 선택
  - 지도학습
	 - 분류 : knn, Logistic Regression, SVM, Decision Tree, RandomForest, GradientBoosting
	 - 회귀 : knn, Linear Regression, Lasso, Ridge, Decision Tree, RandomForest, GradientBoosting
 - 하이퍼파라미터 튜닝

6. 학습
 - model.fit(X_train, y_train) : train 데이터와 test 데이터를 7:3 정도로 나눔 (train_test_split)
 - model.predict(X_test) :  (cross_val_score)

7. 평가
 -	지도학습
	 - 분류 : 정확도, 정밀도, 재현율, f1-score
	 - 회귀 : R^2, MSE, RMSE

 -	비지도학습
	- ARI 값
      
### moral machine ,, codeorg

<hr>

## <2> 
## 데이터 수집
### 수집 데이터 형태
* 정형 – 일정한 규격에 맞춰서 구성된 데이터 (어떠한 역할을 알고 있는 데이터)
    - 관계형 데이터베이스 시스템의 테이블과 같이 고정된 컬럼에 저장되는 데이터 파일 등이 될 수 있다.
     즉, 구조화 된 데이터가 정형 데이터

* 반정형 – 일정한 규격으로 구성되어 있지 않지만 일정한 틀을 갖추기 위해서 태그나 인덱스형태로 구성된 데이터
    - 연산이 불가능한 데이터 ex) XML. HTML, JSON 등

* 비정형 – 구조화 되지 않는 형태의 데이터 (정형과 반대로 어떠한 역할인지 알수 없는 데이터)
    - 형태가 없으며, 연산도 불가능한 데이터 ex) SNS, 영상, 이미지, 음성, 텍스트 등
    >우리가 주로 수집할 데이터들은 반정형 혹은 비정형 데이터라고 보면 된다.

### 스크레이핑, 크롤링
- Scraping: 웹 사이트의 특정 정보를 추출하는 것. 웹 데이터의 구조 분석이 필요
- 로그인이 필요한 경우가 많다
- Crawling: 프로그램이 웹사이트를 정기적으로 돌며 정보를 추출하는 것 (이러한 프로그램을 크롤러, 스파이더라고 한다)

### 웹 크롤러 (Web Crawler)
* 인터넷에 있는 웹 페이지로 이동해서 데이터를 수집하는 프로그램
* 크롤러 = 스크래퍼, 봇, 지능 에이전트, 스파이더 등으로 불림

### 웹 크롤링을 위해 알아둘 것
* Web 구조
    - 클라이언트 = 서버에 정보 또는 서비스를 요청하는 컴퓨터/소프트웨어/사용자 ex) 웹브라우저
    - 서버 = 정보를 보관하고 클라이언트가 요청한 정보 서비스를 제공해주는 컴퓨터/소프트웨어 ex) 영상, 파일, 채팅, 게임, 웹서버

* URL 구조
    - http://192.168.0.10:9000/WebProject?msg=Hello
    - 웹 문서를 교환하기 위한 규칙
    - 주소 또는 IP
    - 포트번호
    - 리소스 경로
    - 쿼리스트링

* 데이터 전송 방식
    - GET, POST  (대표적인 2가지)

* 패킷(Packet) 형식
    - 요청패킷: 클라이언트에서 필요한 헤더 Key/Value를 세팅한 후 요청, 전달
    -응답패킷: 서버에 필요한 Key/Value를 세팅한 후, 응답, 전달
      
* URI는 인터넷에 있는 자원을 나타내는 유일한 주소다. Uniform Resoure Identifier  
  (하위개념으로 url, urn이 있다.)
* url  :  웹  기본 80포트  
* w3school.com : W3스쿨즈는 온라인으로 웹 기술을 배우는 교육용 웹 사이트이다


### 웹 페이지 구성 3요소
* HTML, CSS, Javascript 
  * HTML: Tag, Element, Attribute, Content
  * CSS : 선택자 { 스타일 속성 : 스타일 값; }
    - 선택자(셀렉터) : tag, id, class
  * Javascript :Web Page에서 어떤 동작에 대한 반응이 일어날 수 있도록 해주는 언어
     - DOM (Document Object Model) : 문서를 객체로 조작하기 위한 표준 모델. HTML 문서를 객체로 표현할 때 사용하는 API
    

* 데이터 추출 방법
    -  웹 페이지를 방문하고 크롤링하려는 단어나 문구를 검색하여 찾아낸 정보를 수집하여 기록
    - 이때 파일로 기록해서 저장하거나 데이터베이스에 저장 (저장하는 과정에서 단어나 문구와 연관된 결과 값에 인덱스를 부여)

* 웹 크롤링을 위한 라이브러리   
    -requests: 접근할 웹 페이지의 데이터를 요청/응답받기 위한 라이브러리
    - BeautifulSoup : 응답받은 데이터 중 원하는 데이터를 추출하기 위한 라이브러리

### urllib 사용법
- url 관련 데이터를 처리하는 라이브러리
- http 또는 ftp를 사용해 데이터를 다운로드 받는데 사용

#### 웹에서 파일 다운로드하기
```python
import requests as req
res = req.get('https://www.google.com')  #response 200번대면 잘 연결된것. 400이면 주소 오류일 수도 있다.
res.text
url='https://www.naver.com'
res=req.get(url)
res   #response[200]

# 다른프로그램에서 링크를 들어가는게 안되는 페이지
    ## headers로 해결
headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        'referer':'https://nid.naver.com/login/sso/finalize.nhn?url=https%3A%2F%2Fwww.naver.com&sid=KBwWbyEJdJL521l3&svctype=1'}

url = 'https://prod.danawa.com/list/?cate=11229515&logger_kw=ca_main_more'
res = req.get(url, headers=headers)
res.text

# json 읽는법
import json
from pandas.io.json import json_normalize
url = 'http://rank.search.naver.com/rank.js'
res = req.get(url)
json_normalize(json.loads(res.text), ['data', 'data'])


# 이미지 가져오기
import urllib.request
url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
urllib.request.urlretrieve(url, 'test.png')
```
  
#### urlopen() 사용법
- 위의 urlretrive()는 데이터를 파일에 바로 저장하였다.
- urlopen()을 사용하면 데이터를 파이선에서 읽을 수 있다.
```python
png = urllib.request.urlopen(url).read()  
with open('test2.png','wb')as f:
  f.write(png)
```
  
#### 웹 API 이용하기
- 클라이언트정보를 보여주는 샘플 api 사이트 접속
```python
url = "http://api.aoikujira.com/ip/ini"  
res = urllib.request.urlretrieve(url)
```
  
####  GET  요청을 사용하여 파라미터를 보내는 경우
- URL 끝 부분에 ?를 입력하고 key = value 형식으로 매개변수를 추가한다. 여러개의 파라미터를 넣는 경우 &를 사용하여 구분한다
- 한글 등이 파라미터로 사용될 때는 반드시 이러한 코딩을 해주어야 한다

```python
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
res = urllib.request.urlopen(url)
data = res.read()
text = data.decode('utf-8')
text
```

### 스크레이핑 : 웹에서 원하는 정보를 추출하는 것  
HTML과 XML 문서에서 정보를 추출할 수 있다

### BeautifulSoup 라이브러리
HTML 문자열을 파이썬에서 사용 가능한 객체로 변환
BeautifulSoup( markup, parser )
* markup : requests로 요청/응답 받은 변수
* parser : 원시 코드인 순수 문자열 객체를 해석할 수 있도록 분석  
 - 문자열을 파이썬에서 사용할 수 있도록 해석해주는 프로그램 (lxml, html.parser, html5lib)

데이터를 추출하기 위한 속성

|속성|설명|
|---|---|
|text|하위 태그에 대한 값 전부 출력|
|string|정확히 태그에 대한 값만 출력|

원하는 요소 접근하는 메소드

|메소드|설명|
|---|---|
|find(tag), find(tag, id=값), find(tag, class=값), 
find(tag, attr{속성:속성값})|원하는 태그를 하나만 반환|
|find_all(),
 find_all(tag, limit=숫자)|원하는 태그를 리스트 형태로 반환|
|select(CSS Selector)|CSS Selector를 활용하여 원하는 태그를 리스트 형태로 반환|
|extract()|태그를 지우는 기능|


##### !pip3 install beautifulsoup4 -> beautifulsoup설치하는 방법
```python
from bs4 import BeautifulSoup
html="""html,body,h,p tag""
soup = BeautifulSoup(html, 'html.parser')
h1 = soup.html.body.h1
p1 = soup.html.body.p
p2 = p1.next_sibling.next_sibling
print(h1)
print(p1)
print(p2)
```

#### id 를 사용하는 방법
- 위와 같이 내부 구조를 일일이 파악하고 코딩하는 것은 복잡하다
- find()를 사용하여 간단히 원하는 항목을 찾을 수 있다
```python
from bs4 import BeautifulSoup
html="""h1 id='title', p id='p1', p id='p2'""  
soup = BeautifulSoup(html, 'html.parser')
h1 = soup.find(id='title')
p1 = soup.find(id='p1')
p2 = soup.find(id='p2')
print(h1.string)
print(p1)
print(p2)
```

#### find_all()을 이용하는 경우
```python
html='''html,body
  ul
    li a href='http://www.naver.com' naver 
    li a href='http://www.daum.net' daum  
  ul
'''''
soup=BeautifulSoup(html, 'html.parser')
links = soup.find_all('a')
for a in links:
  href = a.attrs['href']
  print(href)
print(links[0].string)
links[0].attrs    #attribute약자
```

### DOM 요소 파악하기
- Document Object Model: XML이나 HTML 요소에 접근하는 구조를 나타낸다
- DOM 요소의 속성이란 태그 뒤에 나오는 속성을 말한다 < a > 태그의 속성은 href이다
```python
soup.html.body.ul
soup.prettify()     #<html>\n <body>\n  <ul>\n   <li>\n   ...
```
### urlopen() 사용 하기
```python
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
res = urllib.request.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')
title = soup.find('title').string
wf = soup.find('wf').string
print(title)
print('-'*30)
print(wf)
```

### CSS 선택자 사용하기
- CSS 선택자를 사용해서 원하는 요소를 추출할 수 있다.
- h1 과 li 태그를 추출하는 코드
```python
  html='''
html,body
  div id='books'
    h1 위키북스 도서 /h1
    ul class='item'
      li 게임 입문 /li
      li 파이썬 입문 /li
      li 웹 디자인 입문 /li
    ul
  div
'''
soup = BeautifulSoup(html, 'html.parser')
h1 = soup.select_one('div#books > h1')            #id는 샵으로 표시
li_list = soup.select('div#books > ul.item > li')
print(h1)
print(li_list)
```
### CSS 자세히 알아보기
- 웹 페이지의 검사 메뉴를 선택 (우측 버튼)
- 특정 태그를 선택하고 다시 우측 버튼을 누르고 Copy - Copy selector를 선택하면 CSS 선택자가 클립보드에 저장된다 (아래 예시)

### mw-content-text > div > ul:nth-child(6) > li > b > a
- 위에서 nth-child(6)은 6번째에 있는 요소를 가리킨다
- 이를 기반으로 작품목록을 가져오는 프로그램을 작성하겠다.

```python
url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC"
res = urllib.request.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')
A = soup.select('div > ul > li > a')
B = soup.find('li')
#### mw-content-text > div.mw-parser-output > ul:nth-child(6) > li > b > a
#### mw-content-text > div.mw-parser-output > ul:nth-child(6) > li > ul > li:nth-child(1) > a
a_list = soup.select('#mw-content-text > div > ul a')
for t in a_list:
  print( t.string )
```

<hr>

## <3>
### 데이터 수집 이어서  
### CSS를 활용하는 방법 외에 re (정규표현식)을 사용하여 필요한 데이터를 추출할 수 있다  

```python
import re
import requests as req
text = req.get('https://www.google.com').text
re.findall('<div class=(.*?)</div>', text)
```
## 네이버에서 정보처리기사 검색 후 파워링크 부분 가져오기
```python
#파워링크 이름들 가져오기
url = 'https://search.naver.com/search.naver'
res = req.get(url, params={'query':'정보처리기사'} )
soup = BeautifulSoup(res.text, 'html.parser')
```
## 뉴스페이지 스크레이핑
```python
#접근이 안될때는 header~
headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36',
        'Referer':'https://www.google.com/'}
#네이버 뉴스에 제목과, 내용 or 소제목 굵은 글씨
url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=shm&sid1=103&oid=346&aid=0000027271'

res = req.get(url, headers=headers)
```

### Selenium 라이브러리

* 웹 페이지를 테스트(제어)하기 위한 자동 테스팅 모듈

+ [Selenium with Python](https://selenium-python.readthedocs.io/index.html)
+ [Selenium Documentation](https://www.selenium.dev/documentation/en/)

selenium 설치 & 크롬드라이버 설치
* 크롤러와 웹 브라우저를 연결시켜 주기 위한 웹 드라이버 설치

```python
# 설치(20.08 코랩 기준)
!pip install Selenium
!apt-get update # to update ubuntu to correctly run apt install
!apt install chromium-chromedriver  #크롬 드라이버

# 한글 폰트 설치
!apt-get install -y fonts-nanum*

#selenium 설정
from selenium import webdriver  #크롬이라서 웹드라이버
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') #내부 창을 띄울 수 없으므로 설정(코랩이어서 창을 띄울수 없기 때문에 설정)
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Chrome 드라이버 생성 후 Browser 객체 반환 
driver = webdriver.Chrome('chromedriver', chrome_options=chrome_options)
```

### selenium으로 크롬 브라우저 웹 사이트 접근
```python
# 해당 URL로 브라우저 실행
driver.get('http://www.naver.com')
# 검색한 사이트 이미지로 저장
driver.save_screenshot('website.png')
# driver.quit()  #브라우저 종료
```
원하는 태그 찾기  
요소검사를 진행해서 id나 class 또는 태그명을 확인
* driver.find_element_by_css_selector (단수)
* driver.find_elements_by_css_selector (복수)

요소 접근 함수(Element Access Founction)

|단일 객체 반환 함수 <br>(find()과 같은 형태로 반환) | 리스트 반환 함수 <br>(find_all()과 같은 형태로 반환)|
|---|---|
|find_element_by_id<br>find_element_by_class_name<br>find_element_by_css_selector<br>find_element_tag_name<br>find_element_name<br>find_element_by_link | find_elements_by_id<br>find_elements_by_class_name<br>find_elements_by_css_selector<br>find_elements_tag_name<br>find_elements_name<br>find_elements_by_link|


```python
#자주 사용하는 패키지들
import urllib
from urllib.request import urlopen
from urllib.parse import quote_plus     #웹용 인코딩으로 변환(문자가 다를 때)
from bs4 import BeautifulSoup as bs    #축약하면 bs
from selenium.webdriver.common.keys import Keys   #selenium에서 key를 입력할 때 들어있는 것이 keys
import time
```
```python
#이미지 크롤링
url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
kword = input('검색어를 입력하세요 : ')
base_url = url + quote_plus(kword)  #quote_plus가 변환해줌
base_url
driver.get(base_url)
time.sleep(1)   #바로 찍으면 로드하는데 시간이 걸려서 아무것도 찍히지 않는다.
                #time.sleep으로 1초 늦게 찍으면 로딩이 된 이미지가 찍힌다.
driver.save_screenshot('website1.png')
#페이지 다운다운해서 밑에 나오는 이미지 찍기
body = driver.find_element_by_css_selector('body')

for i in range(5):
  body.send_keys(Keys.PAGE_DOWN)
  time.sleep(1)
  driver.save_screenshot(f'website_{i}.png')
mkdir './data'
#이미지 가져오기
imgs = driver.find_elements_by_css_selector('img')
for idx, img in enumerate(imgs):
  # print(img.get_attribute('src'))
  imgUrl = img.get_attribute('src')
  imgName = './data/' + kword + str(idx) + '.jpg'
  try:
    urllib.request.urlretrieve(imgUrl, imgName)
  except:
    print('error : ', imgName, imgUrl)
```

### 네이버 검색
```python
from selenium import webdriver as wb
from selenium.webdriver.common.keys import Keys
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') #내부 창을 띄울 수 없으므로 설정
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = wb.Chrome('chromedriver', chrome_options=chrome_options)
url = 'https://www.naver.com'
driver.get(url)
# driver.save_screenshot('naver01.png')
input_search = driver.find_element_by_id('query')
input_search.send_keys('광주날씨')
input_search.send_keys(Keys.ENTER)      #enter
driver.back()                           #back
time.sleep(1)
driver.save_screenshot('naver02.png')
search_btn = driver.find_element_by_id('search_btn')
search_btn.click()
time.sleep(1)
driver.save_screenshot('naver05.png')
```

### 뉴스 제목, 내용가져오기
```python
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') #내부 창을 띄울 수 없으므로 설정
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

driver = wb.Chrome('chromedriver', chrome_options=chrome_options)


```
### 구글에서 검색어 입력한 후 웹 페이지 결과 띄우기
웹 제어 함수 (Web Control Function)

1. 마우스 제어
 * 클릭 요소 : driver.find_element_by_css_selector().click()
 * submit 타입요소 : driver.find_element_by_css_selector().submit()

2. 키보드 제어
 * driver.find_element_by_css_selector().send_keys(text)
 * Ex1) input 태그에 ‘크롤링’을 입력할 때
   -> driver.find_element_by_css_selector().send_keys(“크롤링”)
 * Ex2) 키보드의 특수키 중 Enter를 입력할 경우
   -> driver.find_element_by_css_selector().send_keys(Keys.ENTER)
   
```python
url = 'http://www.google.com'
driver.get(url)
input = driver.find_element_by_tag_name('input')
input.send_keys('크롤링')
input.send_keys(Keys.ENTER)
driver.save_screenshot('google03.png')
```
### 한솥도시락의 이름,가격 정보 수집
```python
from selenium import webdriver as wb
from bs4 import BeautifulSoup as bs
import time
import pandas as pd
url = 'https://www.hsd.co.kr/menu/menu_list'
driver.get(url)
driver.save_screenshot('hsd01.png')     #<-정적페이지, 동적페이지는 time을 좀가져야 이미지가 로드된다.
soup = bs(driver.page_source, 'html.parser')
num = []
names = []
prices = []

name = soup.find_all('h4',class_='h fz_03')    #find_all = findAll
price = soup.select('div.item-price > strong')

len(name),len(price)  #(20,20)

for i in range(len(name)):
  num.append(i+1)
  names.append(name[i].text)
  prices.append(price[i].text)
info = {'num' : num, 'name' : names, 'price' : prices}
df = pd.DataFrame(info)
df.set_index('num', inplace=True)    #inplace를 넣으면 원본도 바뀐다.
df.head()
#lunchBox_info.csv로 저장
df.to_csv('lunchBox_info.csv', encoding='utf-8')    #인코딩 중요

```
#### 스타벅스 모든 음료 정보 가져오기
```python
#스타벅스 매장 위치 
url = 'https://www.starbucks.co.kr/store/store_map.do'
driver.get(url)
loca_search = driver.find_element_by_class_name('loca_search')
loca_search.click()
#광주지역 클릭
li = driver.find_element_by_css_selector('ul.sido_arae_box > li + li + li')
li.click()
#광주지역 -> 전체 클릭
all = driver.find_element_by_css_selector('ul.gugun_arae_box > li')
all.click()
soup = bs(driver.page_source, 'html.parser')
name = soup.select('#mCSB_3_container > ul > li > strong')
addr = soup.select('#mCSB_3_container > ul > li > p')
names = []
address = []
tels = []
len(name), len(addr)      #(58,58)
for i in range(len(name)):
  names.append(name[i].text)
  address.append(addr[i].text[:-9])
  tels.append(addr[i].text[-9:])
info = {'name' : names, 'address' : address, 'tell' : tels}
df = pd.DataFrame(info)
df.set_index('name', inplace=True)    #inplace를 넣으면 원본도 바뀐다.
df.head()
```

# 4주차
## <1>
```python
#코랩에서 한글사용
#그래프에 reina display 적용
#colab의 한글 폰트설정
!pip install mglearn
```
## 지도학습 - K-Nearest Neighbors (K-NN)
지도 학습 (Supervised Learning)
- 데이터에 대한 Label(명시적인 답)이 주어진 상태에서 컴퓨터를 학습시키는 방법. 

비지도 학습 (Unsupervised Learning)
- 데이터에 대한 Label(명시적인 답)이 없는 상태에서 컴퓨터를 학습시키는 방법.
- 데이터의 숨겨진 특성이나 구조를 파악하는데 사용.

분류 (Classification)
- 미리 정의된 여러 클래스 레이블 중 하나를 예측하는 것.
- 속성 값을 입력, 클래스 값을 출력으로 하는 모델
- 붓꽃(iris)의 세 품종 중 하나로 분류, 암 분류 등. 
- 이진분류, 다중 분류 등이 있다.

회귀 (Regression)
- 연속적인 숫자를 예측하는 것.
- 속성 값을 입력, 연속적인 실수 값을 출력으로 하는 모델
- 어떤 사람의 교육수준, 나이, 주거지를 바탕으로 연간 소득 예측. 
- 예측 값의 미묘한 차이가 크게 중요하지 않다.

일반화, 과대적합, 과소적합

일반화 (Generalization)
- 훈련 세트로 학습한 모델이 테스트 세트에 대해 정확히 예측 하도록 하는 것 .

과대적합 (Overfitting)
- 훈련 세트에 너무 맞추어져 있어 테스트 세트의 성능 저하.

과소적합 (Underfitting)
- 훈련 세트를 충분히 반영하지 못해 훈련 세트, 테스트 세트에서 모두 성능이 저하.

<center>
 <img src="https://image.slidesharecdn.com/2-171030145527/95/2supervised-learningepoch21-9-1024.jpg?cb=1509375471" alt="과대적합" width="40%" />

</center>
<center>
 <img src="https://image.slidesharecdn.com/2-171030145527/95/2supervised-learningepoch21-10-1024.jpg?cb=1509375471" alt="과소적합" width="60%" />

</center>
***일반화 성능이 최대화 되는 모델을 찾는 것이 목표***

과대적합 (Overfitting)
- 너무 상세하고 복잡한 모델링을 하여 훈련데이터에만 과도하게 정확히 동작하는 모델.

과소적합 (Underfitting)
- 모델링을 너무 간단하게 하여 성능이 제대로 나오지 않는 모델.
모델 복잡도 곡선

<center>
 <img src="https://tensorflowkorea.files.wordpress.com/2017/06/fig2-01.png" alt="모델 복잡도 곡선" width="60%" />

</center>
해결방법

- 주어진 훈련데이터의 다양성이 보장되어야 한다 (다양한 데이터포인트를 골고루 나타내야 한다)
- 일반적으로 데이터 양이 많으면 일반화에 도움이 된다.
- 그러나 편중된 데이터를 많이 모으는 것은 도움이 되지 않는다.
- 규제(Regularization)을 통해 모델의 복잡도를 적정선으로 설정한다.

## K-Nearest Neighbors (K-NN)
k-최근접 이웃 알고리즘

- 새로운 데이터 포인트와 가장 가까운 훈련 데이터셋의 데이터  포인트를 찾아 예측
- k 값에 따라 가까운 이웃의 수가 결정
- 분류와 회귀에 모두 사용 가능  
- 입력 값과 k개의 가까운 점이 있다고 가정할 때 그 점들이 어떤 라벨과 가장 비슷한지 (최 근접 이웃)
판단하는 알고리즘

- 매개 변수 : 데이터 포인트 사이의 거리를 재는 방법 (일반적으로 유클리디안 거리 사용), 이웃의 수
 - 장점 : 이해하기 쉬운 모델, 약간의 조정으로 좋은 성능
 - 단점 : 훈련 세트가 크면 속도가 느림, 많은 특성을 처리하기 힘듬

```python
import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=1)
mglearn.plots.plot_knn_classification(n_neighbors=3)

```
- k 값이 작을 수록 모델의 복잡도가 상대적으로 증가.
    (noise 값에 민감)
- 반대로 k 값이 많아질수록 모델의 복잡도가 낮아진다.
- 100개의 데이터를 학습하고 k를 100개로 설정하여 예측하면 빈도가 가장 많은 클래스 레이블로 분류

유클리디안 거리 (Euclidean distance) : 두 점사이의 거리를 계산할 때 쓰이는 방법
- 두 점 (p1, p2, ...)와 (q1, q2, ....)의 거리


유클리디안 거리 공식

 <img src="https://wikidocs.net/images/page/24654/2%EC%B0%A8%EC%9B%90_%ED%8F%89%EB%A9%B4.png" alt="유클리디안 거리" width="60%" />

</center>
KNeighborsClassifier()
```
KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs)
```
- n_neighbors : 이웃의 수 (default : 5)
- weights : 예측에 사용된 가중 함수 (uniform, distance) (default : uniform)
- algorithm : 가까운 이웃을 계산하는데 사용되는 알고리즘 (auto, ball_tree, kd_tree, brute)
- leaf_size : BallTree 또는 KDTree에 전달 된 리프 크기
- p : (1 : minkowski_distance, 2: manhattan_distance 및 euclidean_distance)
- metric : 트리에 사용하는 거리 메트릭스
- metric_params : 메트릭 함수에 대한 추가 키워드 인수
- n_jobs : 이웃 검색을 위해 실행할 병렬 작업 수

KNeighborsClassifier 모델은 k-최근접 이웃 분류 또는 KNN이라고 합니다. <br>
k-NN 알고리즘은 가장 가까운 훈련 데이터 포인트 K개를 최근접 이웃으로 찾아 예측에 사용합니다. <br>
n_neighbors=1 는 1개를 최근접 이웃으로 하겠다는 것입니다.

주요 매개변수(Hyperparameter)
- 거리측정 방법, 이웃의 수, 가중치 함수 

scikit-learn의 경우
- metric  :  유클리디언 거리 방식
- k : 이웃의 수
- weight  : 가중치 함수
     -  uniform : 가중치를 동등하게 설정.
     -  distance :  가중치를 거리에 반비례하도록 설정
  
장단점
- 이해하기 매우 쉬운 모델
- 훈련 데이터 세트가 크면(특성,샘플의 수) 예측이 느려진다
- 수백 개 이상의 많은 특성을 가진 데이터 세트와 특성 값 대부분이 0인 희소(sparse)한 데이터 세트에는 잘 동작하지 않는다
- 거리를 측정하기 때문에 같은 scale을 같도록 정규화 필요

##### weight 가중치 함수 추가설명

예를 들어 
```
영화 : A -> 등급: 5.0 , X까지의 거리: 3.2
영화 : B -> 등급: 6.8 , X까지의 거리: 11.5
영화 : C -> 등급: 9.0 , X까지의 거리: 1.1
```
가 있다고 할 때 

평균을 구하면
> (5.0 + 6.8 + 9.0) / 3 = 6.93

거리에 대한 가중 평균을 구해보면
> (5.0/3.2 + 6.8/11.5 + 9.0/1.1) / (1/3.2 + 1/11.5 + 1/1.1) = 7.9

code
```
print( (5.0 + 6.8 + 9.0) / 3 )
print( (5.0/3.2 + 6.8/11.5 + 9.0/1.1) / (1/3.2 + 1/11.5 + 1/1.1) )
출력
6.933333333333334
7.898546346988861
```

#### iris 데이터를 이용한 KNN 분류 실습

붓꽃 데이터 셋
- 클래스 (class) : 출력될 수 있는 값 (붓꽃의 종류)
- 레이블 (label) : 특정 데이터 포인트에 대한 출력

<center>
 <img src="https://tensorflowkorea.files.wordpress.com/2017/06/1-2.png" alt="붓꽃" width="30%" />

</center>
데이터셋 구성
- 150개의 데이터
- 4개의 정보와 1개의 클래스(3개의 품종)로 구성

| sepal_length|	sepal_width|	petal_length|	petal_width|	species|
|---|---|---|---|---|
| 꽃받침 길이| 꽃받침 넓이| 꽃잎 길이| 꽃잎 넓이| 품종|

#### 사이킷런 이용
```python
#데이터 가져오기
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(type(iris_dataset))

#Bunch는 키와 값의 쌍으로 이루어져 있다.
print(iris_dataset.keys())

#print(iris_dataset.DESCR)     #Data Set Characteristics
#print(iris_dataset.target_names)  ##['setosa' 'versicolor' 'virginica']
print(iris_dataset.target)
print(type(iris_dataset.target))
print(iris_dataset.data.shape)
print(iris_dataset.data[0:3])
```
훈련 세트(training set) <br>
테스트 세트(test set), 홀드아웃 세트(hold-out set)

```python
# 훈련 데이터와 테스트 데이터 준비
from sklearn.model_selection import train_test_split

X = iris_dataset.data 
Y = iris_dataset.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0) #데이터가 섞인다.

# 75% : 25%
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
# 데이터 조사
# 산점도 행렬 : 3개 이상의 특성을 표현
# 붖꽃은 4개의 특성을 가지므로 산점도 행렬을 이용한다.
import pandas as pd

iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

pd.plotting.scatter_matrix(iris_df, c=Y_train)
```
#### seaborn 이용
```python
# 데이터 가져오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
iris.shape    #(150,5)
iris.head()

# sepal_length	sepal_width	petal_length	petal_width	species
# 꽃받침 길이, 꽃받침 넓이, 꽃잎 길이, 꽃잎 넓이, 품종
```
훈련 세트(training set) <br>
테스트 세트(test set), 홀드아웃 세트(hold-out set)
```python
# 훈련 데이터와 테스트 데이터 준비
from sklearn.model_selection import train_test_split
# 훈련 데이터와 테스트 데이터 준비
from sklearn.model_selection import train_test_split

X = iris.iloc[:, :4]
y = iris.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 75% : 25%
X_train.shape, X_test.shape, y_train.shape, y_test.shape
sns.pairplot(iris, hue='species')
```
공통
```python
# 머신러닝 모델
# k-최근접 이웃 알고리즘
# 훈련 데이터에서 새로운 데이터 포인트에 가장 가까운 'k개'의 이웃을 찾는다.
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
# 모델 평가
# 테스트 세트 이용
from sklearn import metrics

pred = model.predict(X_test)
ac_score = metrics.accuracy_score(y_test, pred)
print('정답률 : ',ac_score)
# 예측하기
import numpy as np

X_new = [[5, 2.9, 1, 0.2]]
pre = model.predict(X_new)
print('예측 : ',pre)
```
##### iris 데이터를 이용한 KNN 분류 실습 전체코드
```python
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# 데이터 가져오기
# iris data loading
iris = sns.load_dataset('iris')

# 훈련 데이터와 테스트 데이터 준비
# 75% : 25%
X = iris.iloc[:, :4]
y = iris.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 모델 선택과 학습
# k-최근접 이웃 알고리즘
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)
ac_score = metrics.accuracy_score(y_test, pred)
print('정답률:', ac_score)

# 예측활용
X_new = [[5, 2.9, 1, 0.2]]
pre = model.predict(X_new)
print('예측:', pre)
```
### KNeighborsClassifier 분석
#### 결정경계([descision boundary](https://developers.google.com/machine-learning/glossary#%EA%B2%B0%EC%A0%95-%EA%B2%BD%EA%B3%84decision-boundary))

이웃의 수를 늘릴수록 결정경계는 더 부드러워진다.

이웃을 적게 사용하면 모델의 복잡도가 높아지고, 
많이 사용하면 복잡도는 낮아진다.
```python
import platform
from matplotlib import font_manager, rc 
import matplotlib
```
```
# Windows
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
```
```python
# KNeighborsClassifier 분석
import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()  #make_forge는 인위적으로 만들어진 분류 데이터 셋이다.

fig, axes = plt.subplots(1,3, figsize=(10,3))

for n, ax in zip([1, 3, 9], axes):
  model = KNeighborsClassifier(n_neighbors=n)
  model.fit(X, y)
  mglearn.plots.plot_2d_separator(model, X, ax=ax, fill=True, alpha=0.3)  #이웃 수가 많아질수록 선이 점점 부드러워 진다.
                                            #fill을True로 해주면 경계선을 따라 색칠해준다, alpha값을 조정해서 색깔 조정
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
  ax.set_title(f'{n}neighbor')

```
#### 모델 복잡도와 일반화 사이의 관계

이웃의 수 변화에 따른 훈련 세트와 테스트 세트의 성능 변화
- 데이터셋 : wisconsin의 유방암 데이터셋
- 총 569건의 데이터로 악성(212), 양성 (357)으로 구성

<center>
 <img src="https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile7.uf.tistory.com%2Fimage%2F99306C335A1685AA111704" alt="wisconsin의 유방암 데이터셋" width="30%" />

</center>

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)
# print(cancer.DESCR)   #description
import numpy as np

# 양성과 악성 데이터의 수
# zip() : 2개의 데이터를 연결
# bincount() : 클래스별 개수를 반환 #numpy
for n, v in zip(cancer.target_names, np.bincount(cancer.target)):
  print({n:v})

#2. 데이터프레임으로 바꿔서 valuecount하는 방법을 더 많이 쓰긴 함.
# 특성의 명칭
cancer.feature_names
# 훈련 데이터와 테스트 데이터 분리
# stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. 
# stratify 값을 target으로 지정해주면 
# 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. 
# (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고
#  classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

X_train.shape, X_test.shape
np.bincount(y_train), np.bincount(y_test), np.bincount(cancer.target)
```

## <2>
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)
# print(cancer.DESCR)   #description

import numpy as np

# 양성과 악성 데이터의 수
# zip() : 2개의 데이터를 연결
# bincount() : 클래스별 개수를 반환 #numpy
for n, v in zip(cancer.target_names, np.bincount(cancer.target)):
  print({n:v})

#2. 데이터프레임으로 바꿔서 valuecount하는 방법을 더 많이 쓰긴 함.

# 특성의 명칭
cancer.feature_names
```
```python

# 훈련 데이터와 테스트 데이터 분리
# stratify: default=None 입니다. classification을 다룰 때 매우 중요한 옵션값입니다. 
# stratify 값을 target으로 지정해주면 
# 각각의 class 비율(ratio)을 train / validation에 유지해 줍니다. 
# (한 쪽에 쏠려서 분배되는 것을 방지합니다) 만약 이 옵션을 지정해 주지 않고
#  classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있습니다.

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

X_train.shape, X_test.shape

np.bincount(y_train), np.bincount(y_test), np.bincount(cancer.target)
```
과대적합과 과소적합의 특징을 발견 (이웃의 수가 적을수록 모델이 복잡해지므로 그래프가 수평으로 뒤집힌 형태가 나타남) <br>
이웃의 수가 하나일 때 훈련 데이터에 대한 예측이 완벽하나, 이웃의 수가 늘어나면 모델은 단순해지고 훈련 데이터의 정확도는 줄어든다.

정확도가 가장 좋을 때는? -> 중간정도인 6개를 사용했을 경우

```python
train_acc = []
test_acc = []

n_neighbors = range(1,40)

for n in n_neighbors:
  model = KNeighborsClassifier(n_neighbors=n)
  model.fit(X_train, y_train)
  model.score(X_train, y_train)
  train_acc.append(model.score(X_train, y_train))
  test_acc.append(model.score(X_test, y_test))

plt.plot(n_neighbors, train_acc, label = '훈련 정확도')
plt.plot(n_neighbors, test_acc, label = '테스트 정확도')
plt.legend()
plt.show()

```
### K-NN 회귀

k-NN을 회귀에 사용한 경우
- 여러 개의 최근접 이웃을 사용할 경우에는 이웃 간의 평균이 예측 <br> (분류에서는 이웃의 레이블 개수를 확인해서 다수결로 정했지만, 회귀에서는 이웃들의 평균을 계산한다는 점이 차이)

- 분류는 모델과 비교하여 유사한 데이터의 개수(불연속)로 판단한다면 회귀는 데이터의 평균유사도 (연속적인)와 같은 수치로 판단

Classification(분류)는 연속적이지 않은 레이블, 다시 말해 ‘무엇’인지를 예측하지만, 회귀(Regression)는 연속된 수치, 즉 ‘얼마나’를 예측

```python
import mglearn
import matplotlib.pyplot as plt


mglearn.plots.plot_knn_regression(n_neighbors=3)

```
KNeighborsRegressor()

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples=9)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = KNeighborsRegressor(n_neighbors = 3)
model.fit(X_train, y_train)
model.predict(X_test)
```
#### K-NN 회귀 실습 01

IMDb 영화 데이터 세트

회귀에 대한 이야기니까 당연히 "평이 좋다" vs "평이 나쁘다" 레이블로 분류하는 게 아니라 <br>
실제 IMDb 등급(별점)을 예측하는 것이 과제의 목표

```python
from sklearn.neighbors import KNeighborsRegressor

# 영화에 대한 3개의 특성
X_train = [
  [0.5, 0.2, 0.1],
  [0.9, 0.7, 0.3],
  [0.4, 0.5, 0.7]
]
# 영화에 대한 별점
y_train = [5.0, 6.8, 9.0]

# 코드작성(모델)
model = KNeighborsRegressor(n_neighbors=3, weights='distance')
model.fit(X_train, y_train)


# 새로운 영화 3건에 대해 별점을 예측
X_test = [
  [0.2, 0.1, 0.7],
  [0.4, 0.7, 0.6],
  [0.5, 0.8, 0.1]
]

# 코드작성(예측)
model.predict(X_test)
```
##### weight 가중치 함수 추가설명

예를 들어 
```
영화 : A -> 등급: 5.0 , X까지의 거리: 3.2
영화 : B -> 등급: 6.8 , X까지의 거리: 11.5
영화 : C -> 등급: 9.0 , X까지의 거리: 1.1
```
가 있다고 할 때 

평균을 구하면
> (5.0 + 6.8 + 9.0) / 3 = 6.93

거리에 대한 가중 평균을 구해보면
> (5.0/3.2 + 6.8/11.5 + 9.0/1.1) / (1/3.2 + 1/11.5 + 1/1.1) = 7.9

code
```
print( (5.0 + 6.8 + 9.0) / 3 )
print( (5.0/3.2 + 6.8/11.5 + 9.0/1.1) / (1/3.2 + 1/11.5 + 1/1.1) )
출력
6.933333333333334
7.898546346988861
```

# 지도학습 - 선형회귀 (Linear Regression)

회귀 모델이란?

- 어떤 자료에 대해서 그 값에 영향을 주는 조건을 고려하여 구한 평균 <br>
(어떤 데이터들이 굉장히 크거나 작을지라도 전체적으로 이 데이터들은 전체 평균으로 회귀하려는 특징이 있다는 통계학 기법)
- \\( y = h(x_1, x_2, x_3, ..., x_k; W_1, W_2, W_3, ..., W_k) + \epsilon \\)

 - h() : 조건에 따른 평균을 구하는 함수 (회귀 모델)
 - x : 어떤 조건(특성)
 - W : 각 조건의 영향력(가중치)
 - e : ‘오차항’을 의미. 다양한 현실적인 한계로 인해 발생하는 불확실성으로 일종의 잡음(noise)

선형 모델이란?

- 입력 특성에 대한 선형 함수를 만들어 예측을 수행

- 다양한 선형 모델이 존재

- 분류와 회귀에 모두 사용 가능

<center>

시험성적 데이터

|X ( 학습 시간 )| Y ( 시험 점수 )|
|---|---|
|9 |90|
|8 |80|
|4| 40|
|2| 20|

7시간 공부 할 경우 성적은?
</center>

```python
import matplotlib.pyplot as plt
%matplotlib inline

x = [9, 8, 4, 2]
y = [90, 80, 40, 20]
plt.xlim(0,10)
plt.ylim(0,100)
plt.plot(x,y, 'b-o')
plt.grid()
```
 \\( y=ax+b \\)

 - a : 기울기
 - b : 절편


<center>

시험성적 데이터

|X ( 학습 시간 )| Y ( 시험 점수 )|
|---|---|
|8 |97|
|6 |91|
|4| 93|
|2| 81|

7시간 공부 할 경우 성적은?
</center>

```python
import matplotlib.pyplot as plt
%matplotlib inline
x = [8, 6, 4, 2]
y = [97, 91, 93, 81]
plt.xlim(0,10)
plt.ylim(75,100)
plt.plot(x,y, 'bo')
plt.grid()
```
 \\( y=ax+b \\)

 - a : 기울기
 - b : 절편

최소제곱법

> \\( a=\frac { (x-x평균)(y-y평균)의 합 }{ { (x-x평균) }^{ 2 }의 합 }  \\)

- 공부한 시간(x)의 평균: (2+4+6+8) / 4 = 5
- 성적(y)의 평균: (81+93+91+97) / 4 = 90.5

> \\( b=y의 평균- (x의 평균 \times 기울기 a) \\)

- b = 90.5 - (2.3 x 5) = 79
```python
a = ( (2-5)*(81-90.5)+(4-5)*(93-90.5)+(6-5)*(91-90.5)+(8-5)*(97-90.5) )  /  ( (2-5)**2 + (4-5)**2 + (6-5)**2 + (8-5)**2 )
a1 = ( (2-5)*(81-90.5)+(4-5)*(93-90.5)+(6-5)*(91-90.5)+(8-5)*(97-90.5) ) 
a2 = ( (2-5)**2 + (4-5)**2 + (6-5)**2 + (8-5)**2 )
a1, a2, a

# 오차가 최저가 되는 직선
import numpy as np
# 기울기 a를 최소제곱법으로 구하는 함수
def compute_a(x, y, mean_x, mean_y):
  #분자부분 :
  dc = 0
  for i in range(len(x)):
    dc += (x[i] - mean_x) * (y[i] - mean_y)


  #분모부분 :
  divisor = 0
  for i in range(len(x)):
    divisor += (x[i]-mean_x)**2
  
  a = dc / divisor
  return a

x = [8, 6, 4, 2]
y = [97, 91, 93, 81]
mean_x = np.mean(x)
mean_y = np.mean(y)
a = compute_a(x, y, mean_x, mean_y)
b = mean_y - (mean_x * a)

y_pred = [ a * x1 + b for x1 in x]

plt.plot(x, y_pred, 'r-o')
plt.plot(x, y, 'bo')
plt.grid()
plt.show()

```
## 선형회귀(Linear Regression) <br>
 또는 최소제곱법(Ordinary Least Squares)
 - 종속변수(응답변수) y와 한 개 이상의 독립변수(입력변수) x와의 상관관계를 모델링한 것

 >  \\( y=Wx+b \\)  
  - (W : 가중치, b : 편향(bias))

 > \\( H(x)=Wx+b \\)  
  - H(x) : Linear 하게 Hypothesis(가설)을 세운다는 것
  - 데이터를 가장 잘 대변할 수 있는 H(x)의 W와 b를 정하는 것이 Linear Regression의 목적

```python
import numpy as np
import mglearn

X, y = mglearn.datasets.make_wave(100)
plt.scatter(X, y)
plt.show()
import mglearn
mglearn.plots.plot_linear_regression_wave()
```
- 비용함수 (Cost / Cost function) : 그려진 직선 Hypothesis(H(x))와 실제 데이터(y)의 차이

  - Cost = H(x) - y에 데이터를 대입하여 Cost의 총합을 구하는 것이 가능
  - Cost의 총합이 작은 Hypothesis일수록 데이터를 잘 대변하는 훌륭한 Linear Regression
  - Cost는 양수일 수도, 음수일 수도 있기에 이러한 문제를 방지하고자 총합을 구할 때 Cost값을 제곱하여 평균을 내는 방식(평균제곱오차, MSE, Mean Squared Error)을 사용
  >  \\( cost(W,b)=\cfrac { 1 }{ m } \sum _{ i=1 }^{ m } { (H({ x }^{ (i) })-{ y }^{ (i) }) }^{ 2 }  \\)

   > \\( H(x)=Wx+b \\)  

 - 머신러닝(or 딥러닝)에서 learning의 목적은 Cost를 정의하고 이를 최소화하는 것
#### 평균제곱오차 (MSE, Mean Squared Error) - 잘못그은 선 바로잡기

- 실제값과 예측값의 차이를 분석하기 위한 것
- 음수가 존재하는 경우 오차가 줄어드는 문제 -> 자승을 취함
- 평균오차가 자승으로 인해 커지는 문제 -> 제곱근을 취함

|x(hour) | y(score)|
|---|---|
|0|0|
|1|1|
|2|2|
|3|3|

\\( H(x)=1 \times  x+0 \\)  

\\( H(x)=0.5 \times x+0 \\)

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 2, 3])

# 가중치(계수) 및 편향(절편)
w = 1
b = 0
y_pred1 = w * x + b
plt.plot(x, y_pred1, 'b-o')

w = 0.5
b = 0
y_pred2 = w * x + b
plt.plot(x, y_pred2, 'r-o')

plt.grid()
plt.show()
```

#### 평균제곱오차 (MSE, Mean Squared Error) 연습 01
가설의 MSE 값을 계산해보자.
\\( \cfrac { { (H({ x }^{ (1) })-{ y }^{ (1) }) }^{ 2 } + { (H({ x }^{ (2) })-{ y }^{ (2) }) }^{ 2 }+ { (H({ x }^{ (3) })-{ y }^{ (3) }) }^{ 2 }+ { (H({ x }^{ (4) })-{ y }^{ (4) }) }^{ 2 } }{ 4 } = ? \\)

```python
# y_pred(예측값), y(실제값)
def MSE(y_pred, y):
  cost = np.sum((y_pred - y)**2) / len(y)
  return cost

MSE(y_pred1, y)     #0.0
MSE(y_pred2, y)     #0.875
```

## <3>
#### 경사하강법 (Gradient descent algorithm) - 오차 수정하기

어떻게 비용함수 값이 최소가 되는 W 파라미터를 구할 수 있을까?

- 점진적인 하강, 점진적으로 반복적인 계산을 통해 W 파라미터 값을 업데이트 하면서 오류 값이 최소가 되는 값을 구하는 방식

- 함수의 기울기(경사)를 구하여 기울기가 낮은 쪽으로 
계속 이동하여 값을 최적화 시키는 방법 <br> (오차 (기울기)가 가장 작은 방향으로 이동시키는 방법)
  
learning_rate(학습 속도)란?

- W와 b의 미분 값(W_grad, b_grade)을 얼마만큼 반영할지를 결정하는 값.
- 주로 0.001, 0.00001과 같은 매우 작은 값을 사용하며 learning_rate가 클수록 변화가 빠르며, learning_rate가 작을수록 변화가 느리다고 예상.
- 꼭 변화가 빠르다고 해서 결과를 빨리 볼 수 있는 것은 아님.

```python
import numpy as np
import matplotlib.pyplot as plt
w_val = []
cost_val = []
xx_val = []

n_samples = 200
x = np.random.randn(n_samples)
x.shape
y = 2 * x + 4 + np.random.randn(n_samples)
# plt.scatter(x, y)

n_epoch = 20    # 반복횟수
lr = 0.5        # 학습속도

w = np.random.uniform()
b = np.random.uniform()

for epoch in range(n_epoch):
  y_pred = w * x + b
  cost = MSE(y_pred, y)
  xx = lr * ((y_pred - y) * x).mean()
  print(f'{epoch:3} w={w:.6f}, b={b:.6f}, cost={cost:.6f}, xx={xx:.6f}')

  w = w - xx
  b = b - lr * ((y_pred - y) * x).mean()


  w_val.append(w)
  cost_val.append(cost)
  xx_val.append(xx)

plt.plot(range(n_epoch), cost_val)

```
#### LinearRegression 실습 01

배달시간 예측
- 설정 거리의 장소에 배달하려면 얼마나 걸리는지 예측

```python
import numpy as np 
from matplotlib import pyplot as plt 
# 배달거리와 배달시간 데이터
data = np.array([
    [100, 20], 
		[150, 24], 
		[300, 36], 
		[400, 47], 
		[130, 22], 
		[240, 32],
		[350, 47], 
		[200, 42], 
		[100, 21], 
		[110, 21], 
		[190, 30], 
		[120, 25], 
		[130, 18], 
		[270, 38], 
		[255, 28]])

x = data[:, 0]
y = data[:, 1]
plt.scatter(x,y)
plt.xlim(0,450)
plt.ylim(0,50)
plt.grid()

# 기울기 a를 최소제곱법으로 구하는 함수
def compute_a(x, y, mean_x, mean_y):
    # 분자 부분
    dc = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    # 분모 부분  
    d = sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    return dc / d

mx = np.mean(x)
my = np.mean(y)
# 기울기
a = compute_a(x, y, mx, my)
# 절편
b = my - (mx * a)
a, b
```
```python
# 1
y_pred = [a * x1 + b for x1 in x]
plt.plot(x, y_pred, 'r-o')
plt.plot(x, y, 'bo')
plt.grid()
# 2
x1 = np.array(x)
y_pred = a * x1 + b
plt.plot(x, y_pred, 'r-o')

#예측하기(거리가 350인 경우 배달시간)
t = a * 350 + b
t
```
#### 일반 선형회귀

예측값과 실제 값의 cost를 최소화할 수 있도록 W(가중치, 회귀계수)를 최적화하며, 규제(Regularization)를 적용하지 않은 모델
단순 선형회귀 (Simple Linear Regression)
> \\( H({ x })={ W }{ x }+b \\)

다변수 선형회귀 (Multi-variable Linear Regreesion)
> 변수가 3개 일때의 H(x) <br>
> \\( H({ x }_{ 1 },{ x }_{ 2 },{ x }_{ 3 })={ W }_{ 1 }{ x }_{ 1 }+{ W }_{ 2 }{ x }_{ 2 }+{ W }_{ 3 }{ x }_{ 3 }+b \\)

> 변수가 n개 일때의 H(x) <br>
>  \\( H({ x }_{ 1 },{ x }_{ 2 },{ x }_{ 3 },\dots ,{ x }_{ n })={ W }_{ 1 }{ x }_{ 1 }+{ W }_{ 2 }{ x }_{ 2 }+{ W }_{ 3 }{ x }_{ 3 }+\dots +{ W }_{ n }{ x }_{ n }+b \\)

다항 회귀 (Polynomial Regreesion)
> 회귀가 독립변수의 단항식이 아닌 2차, 3차 방정식과 같은 다항식으로 표현되는 것 <br>
> 차수가 높아질수록 과적합의 문제가 발생
> 
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

n = 100
x = 6 * np.random.rand(n, 1) - 3
x.shape   #(100,1)
y = 0.5 * x**2 + x + 2 + np.random.rand(n, 1)
#다항의 특징 추가. , . 데이터 전처리

plt.scatter(x, y, s=5)
```
```python
# 전처리단계

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
x.shape, x_poly.shape
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_poly, y)
model.coef_, model.intercept_
```
#### LinearRegression 실습 02

wave 데이터셋에 선형회귀 적용
LinearRegression()
```
LinearRegression(fit_intercept, normalize, copy_X, n_jobs)
```
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default : True)
- normalize : 매개변수 무시 여부
- copy_X : X의 복사 여부
- n_jobs : 계산에 사용할 작업 수
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
X, y = mglearn.datasets.make_wave(60)
plt.scatter(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print('가중치(계수, 기울기, w):', model.coef_)
print('편향(절편, b):', model.intercept_)

print('훈련 점수:', model.score(X_train, y_train))
print('테스트 점수:', model.score(X_test, y_test))

```
#### LinearRegression 실습 03 - 1
변수가 1개인 경우
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

n_samples = 200
x = np.random.randn(n_samples)
w = 2
b = 3
y = w * x + b + np.random.randn(n_samples)  # 노이즈
plt.scatter(x, y)
# 위에 주어진 x, y를 이용하여 LinearRegression을 만들고, 
# 가중치와 편향을 출력해 보세요.

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

model = LinearRegression()

# x = np.reshape(x,(200,1))
# y = np.reshape(y, (200,1)) 
model.fit(x.reshape(-1,1), y)
model.coef_, model. intercept_
#기울기         #편향
```