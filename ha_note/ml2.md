# 5주차
## <1>
```python
!pip install mglearn
```
#### LinearRegression 실습 03 - 2
변수가 2개인 경우
다변수 선형 회귀 (Multi-variable Linear Regreesion)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

n_samples = 200
x1 = np.random.randn(n_samples)
x2 = np.random.randn(n_samples)
y = 2 * x1 + 3 * x2 + 4 + np.random.randn(n_samples)

# 주어진 x1, x2, y를 가지고 선형모델 적용 후 가중치와 편향을 출력해 보세요.
# x = x1 + x2

X = pd.DataFrame({'x1':x1, 'x2':x2})
# X = np.hstack( (x1.reshape(-1, 1), x2.reshape(-1, 1)) )

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
model.coef_, model.intercept_
```

#### LinearRegression 실습 04

보스턴 집값 데이터셋 구성
- 506개의 데이터
- 13개의 정보와 1개의 클래스로 구성
```
0 CRIM : 인구 1인당 범죄 발생 수
1 ZN : 25,000평방 피트 이상의 주거 구역 비중
2 INDUS : 소매업 외 상업이 차지하는 면적 비율
3 CHAS : 찰스강 위치 변수 (1: 강 주변, 0: 이외)
4 NOX : 일산화질소 농도
5 RM : 집의 평균 방 수
6 AGE : 1940년 이전 지어진 비율
7 DIS : 5가지 보스턴 시 고용 시설까지의 거리
8 RAD : 순환고속도로의 접근 용이성
9 TAX : $10,000당 부동산 세율 총계
10 PTRATIO : 지역별 학생과 교사 비율
11 B : 지역별 흑인 비율
12 LSTAT : 급여가 낮은 직업에 종사하는 인구 비율 (%)
13 MEDV : 가격 (단위 : $1,000)
```
```python
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
boston.data.shape
# 데이터 세트 DataFrame 변환
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['PRICE'] = boston.target
df.head()

# 훈련 데이터와 테스트 데이터 준비
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 모델 선택과 학습
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print('훈련 점수:', model.score(X_train, y_train) )
print('테스트 점수:', model.score(X_test, y_test))

# 예측/평가
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)       #mse는 제곱이기 때문에 루트를 씌워준다.
r2score = r2_score(y_test, y_pred)    #명시적인 표현

print(f'MSE={mse:.3f}, RMSE={rmse:.3f}, R2SCORE={r2score:.3f}')

# "가중치(계수, 기울기 파라미터 W) :"  # N 소수점 자릿수까지 반올림
print("가중치(계수, 기울기 파라미터 W) :", np.round(model.coef_, 1))
print("편향(절편 파라미터 b) :", model.intercept_)

# 특성(피처)별 회귀 계수 값 순으로 출력
coeff = pd.Series(data=np.round(model.coef_, 1), index=X.columns)
coeff.sort_values(ascending=False)

# 데이터 조사
# 시각화
# 2행 4열, axs는 4x2 개의 ax를 갖음
# 시본의 regplot을 이용해 산점도와 선형 회귀 직선을 함께 표현
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i, feature in enumerate(X.columns):
  row = int(i/4)
  col = i % 4
  sns.regplot(x=feature, y='PRICE', data=df, ax=axes[row][col] )
```
#### LinearRegression 실습 05
유방암 데이터셋에 선형회귀 적용
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)

model.score(X_test, y_test)
```
#### LinearRegression 실습 06
확장 보스턴 집값 셋에 선형회귀 적용
```python
import mglearn
from sklearn.datasets import load_boston
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)
```

#### LinearRegression 실습 07
붓꽃 데이터 셋에 선형회귀 적용
```python
from sklearn.datasets import load_iris
iris = load_iris()
#sepal 꽃받침
#petal 꽃잎
iris.feature_names
```
꽃받침 길이와 꽂잎 길이를 이용한 선형회귀
```python
# 꽃받침 길이
# 꽃잎 길이
X = iris.data[:, 0]
y = iris.data[:, 2]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1),
                                                    y.reshape(-1,1), random_state=0)
%matplotlib inline
import matplotlib.pyplot as plt

#두 가지 특성을 시각화 
plt.scatter(X_train, y_train)
xx = np.linspace(4, 8, 80)    #구간 안에서 일관성있게 값이 생긴다.
yy = model.coef_ * xx + model.intercept_
plt.scatter(xx, yy, marker='.')
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print('훈련 점수:', model.score(X_train, y_train) )
print('테스트 점수:', model.score(X_test, y_test) )
```
장단점
- k-NN에 비해 더 제약이 있는 것처럼 보이지만 특성이 많은 데이터셋의 경우에는 우수한 성능을 낼 수 있다.
- 모델의 복잡도를 제어할 방법이 없어 과대적합 되기 쉽다.

<hr>