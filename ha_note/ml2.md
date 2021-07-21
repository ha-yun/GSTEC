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


## <2>
```python
!pip install mglearn
import mglearn
import sklearn
sklearn.__version__
```
### 지도학습 - 규제선형모델(Ridge_Lasso_Regression)
Linear Regression의 문제

- 단순 선형회귀 : 단 하나의 특성(feature)을 가지고 라벨값(label) 또는 타깃(target)을 예측하기 위한
회귀 모델을 찾는 것
- 다중 선형회귀 : 여러 개의 특성을 활용해서 회귀모델을 찾는 것  

다중 선형 회귀 모델은 과대적합(overfitting) 될 때가 종종 발생
주어진 샘플들의 특성값들과 라벨값의 관계를 필요이상으로 너무 자세하게 복잡하게 분석했다는 것
- 새로운 데이터가 주어졌을 때 제대로 예측해내기가 어려움 (일반화 능력이 떨어짐)
- Ridge / Lasso / Elastic Regression 등장

Bias(편향) 오차와 Variance(분산) 오차

특성이 증가하면 복잡성이 증가 -> variance는 증가, bias는 감소 -> Overfitting 발생
- bias 감소가 variance의 증가와 같아지는 최적의 point를 찾아야 함
- Overfitting 문제 해결 : 데이터의 복잡도 줄이기, 정규화를 통한 분산 감소


|구분| 모델 복잡도| 적합성|
|---|---|---|
|Bias 오차가 낮은 & Variance 오차가 높은|복잡| 과대적합(over fitting)|
 |Bias 오차가 높은 Variance 오차가 낮은| 단순| 과소적합(under fitting)|

* Bias(편향) 에러가 높아지는 것은  많은 데이터를 고려하지 않아 (=모델이 너무 단순)  정확한 예측을 하지 못하는 경우
* Variance(분산) 에러는 노이즈까지 전부 학습하여 (=모델이 너무 복잡) 약간의 input에도 예측 Y 값이 크게 흔들리는 것

이 두가지 에러가 상호 Trade-off 관계에 있어서 이 둘을 모두 잡는 것은 불가능 한 딜레마가 발생

정규화(Regularization, 규제)

- 과대적합이 되지 않도록 모델을 강제로 제한하는 것을 의미.
- 가중치(w)의 값을 조정하여 제약을 주는 것.

- L1 규제 : Lasso
  - <font  color=yellow> w의 모든 원소에 똑같은 힘으로 규제를 적용하는 방법. 특정 계수들은 0이 됨.      
  - 특성선택(Feature Selection)이 자동으로 이루어진다. </font>

- L2 규제 : Ridge
  - <font  color=yellow> w의 모든 원소에 골고루 규제를 적용하여 0에 가깝게 만든다. </font>

## Ridge Regression
평균제곱오차식에 alpha 항이 추가
- alpha 값을 크게 하면 패널티 효과가 커지고(가중치 감소),
- alpha 값을 작게 하면 그 반대가 된다.
- 기존 선형회귀에서는 적절한 가중치와 편향을 찾아내는 것이 관건
- 추가적인 제약 조건(규제항)을 포함 – 가중치에 대한 제곱의 합을 사용
- <font  color=yellow> MSE가 최소가 되게 하는 가중치(w)와 편향(b)을 찾는 </font> 동시에 <font  color=yellow> MSE와 규제항의 합이 최소</font>가 되어야 함 -> 가중치 W의 모든 원소가 0이 되거나 0에 근사하도록 -> <font  color=yellow>학습한 가중치 (W)의 제곱을 규제항 (L2 규제)</font>으로 사용

> cost(W,b)  
> =MSE+규제항  
> =MSE+α⋅L2norm  
> =1/m m∑i=1 (H(x(i))−y(i))^2 + α n∑j=1 wj^2  
> (n:가중치의개수,α:규제의정도)

> MSE에 의한 Overfitting을 줄이기 위해 α를 크게 함 -> 정확도 감소 -> α가 너무 크면 MSE의 비중이 작아져서 과소적합 가능성 증가

- α가 증가하면 bias는 증가하고 variance는 감소하며 α가 0이 되면 MSE와 동일하게 되어 선형 회귀모델이 됨 -> Ridge 모델은 <font  color=yellow>bias을 약간 손해보면서 variance를 크게 줄여</font> 성능의 향상
- 단점 : <font  color=yellow>몇몇 변수가 중요하더라도 모든 변수에 대해 적합을 해야 하고 완벽한 0은 나오지 않음</font> -> 예측의 문제가 아니라 해석의 문제
Ridge()
  
```
Ridge(alpha, fit_intercept, normalize, copy_X, max_iter, tol, solver, random_state)
```

- alpha : 값이 클수록 강력한 정규화(규제) 설정하여 분산을 줄임, 양수로 설정
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default : True)
- normalize : 매개변수 무시 여부
- copy_X : X의 복사 여부
- max_iter : 계산에 사용할 작업 수
- tol : 정밀도
- solver : 계산에 사용할 알고리즘 (auto, svd, cholesky, lsqr, sparse_cg, sag, saga)
- random_state : 난수 seed 설정

#### Ridge_Lasso_Regression 실습 01
확장 보스턴 집값 셋에 선형회귀 적용
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
%matplotlib inline

# 확장 보스턴 집값
import mglearn
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

model = LinearRegression().fit(X_train, y_train)
print('훈련점수 : ', model.score(X_train, y_train))     #훈련점수 :  0.9520519609032727
print('테스트점수 : ', model.score(X_test,))     #테스트점수 :  0.607472195966557
```
확장 보스턴 집값 셋에 릿지회귀 적용
```python
# model_ridge
model_ridge = Ridge().fit(X_train, y_train)
print('훈련점수 : ', model_ridge.score(X_train, y_train))   #훈련점수 :  0.8857966585170941
print('테스트점수 : ', model_ridge.score(X_test, y_test))    #테스트점수 :  0.7527683481744751

print('---------------------------------------------------')

# model_ridge_10
model_ridge_10 = Ridge(alpha=10).fit(X_train, y_train)
print('훈련점수 : ', model_ridge_10.score(X_train, y_train))    #훈련점수 :  0.7882787115369614
print('테스트점수 : ', model_ridge_10.score(X_test, y_test))    #테스트점수 :  0.6359411489177309

print('=====================================================')

model_ridge_01 = Ridge(alpha=0.1).fit(X_train, y_train)
print('훈련점수 : ', model_ridge_01.score(X_train, y_train))    #훈련점수 :  0.9282273685001992
print('테스트점수 : ', model_ridge_01.score(X_test, y_test))     #테스트점수 :  0.7722067936479818

```
시각화
```python
plt.plot(model_ridge_10.coef_, '^', label='Ridge alpha=10')
plt.plot(model_ridge.coef_, 's', label='Ridge alpha=1')
plt.plot(model_ridge_01.coef_, 'v', label='Ridge alpha=0.1')
plt.plot(model.coef_, 'o', label='LinearRegression') #많이 퍼져있어서, 분산이 많아서, 과적합이 일어났던 것

plt.hlines(0,0, len(model.coef_))   #선을 그어준다. 선에서 떨어질수록 가중치 값이 큰거.,
plt.ylim(-25, 25)
```

## Lasso Regression
- 릿지 회귀의 단점을 해결하기 위해 대안으로 나온 방법
학습한 가중치의 절대값을 규제항(L1 규제)으로 사용 – 가중치의 절대값의 합을 사용
- 학습한 가중치의 절대값을 규제항(L1 규제)으로 사용 – 가중치의 절대값의 합을 사용

>cost(W,b)  
> =MSE+규제항  
> =MSE+α⋅L1norm  
> = 1/m m∑i=1 (H(x(i))−y(i))^2 + α n∑j=1 |wj|  
> (n:가중치의개수,α:규제의정도)

- 적당한 α만으로 몇몇 계수를 정확하게 0으로 만들 수 있음 -> 해석을 용이하게 함
- MSE와 규제항의 합이 최소가 되게 하는 파라미터 W와 b를 찾는 것이 Lasso의 목표
- MSE항이 작아질 수록 오차가 작아지고, L1-norm이 작아질 수록 많은 가중치들이 0이 되거나 0에 가까워짐 -> 데이터 전 처리에 주로 사용 (필요 없는 데이터 제거)
- Ridge와 Lasso의 성능 차이는 사용하는 데이터의 상황에 따라 다름 -> 유의미한 변수가 적을 때는 Lasso가 반대의 경우는 Ridge가 더 좋은 성능을 보임.

Lasso()
```
Lasso(alpha, fit_intercept, normalize, precompute, copy_X, max_iter, tol, warm_start, positive, solver, random_state, selection)
```
- alpha : 값이 클수록 강력한 정규화(규제) 설정하여 분산을 줄임, 양수로 설정
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default : True)
- normalize : 매개변수 무시 여부
- precompute : 계산속도를 높이기 위해 미리 계산된 그램 매트릭스를 사용할 것인지 여부
- copy_X : X의 복사 여부
- max_iter : 계산에 사용할 작업 수
- tol : 정밀도
- warm_start : 이전 모델을 초기화로 적합하게 사용할 것인지 여부
- positive : 계수가 양수로 사용할 것인지 여부
- solver : 계산에 사용할 알고리즘 (auto, svd, cholesky, lsqr, sparse_cg, sag, saga)
- random_state : 난수 seed 설정
- selection : 계수의 업데이트 방법 설정 (random으로 설정하면 tol이 1e-4보다 높을 때 빠른 수렴)

```python
from sklearn.linear_model import Lasso
# lasso
model_lasso = Lasso().fit(X_train, y_train)

print('훈련점수:', model_lasso.score(X_train, y_train) )    #훈련점수: 0.29323768991114607
print('테스트점수:', model_lasso.score(X_test, y_test) )     #테스트점수: 0.20937503255272294
model_lasso.coef_   #특성을 4개만 써서 점수가 낮음(과소적합)
print('사용한 특성 수:', np.sum( model_lasso.coef_ != 0 ) ) #0이 아닌 값의 개수가 나온다.    #사용한 특성 수: 4


from sklearn.linear_model import Lasso
# lasso
model_lasso_001 = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)

print('훈련점수:', model_lasso_001.score(X_train, y_train) )    #훈련점수: 0.8962226511086497
print('테스트점수:', model_lasso_001.score(X_test, y_test) )     #테스트점수: 0.7656571174549982
model_lasso.coef_   #특성을 4개만 써서 점수가 낮음(과소적합)
print('사용한 특성 수:', np.sum( model_lasso_001.coef_ != 0 ) ) #0이 아닌 값의 개수가 나온다.    #사용한 특성 수: 33


from sklearn.linear_model import Lasso
# lasso
model_lasso_00001 = Lasso(alpha=0.0001, max_iter=10000).fit(X_train, y_train)

print('훈련점수:', model_lasso_00001.score(X_train, y_train) )     #훈련점수: 0.9501169448631187    
print('테스트점수:', model_lasso_00001.score(X_test, y_test) )       #테스트점수: 0.6506993940304697
model_lasso.coef_   #특성을 4개만 써서 점수가 낮음(과소적합)
print('사용한 특성 수:', np.sum( model_lasso_00001.coef_ != 0 ) ) #0이 아닌 값의 개수가 나온다.  #사용한 특성 수: 97
```
```python
plt.plot(model_lasso.coef_, 's', label='Lasso alpha=1')
plt.plot(model_lasso_001.coef_, '^', label='Lasso alpha=0.01')
plt.plot(model_lasso_00001.coef_, 'v', label='Lasso alpha=0.0001')

plt.plot(model.coef_, 'o', label='LinearRegression')
plt.hlines(0, 0, len(model.coef_))
plt.ylim(-25, 25)
plt.legend(ncol=2, loc=(0,1))
```
## ElasticNet Regression
- 선형 회귀에 2가지 규제항 (L1 규제항, L2 규제항)을 추가한 것

> cost(W,b)  
> =MSE+규제항  
> =MSE+α1⋅L1norm+α2⋅L2norm  
> =1/m * m∑i=1(H(x(i))−y(i))^2 + α1 n∑j=1 |wj| + α2 n∑j=1 wj^2  
> (n:가중치의 개수,α:규제의 정도)

```
ElasticNet(alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X,
tol, warm_start, positive, random_state, selection)
```
- alpha : 값이 클수록 강력한 정규화(규제) 설정하여 분산을 줄임, 양수로 설정
- l1_ratio : L1 규제의 비율 (혼합비율?)
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default : True)
- normalize : 매개변수 무시 여부
- precompute : 계산속도를 높이기 위해 미리 계산된 그램 매트릭스를 사용할 것인지 여부
- copy_X : X의 복사 여부
- max_iter : 계산에 사용할 작업 수
- tol : 정밀도
- warm_start : 이전 모델을 초기화로 적합하게 사용할 것인지 여부
- positive : 계수가 양수로 사용할 것인지 여부
- random_state : 난수 seed 설정
- selection : 계수의 업데이트 방법 설정 (random으로 설정하면 tol이 1e-4보다 높을 때 빠른 수렴)
 
## 다항회귀
```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

n = 100
x = 6 * np.random.rand(n, 1) - 3
y = 0.5 * x**2 + 1 * x + 2 + np.random.rand(n, 1)
plt.scatter(x, y, s=5)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 데이터 변환 과정과 머신러닝을 연결해주는 파이프라인
from sklearn.pipeline import make_pipeline  

poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
x.shape, x_poly.shape

x[0], x_poly[0]   # (array([-2.32495759]), array([-2.32495759,  5.40542778]))

model = LinearRegression().fit(x_poly, y)
model.coef_, model.intercept_   # (array([[1.00772136, 0.51179524]]), array([2.46168853]))

model_lr = make_pipeline(PolynomialFeatures(2, include_bias=False),
                         LinearRegression())
model_lr.fit(x, y)

# 다항회귀 그래프
plt.scatter(x, y, s=5)
xx = np.linspace(-3, 3, 1000)
y_pred = model_lr.predict(xx.reshape(-1,1)) # xx[:, np.newaxis]  랑 같은 뜻 1차원을 2차원으로

plt.plot(xx, y_pred)
```
#### PolynomialFeatures()
```
PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=True)
```

- degree : 차수
- interaction_only: True면 2차항에서 상호작용항만 출력
- include_bias : 상수항 생성 여부


다항 변환

- 입력값  x 를 다항식으로 변환한다.

>x→[1,x,x2,x3,⋯]  
만약 열의 갯수가 두 개이고 2차 다항식으로 변환하는 경우에는 다음처럼 변환한다.  
[x1,x2]→[1,x1,x2,x21,x1x2,x22]  
예)  
[x1=0,x2=1]→[1,0,1,0,0,1]  
[x1=2,x2=3]→[1,2,3,4,6,9]
 
```python
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3,2)
X
poly = PolynomialFeatures(2, include_bias=False)
poly.fit_transform(X)
# 다항차수는 적용하지 않고, 오직 상호작용(교호작용) 효과만을 분석하려면 
# interaction_only=True 옵션을 설정해주면 됩니다. 
# degree를 가지고 교호작용을 몇 개 수준까지 볼지 설정해줄 수 있습니다.

poly = PolynomialFeatures(2, interaction_only=True)
poly.fit_transform(X)
```
## Linear / Ridge / Lasso / ElasticNet Regression의 비교
|구분| 릿지회귀| 라쏘회귀| 엘라스틱넷|
|---|---|---|---|
|제약식| L2 norm| L1 norm| L1+L2 norm|
|변수선택 |불가능| 가능| 가능|
|solution| closed form| 명시해 없음| 명시해 없음|
|장점| 변수간 상관관계가 높아도 좋은 성능| 변수간 상관관계가 높으면 성능↓| 변수간 상관관계를 반영한 정규화|
|특징| 크기가 큰 변수를 우선 적으로 줄임|비중요 변수를 우선적 으로 줄임|상관관계가 큰 변수를 동시에 선택/배제|

```python
import seaborn as sb

np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = np.sin(2 * np.pi * X) + np.random.rand(n_samples) * 0.1   #+np.random.rand(n_samples) 노이즈 추가, 0.1을 곱한 이유는 노이즈를 약간만 첨가하기 위함
plt.scatter(X, y)

model_lr = make_pipeline(PolynomialFeatures(9), LinearRegression())
model_lr.fit(X.reshape(-1,1),y)    

xx = np.linspace(0, 1, 1000)
y_pred = model_lr.predict(xx.reshape(-1,1))
plt.plot(xx, y_pred)
plt.scatter(X, y)

##과적합
model_lr = make_pipeline(PolynomialFeatures(15), LinearRegression())
model_lr.fit(X.reshape(-1, 1), y)
print(model_lr.steps[1][1].coef_)

xx = np.linspace(0, 1, 1000)
y_pred = model_lr.predict( xx.reshape(-1, 1) )
plt.plot(xx, y_pred)
plt.scatter(X, y)
```
```python
model_lr = make_pipeline(PolynomialFeatures(9), Ridge(alpha=0.001))
model_lr.fit(X.reshape(-1, 1), y)
print(model_lr.steps[1][1].coef_)

xx = np.linspace(0, 1, 1000)
y_pred = model_lr.predict( xx.reshape(-1, 1) )
plt.plot(xx, y_pred)
plt.scatter(X, y)
plt.ylim(-1.5, 1.5)
```
```python
model_lr = make_pipeline(PolynomialFeatures(9), Lasso(alpha=0.000001))
model_lr.fit(X.reshape(-1, 1), y)
print(model_lr.steps[1][1].coef_)

xx = np.linspace(0, 1, 1000)
y_pred = model_lr.predict( xx.reshape(-1, 1) )
plt.plot(xx, y_pred)
plt.scatter(X, y)
plt.ylim(-1.5, 1.5)
```
```python
model_lr = make_pipeline(PolynomialFeatures(9), 
                         ElasticNet(alpha=0.0001, l1_ratio=1.0, max_iter=10000))
model_lr.fit(X.reshape(-1, 1), y)
print( model_lr.steps[1][1].coef_ )

xx = np.linspace(0, 1, 1000)
y_pred = model_lr.predict( xx.reshape(-1, 1) )
plt.plot(xx, y_pred)
plt.scatter(X, y)
plt.ylim(-1.5, 1.5)
```