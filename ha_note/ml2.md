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

## <3>
## Linear Classifier (선형분류)
- 계산한 값이 0보다 작은 클래스는 -1, 0보다 크면 +1이라고 예측(분류)
> ŷ = w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b > 0 <br>
> Linear Regression와 매우 비슷하지만 가중치(w) 합을 사용하는 대신 예측한 값을 임계치 0 과 비교

- 이진 선형 분류기는 선, 평면, 초평면을 이용하여 2개의 클래스를 구분하는 분류기

경사하강법(Gradient Descent) 최적화 알고리즘을 사용하여 선형 모델을 작성

[SGDClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

```
SGDClassifier(alpha, average, class_weight, epsilon, eta0, fit_intercept, l1_ratio, learning_rat, loss, max_iter, n_iter, n_jobs, penalty, power_t, random_state, shuffle, tol, verbose, warm_start)
```
확률적 경사하강법(SGD, Stochastic Gradient Descent)을 이용하여 선형모델을 구현

- lossstr : 손실함수 (default='hinge')
- penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
- alpha : 값이 클수록 강력한 정규화(규제) 설정 (default=0.0001)
- l1_ratio : L1 규제의 비율(Elastic-Net 믹싱 파라미터 경우에만 사용) (default=0.15)
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default=True)
- max_iter : 계산에 사용할 작업 수 (default=1000)
- tol : 정밀도
- shuffle : 에포크 후에 트레이닝 데이터를 섞는 유무 (default=True)
- epsilon : 손실 함수에서의 엡실론, 엡실론이 작은 경우, 현재 예측과 올바른 레이블 간의 차이가 임계 값보다 작으면 무시 (default=0.1)
- n_jobs : 병렬 처리 할 때 사용되는 CPU 코어 수
- random_state : 난수 seed 설정
- learning_rate : 학습속도 (default='optimal')
- eta0 : 초기 학습속도 (default=0.0)
- power_t : 역 스케일링 학습률 (default=0.5)
- early_stopping : 유효성 검사 점수가 향상되지 않을 때 조기 중지여부 (default=False)
- validation_fraction : 조기 중지를위한 검증 세트로 설정할 교육 데이터의 비율 (default=0.1)
- n_iter_no_change : 조기중지 전 반복횟수 (default=5)
- class_weight : 클래스와 관련된 가중치 {class_label: weight} or “balanced”, default=None
- warm_start : 초기화 유무 (default=False)
- average : True로 설정하면 모든 업데이트에 대한 평균 SGD 가중치를 계산하고 결과를 coef_속성에 저장 (default=False)


#### LinearClassifier 실습 01

붓꽃 데이터 셋에 선형분류 적용
```python
from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()
iris.data.shape
X = iris.data
y = iris.target
X2 = X[:,:2]    #꽃받침의 길이와 넓이

import matplotlib.pyplot as plt
%matplotlib inline

plt.scatter(X2[:, 0], X2[:, 1], c=y)

y2 = y.copy()
y2[y2==2]=1
plt.scatter(X2[:, 0], X2[:, 1], c=y2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2,
                                                    test_size=0.5, random_state=0)
X_train.shape

from sklearn.linear_model import SGDClassifier
model = SGDClassifier(random_state=0)
model.fit(X_train, y_train)
model.coef_, model.intercept_

print(model.score(X_train, y_train))        #0.9866666666666667
print(model.score(X_test, y_test))          #1.0
```

```python
import numpy as np

w0 = model.coef_[0,0]
w1 = model.coef_[0,1]
b = model.intercept_

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
x0 = np.linspace(4, 8, 50)
x1 = -(w0 * x0 + b)/w1        # w0 * x0 + w1 * x1 + b
plt.plot(x0, x1)
plt.xlim(4,8)
plt.ylim(2,4.5)
```

#### 4개 속성 모두 이용
세가지 꽃 구분
```python
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=0)

model = SGDClassifier(random_state=0).fit(X_train, y_train)
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

model.coef_, model.intercept_   #선형분류 3번 동작
```
```python
!pip install mglearn
```

## Logistic Regression
선형 회귀로 풀리지 않는 문제 -> 독립변수와 종속변수가 비선형 관계인 경우

 <img src="https://www.geogebra.org/resource/SYzqYk7Y/xIOqFXlBGbAVa8OG/material-SYzqYk7Y.png" alt="비선형1" width="40%" />

  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Polynomialdeg5.svg/1200px-Polynomialdeg5.svg.png" alt="비선형2" width="40%" />

공부시간과 합격률의 관계 -> 연속적으로 변하는 공부시간의 결과에 대해 “합격했다” 또는
“합격하지 않았다?”의 결과만 필요  
분류방법 1 -> 선형분류

더 나은 분류 방법은?
- 1958년 D.R.Cox가 제안한 확률 모델로 <font  color=yellow>독립변수의 선형 결합을 이용하여 종속변수 (사건의 발생 가능성)을 예측</font>하는데 사용되는 통계 기법 
 - 종속 변수가 <font  color=yellow>범주형 데이터를 대상</font> -> 분류

- 선형 회귀는 독립변수 x가 변화할 때 종속변수 y가 어떻게 변하는 지 예측하는 것
- 선형 회귀에서 x와 y의 범위 : [-∞, ∞]

- 연속적으로 변하는 2개의 결과만을 확인하는 경우
 - 조사회수가 많아지면 종속 변수 y는 확률로 표현됨
 - 독립변수의 범위는 [-∞, ∞], 종속변수의 범위는 [0, 1]
 - 선형회귀를 적용하면 종속변수의 값 범위를 넘어가는 문제가 발생 
 - 예측정확도 하락 
 - <font  color=yellow>로지스틱 모형</font> 적용
로지스틱 회귀는 선형 회귀 분석과 유사하지만 종속 변수(y)가 범주형 데이터를 대상으로 하며, 입력 데이터(x)가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류모델 기법으로 사용

합격을 1, 불합격을 0으로 하는 1과 0사이의 직선은 그리기 어렵움
 - 참(1)과 거짓(0) 사이를 구분하는 S자 형태의 곡선이면 편리

로지스틱 회귀는 선형 회귀와 마찬가지로 적절한 선을 그려가는 과정

시그모이드 함수 e(자연상수)는 무리수 값 2.71828...
 - 파이처럼 상수로 고정된 값

#### 로지스틱 함수 (sigmoid 함수)
오즈 (odds) : 성공확률이 실패확률에 비해 몇 배 높은가를 나타냄, 범위 [0, 1]

> \\( odds=\frac { p(y) }{ 1-p(y) }  \\)

로짓 변환 : 오즈에 자연로그를 취한 것으로 입력 값의 범위가 [0, 1] 일 때 출력 값의 범위를 [-∞, ∞]
로 조정

> \\( \log _{ e } {\frac { p(y) }{ 1-p(y) }} =\ln { \frac { p(y) }{ 1-p(y) }  }  \\)

로지스틱 함수 (sigmoid 함수) : 독립변수 x가 어느 숫자이든 상관없이 종속 변수의 값의 범위가 항
상 [0, 1] 범위에 있도록 함

> \\( \ln { \frac { p(y) }{ 1-p(y) }  }  = z  \\) <br>

> -> \\(  \frac { p(y) }{ 1-p(y) }   = { e }^{ z  }  \\) <br>

> -> $ \\ \begin{align}
p & = \frac {  { e }^{ z  }  }{ 1 +  { e }^{ z  }  } \\
& = \frac {  1  }{ 1 +  { e }^{(- z)  }  } 
\end{align} \\ $

#### 로지스틱 회귀 (Logistic Regression)
- 간단하면서도 파라미터의 수가 적어서 빠르게 예측
 - 다른 알고리즘과의 비교 기준점으로 사용
- 로지스틱 함수를 사용하여 확률을 추정하며 2 클래스 및 다중 클래스 분류를 위한 강력한 통계 방법으로 빠르고 단순
- 직선 대신 S 모양 곡선을 사용한다는 사실 때문에 데이터를 그룹으로 나누는 데 적합
- 용도 : 신용 점수, 마케팅 캠페인의 성공률 측정, 특정 제품의 매출 예측률, 특정 날에 지진이 발생할 확률 등

- 선형 회귀 방식을 분류에 적용한 알고리즘 (선형 회귀 계열)

- Linear Regression와 매우 비슷하지만 가중치(w) 합을 사용하는 대신 예측한 값을 임계치 0 과 비교
- 계산한 값이 0보다 작은 클래스는 -1, 0보다 크면 +1이라고 예측(분류)

- 이 규칙은 Classifier에 쓰이는 모든 Linear model에서 동일
- 가장 널리 알려진 두 개의 linear classifier algorithm
  1. Logistic Regression
  1. Support Vector Classifier의 Linear SVC
    
[LogisticRegression()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
```
LogisticRegression(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
```
- <font  color=yellow> penalty : 규제에 사용 된 기준을 지정 (l1, l2, elasticnet, none) – default : l2 </font>
- dual : 이중 또는 초기 공식
- tol : 정밀도
- <font  color=yellow> C : 규제 강도 </font>
- fit_intercept : 모형에 상수항 (절편)이 있는가 없는가를 결정하는 인수 (default : True)
- intercept_scaling : 정규화 효과 정도
- class_weight : 클래스의 가중치
- random_state : 난수 seed 설정
- solver : 최적화 문제에 사용하는 알고리즘
- max_iter : 계산에 사용할 작업 수
- multi_class : 다중 분류 시에 (ovr, multinomial, auto)로 설정
- verbose : 동작 과정에 대한 출력 메시지
- warm_start : 이전 모델을 초기화로 적합하게 사용할 것인지 여부
- n_jobs : 병렬 처리 할 때 사용되는 CPU 코어 수
- <font  color=yellow> l1_ratio : L1 규제의 비율(Elastic-Net 믹싱 파라미터 경우에만 사용) </font>

#### LogisticRegression 실습 01
학습시간 대비 합격분류 적용
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# 학습시간 대비 합격 데이터
pass_time = [8, 9, 9, 9.5, 10, 12, 14, 14.5, 15, 16, 16, 16.5, 17, 17, 17, 17.5, 20, 20, 20]
fail_time = [1, 2, 2.1, 2.6, 2.7, 2.8, 2.9, 3, 3.2, 3.4, 3.5, 3.6, 3, 5, 5.2, 5.4]
```
```python
# X
X = np.hstack( (pass_time, fail_time) )

# y
y1 = [1] * len(pass_time)
y0 = [0] * len(fail_time)
y = np.hstack( (y1, y0) )
y

# 시각화
plt.scatter(X, y)
plt.grid()
```
```python
# 모델학습
model = LogisticRegression()
model.fit(X.reshape(-1,1), y)
print(model.coef_, model.intercept_)

# 예측 분류
model.predict([[7,]])   # array([1]) -> 7시간은 합격

model.predict_proba([[7,]]) # array([[0.46974336, 0.53025664]]) 불합격할 확률, 합격할 확률

# 모델시각화
def logreg(z):
  return 1 / (1 + np.exp(-z))

xx = np.linspace(1, 21, 100)
yy = logreg(model.coef_ * xx + model.intercept_)[0]
plt.plot(xx, yy, c='r')

plt.scatter(X, y)
plt.grid()
```
```python
# 가중치값, 절편을 바꿨을 때 변화
w_list = [1.0]   # 가중치  
b_list = [-2, 0, 2]          # 편향
xx = np.linspace(-10, 10, 100)
for w in w_list:
  for b in b_list:
    yy = logreg(w * xx + b)
    plt.plot(xx, yy, label = f'{w}')
plt.legend()
# 중심점은 그대로 있고, 기울기만 변화 (가중치에 따라)
# 기울기는 그대로 있고, 확률 변화(편향에 따라)
```
* 로지스틱회귀를 퍼셉트론 방식으로 표현

<center>
 <img src="https://thebook.io/img/080228/100.jpg" alt="퍼셉트론" width="40%" />

</center>


<hr>

# 6주차
## <1>
#### LogisticRegression 실습 02
forge 데이터 셋을 이용  
- forge데이터는 0과 삼각형으로 구분되는 함수
```python
!pip install mglearn
from mglearn.datasets import make_forge
import matplotlib.pyplot as plt
import numpy as np
import mglearn
from sklearn.linear_model import LogisticRegression
%matplotlib inline

X, y = make_forge()
# C 값을 변경해가면서 결정경계가 어떻게 변하는지 확인해보자.
model = LogisticRegression(C=0.1, max_iter=100).fit(X, y)

mglearn.plots.plot_2d_separator(model, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()
```
#### LogisticRegression 실습 03
유방암 데이터를 이용한 분석 - 1
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.data.shape
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape   #(426, 30)

model = LogisticRegression(C=1, max_iter=10000).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)  #(0.960093896713615, 0.951048951048951)


# (실습) C=100, C=0.01 등으로 변경하면서 학습 점수의 결과를 비교
model_100 = LogisticRegression(C=100, max_iter=10000).fit(X_train, y_train)
model_100.score(X_train, y_train), model_100.score(X_test, y_test)  #(0.9812206572769953, 0.958041958041958)
model_01 = LogisticRegression(C=0.01, max_iter=10000).fit(X_train, y_train)
model_01.score(X_train, y_train), model_01.score(X_test, y_test)    #(0.9530516431924883, 0.9440559440559441)
```
규제 매개변수 C 설정을 다르게 하여 학습 시킨 모델의 계수 표시
```python
# 다른 C 값의 결과 추가
plt.plot(model.coef_.T, 'o')  #.T : 전치행렬
plt.xticks(range(30), cancer.feature_names, rotation=90)
plt.grid()
plt.show()

# 실습 - C 값에 따른 가중치 변화 시각화
plt.plot(model_100.coef_.T, 'o')
plt.xticks(range(30), cancer.feature_names, rotation=90)
plt.grid()
plt.show()

# 실습 - C 값에 따른 가중치 변화 시각화
plt.plot(model_01.coef_.T, 'o')
plt.xticks(range(30), cancer.feature_names, rotation=90)
plt.grid()
plt.show()
```
(실습) C=100, C=0.01 등으로 변경하면서 학습시킨 학습모델의 계수를 위의 그래프에 표시
```python
# 실습 - C 값에 따른 가중치 변화 시각화
for C, marker in zip( [100, 1, 0.01], ['^', 'o', 'v'] ):
  model = LogisticRegression(C=C, max_iter=10000).fit(X_train, y_train)
  model.score(X_train, y_train), model.score(X_test, y_test)
  plt.plot(model.coef_.T, marker, label=f'{C:.3f}')

plt.xticks(range(30), cancer.feature_names, rotation=90)
plt.grid()
plt.legend()
plt.show()
# C100이 더 넓게 분포한다..
```

C를 L1규제로 사용할 경우 분류 정확도와 계수 그래프를 표시
- Regularization에서 모든 특성을 이용할지 일부 특성만을 사용할지 결정하는 주요 parameter는 'penalty'
```python
model_l1 = LogisticRegression(C=1, penalty='l1', solver='liblinear', max_iter=10000)
model_l1.fit(X_train, y_train)
# C를 올리면 특성 수가 늘어난다.
print('사용한 특성 수:', np.sum(model_l1.coef_ != 0 ) )
model_l1.score(X_train, y_train), model_l1.score(X_test, y_test)
#사용한 특성 수: 11
#(0.9624413145539906, 0.958041958041958)
```
#### LogisticRegression 실습 04
와인 데이터를 이용한 분류
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

wine = load_wine()
wine.keys()
wine.data.shape
wine.target

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['등급'] = wine.target
df.head(3)

# 학습 데이터와 테스트 데이터 준비
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# 모델 선택 및 학습 그리고 평가
model = LogisticRegression(C=0.1, max_iter=10000)
model.fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)

y_pred = model.predict(X_test)
print(y_test.values)
print(y_pred)
```
#### LogisticRegression 실습 05
wave 데이터를 이용한 다중 분류
```python
# 데이터 생성 및 시각화
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()

# 훈련셋 및 테스트셋 분리 및 LogisticRegression 실행
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression(C=1, max_iter=10000)
model.fit(X_train, y_train)

#평가하기
model.score(X_train, y_train), model.score(X_test, y_test)

# 예측하기
y_pred = model.predict(X_test)
print(y_test)
print(y_pred)

# 실행 결과 시각화
xx = np.linspace(-10, 10, 50)
yy = -(model.coef_[0][0] * xx + model.intercept_[0]) / model.coef_[0][1]
plt.plot(xx, yy)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.show()
```

<hr>

## <2>
#### LogisticRegression 실습 06
붓꽃 데이터 셋을 이용한 다중 분류
```python
!pip install mglearn
# 데이터 로드
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris_dataset = load_iris()

# 데이터 분리
from sklearn.model_selection import train_test_split
X = iris_dataset.data[:, 2:]
y = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# LogisticRegression 수행
model = LogisticRegression(C=1, max_iter=10000).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)

# 결정경계 시각화 
mglearn.plots.plot_2d_classification(model, X_train)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()

from sklearn.svm import LinearSVC

model = LinearSVC(C=1, max_iter=10000).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)

# 결정경계 시각화 
mglearn.plots.plot_2d_classification(model, X_train)
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()

# 결정경계
# 꽃잎의 너비가 0~3cm 인 꽃에 대해 모델의 추정 확률을 계산
X = iris_dataset.data[:, 3:] # 꽃잎의 너비
y = (iris_dataset.target == 2).astype(np.int) # 'virginica'

model = LogisticRegression().fit(X, y)

xx = np.linspace(-10, 10, 500).reshape(-1,1)
y_proba = model.predict_proba(xx)
plt.plot(xx, y_proba[:,0], 'r--', label='not virginica')
plt.plot(xx, y_proba[:,1], 'g--', label='virginica')
plt.xlim(0,3)
plt.legend()
```
#### Logistic Regression 특징
- C 값에 의해 규제 (L1, L2 모두 사용)
- 학습속도가 빠르고 예측도 빠름
- 매우 큰 데이터셋과 희소한 데이터 셋에도 잘 동작함 (solver='sag')

#### LogisticRegression 실습 07
유방암 데이터를 이용한 분석 - 2
모델비교 (kNN, 결정트리, 랜덤포레스트)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   # K-fold cross validation
from sklearn import metrics
from sklearn.datasets import load_breast_cancer

# data = pd.read_csv('data/breast_cancer.csv')
# print(data.shape)
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# 모든 속성을 다 사용하여 로지스틱 회귀 분석을 수행
model = LogisticRegression(C=10, max_iter=10000).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)

# 모든 속성을 다 사용한 경우의 kNN의 성능
from sklearn.neighbors import KNeighborsClassifier  
for n in range(1, 21, 2):
  model = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
  print(f'k={n}', model.score(X_train, y_train), model.score(X_test, y_test) )

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)  #(1.0, 0.916083916083916)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
model.score(X_train, y_train), model.score(X_test, y_test)  #(1.0, 0.972027972027972)
```
선형모델은 계산을 하다보니 값에 영향을 받지만
트리계열모델은 스스로 if문을 만들어서 학습을 몹시 잘한다.  
대표적인 분류모델로 많이 쓰임.  
n_estimator은 휴전기.? 디시젼 트리  

#### 소프트맥스
- 세 개 이상 입력 값을 다루기 위함(다중분류)
- 다항 로지스틱스
> \\( { y }_{ k }=\frac { exp({ a }_{ k }) }{ \sum _{ i=1 }^{ n }{ exp({ a }_{ i }) }  }  \\)

```python
score = [-1,-0.5, 1.0, 1.5]

prob = []
sum = 0 
for s in score:
  prob.append(np.exp(s))
  sum += np.exp(s)

y = prob/sum
print(y)
print(np.sum(y))

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a) # 밑이 자연상수 e인 지수함수로 변환해준다.e^a
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)
print(np.sum(y))

def softmax(a):
  exp_a = np.exp(a) # 지수함수
  sum_exp_a = np.sum(exp_a) # 지수의 합
  y = exp_a / sum_exp_a
  return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))
```

<hr>

## <3>
```python
from sklearn.datasets import load_wine

# 데이터 로드
wine = load_wine()
wine

# 훈련(학습)셋 및 테스트(평가)셋 분리
from sklearn.model_selection import train_test_split
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape

# 모델학습
from sklearn.svm import LinearSVC
model = LinearSVC(C=1, max_iter=10000).fit(X_train, y_train)

# 스코어 확인
model.score(X_train, y_train), model.score(X_test,y_test)

# 예측
y_pred = model.predict(X_test )
y_pred
```
#### Linear SVC 실습 02
붓꽃 데이터 분류
```python
# 데이터 로드
from sklearn.datasets import load_iris
iris_dataset = load_iris()

# 훈련셋 및 테스트셋 분리
from sklearn.model_selection import train_test_split
X = iris_dataset.data
y = iris_dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train.shape

# 모델학습
from sklearn.svm import LinearSVC
model = LinearSVC(C=100, max_iter=100000, random_state=42).fit(X_train, y_train)

# 스코어 확인
model.score(X_train, y_train), model.score(X_test, y_test)

# 예측
y_pred = model.predict(X_test)
y_pred
```

## Linear SVC와 Logistic Regression의 장단점

- 선형 모델의 주요 매개변수는 회귀 모델에서는 alpha였고 LinearSVC와 LogisticRegression에서는 C
- alpha 값이 클수록, C 값이 작을수록 모델이 단순해짐
- 회귀 모델에서 이 매개변수를 조정하는 일이 매우 중요

- L1 규제를 사용할지 L2 규제를 사용할지를 정해야 함
- 중요한 특성이 많지 않다고 생각하면 L1 규제를 사용하고 그렇지 않으면 기본적으로 L2 규제를 사용

- 선형 모델은 학습 속도가 빠르고 예측도 빠름
- 매우 큰 데이터셋과 희소한 데이터셋에도 잘 작동
- 수십만에서 수백만 개의 샘플로 이뤄진 대용량 데이터셋이라면 기본 설정보다 빨리 처리하도록 LogisticRegression과 Ridge에 solver=’sag’ 옵션을 줌

- 선형 모델은 샘플에 비해 특성이 많을 때 잘 작동
- 다른 모델로 학습하기 어려운 매우 큰 데이터셋에도 선형 모델을 많이 사용

## SVM (Support Vector Machines)

- 입력 데이터에서 단순한 초평면(hyperplane)으로 정의되지 않는 더 복잡한 모델을 만들 수 있도록 확장한 것 (복잡한 분류문제)

- 서포트 벡터 머신을 분류와 회귀에 모두 사용할 수 있음 <br>
(선형, 비선형 분류, 회귀, 이상치 탐색에도 사용되는 다목적 머신러닝 모델)
  
```python
# warnig 무시하기
from warnings import filterwarnings
filterwarnings('ignore')

import mglearn
import matplotlib.pyplot as plt
from mglearn.datasets import make_blobs

# 선형적으로 구분되지 않는 클래스를 가진 이진 분류 세트
X, y = make_blobs(centers=4, random_state=8)
# 이진 분류 세트 시각화
# 클래스 2개로 분리, 이진분류
y = y%2  # 홀짝으로 분리하기
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.show()

# 선형 분류
from sklearn.svm import LinearSVC

model = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(model, X)
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.show()
```
- 분류를 위한 선형 모델은 직선으로만 데이터 포인트를 나눌 수 있어서 이런 데이터셋에는 잘 들어 맞지 않음

선형 모델과 비선형 특징
- 직선과 초평면은 유연하지 못하여 저차원 데이터셋에서는 선형 모델이 매우 제한적
- 선형 모델을 유연하게 만드는 한 가지 방법은 특성끼리 곱하거나 특성을 거듭제곱하는 식으로 새로운 특성을 추가하는 것


특성을 추가하여 입력 특성을 확장 (2차원 -> 3차원)
- 특성1에서 유용한 세 번째 특성을 추가하여 확장한 데이터 세트
- (특성0, 특성1) -> (특성0, 특성1, 특성 ** 2)
- 3차원 산점도로 표현
Matplotlib은 mpl_tookits라는 모듈로 3차원 그래프를 그릴 수 있다.
  
```python
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np

# 두 번째 특성을 제곱하여 추가
X_new =  np.hstack([X, X[:,1:]**2])
print(X.shape, X_new.shape)

# 3차원 그래프
fig=plt.figure()
ax = Axes3D(fig, azim=-30, elev=-150)

# y == 0인 포인트를 먼저 그리고 그 다음 y == 1인 포인트를 그림
mask = y ==0
ax.scatter(X_new[mask,0], X_new[mask, 1], X_new[mask,2], c='b')

# ~ 비트 NOT : x의 비트를 뒤집음
mask = y == 1
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='r', marker='^')
plt.xlabel('feature 0')
plt.ylabel('feature 1')
ax.set_xlabel('feature 1 ** 2')
```
SVM을 이용하여 분류
- 선형 모델과 3차원 공간의 평면을 사용해 두 클래스를 구분
- 확장한 3차원 데이터 세트에서 선형 SVM이 만든 결정 경계
```python
model = LinearSVC().fit(X_new, y)
w, b = model.coef_.ravel(), model.intercept_

# 선형 결정 경계 그리기
# 3차원 그래프
fig = plt.figure()
ax = Axes3D(fig, azim=-30, elev=-150)

xx = np.linspace(X_new[:, 0].min(), X_new[:, 0].max(), 50)
yy = np.linspace(X_new[:, 1].min(), X_new[:, 1].max(), 50)

XX, YY = np.meshgrid(xx, yy)
zz = -(w[0] * XX + w[1] * YY + b) / w[2]
print(XX.shape, YY.shape, zz.shape)

ax.plot_surface(XX, YY, zz)

# y == 0인 포인트를 먼저 그리고 그 다음 y == 1인 포인트를 그림
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b')
# ~ 비트 NOT : x의 비트를 뒤집음
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^')
plt.xlabel('feature 0')
plt.ylabel('feature 1')
ax.set_zlabel('feature 1 ** 2')
```
- 원래 특성으로 투영해보면 이 선형 SVM 모델은 더 이상 선형이 아님
- 직선보다 타원에 가까운 모습을 확인

## 커널 기법 (Kernel trick)
- 데이터셋에 비선형 특성을 추가하여 선형 모델을 강력하게 만들 수 있음
- 하지만, 어떤 특성을 추가해야 할지 알 수 없고, 특성을 많이 추가하면 연산 비용이 커짐
- 커널 기법 : 새로운 특성을 만들지 않고 고차원 분류기를 학습시킬 수 있음 (데이터 포인트들의 거리를 계산 - 스칼라 곱)
  
- 고차원 공간 맵핑 방법 : 가우시안 커널, RBF (Radial Basis Function) 커널
- 주로 RBF 커널이 사용

## SVM
- Support Vector : 클래스 사이의 경계에 위치한 데이터 포인트
- 새로운 데이터 포인트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정 -> SVC 객체의 dual_coef_ 속성에 저장

```python
# RBF 커널을 이용한 SVM으로 만든 결정 경계와 서포트 벡터 시각화
from sklearn.svm import SVC

X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:,0], X[:,1], y)

model = SVC(C=1, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(model,X)

# 서포트 벡터
sv = model.support_vectors_
print('사용한 서포트 벡터 수:', len(sv))
sv_y = model.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0], sv[:,1],sv_y, s=15)
```
- 선형이 아닌 부드러운 비선형 경계를 만들어냄
- C와 gamma 두 매개변수를 사용

#### SVM의 튜닝 (C, gamma)
- gamma 매개변수는 가우시안 커널 폭의 역수에 해당
 - gamma 매개변수가 하나의 훈련 샘플이 미치는 영향의 범위를 결정
 - 가우시안 커널의 반경이 클수록 훈련 샘플의 영향 범위도 커짐

- C 매개변수는 선형 모델에서 사용한 것과 비슷한 규제 매개변수
 - 각 포인트의 중요도(정확히는 dual_coef_ 값)를 제한

```python
# C와 gamma 매개변수 설정에 따른 결정 경계와 서포트 벡터 시각화
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for C, axs in zip([-1, 0, 3], axes):
  for gamma, ax in zip([-1,0,1],axs):
    mglearn.plots.plot_svm(C, gamma, ax)
```

<hr>

# 7주차
## <1>
#### SVM 실습 01
유방암 데이터 셋에 SVM 적용

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 데이터 로드
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 모델 학습
model = SVC(C=10000, ).fit(X_train, y_train)

# 평가
model.score(X_train, y_train), model.score(X_test, y_test)

# 유방암 데이터 세트의 특성 값 범위 시각화 (y 축은 로그 스케일)
plt.boxplot(X_train)
plt.yscale('symlog')
plt.show()
```
- 유방암 데이터셋의 특성은 자릿수 자체가 완전히 다름

- 일부 모델(선형 모델 등)에서도 어느 정도 문제가 될 수 있지만, 커널 SVM에서는 영향이 아주 큼

##### SVM을 위한 전처리

- 특성 값의 범위가 비슷해지도록 조정하는 것

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.min(axis=0)
X_train_scaled.max(axis=0)
```
##### 스케일링된 데이터를 SVC에 적용하기
```python
# gamma 파라미저 조정
model = SVC(C=100, gamma=0.02 ).fit(X_train_scaled, y_train)
model.score(X_train_scaled, y_train), model.score(X_test_scaled, y_test)    #(0.9835680751173709, 0.972027972027972)

# C 파라미터 조정
model = SVC(C=1000, gamma=0.01).fit(X_train_scaled, y_train)
model.score(X_train_scaled, y_train), model.score(X_test_scaled, y_test)    #(0.9859154929577465, 0.9790209790209791)
```

## 장단점
- SVM은 강력한 모델이며 다양한 데이터셋에서 잘 작동
- SVM은 데이터의 특성이 몇 개 안 되더라도 복잡한 결정 경계를 만들 수 있음
- 저차원과 고차원의 데이터(즉 특성이 적을 때와 많을 때)에 모두 잘 작동하지만 샘플이 많을 때는 잘 맞지 않음
- 10,000개의 샘플 정도면 SVM 모델이 잘 작동하겠지만 100,000개 이상의 데이터셋에서는 속도와 메모리 관점에서 도전적인 과제
- SVM의 또 하나의 단점은 데이터 전처리와 매개변수 설정에 신경을 많이 써야 한다는 점
- 그런 이유로 대부분 랜덤 포레스트나 그래디언트 부스팅 같은 (전처리가 거의 또는 전혀 필요 없는) 트리 기반 모델을 애플리케이션에 많이 사용
- SVM 모델은 분석하기도 어렵고 예측이 어떻게 결정되었는지 이해하기 어렵고 비전문가에게 모델을 설명하기가 난해함
- 하지만 모든 특성이 비슷한 단위이고(예를 들면 모든 값이 픽셀의 컬러 강도) 스케일이 비슷하면 SVM을 시도해볼 만함
- 커널 SVM에서 중요한 매개변수는 C와 gamma이며 모두 모델의 복잡도를 조정하며 둘 다 큰 값이 더 복잡한 모델을 만듬
- 연관성이 많은 이 두 매개변수를 잘 설정하려면 C와 gamma를 함께 조정

## Decision Tree
- 결정 트리(decision tree)는 분류와 회귀 문제에 널리 사용하는 모델 <br>
(분류와 회귀에 모두 사용)

- 분할(Split)와 가지치기 (Pruning) 과정을 통해 생성 <br>
(Tree를 만들기 위해 예/아니오 질문을 반복하며 학습)

- 다양한 앙상블(ensemble) 모델이 존재한다 <br>
     (RandomForest, GradientBoosting, XGBoost)
  
참고
[트리(Tree)의 개념과 특징을 이해 - 블로그](https://gmlwjd9405.github.io/2018/08/12/data-structure-tree.html)

[트리(그래프) - 나무위키](https://namu.wiki/w/%ED%8A%B8%EB%A6%AC(%EA%B7%B8%EB%9E%98%ED%94%84))

[이진 트리 - 위키백과](https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%A7%84_%ED%8A%B8%EB%A6%AC)

#### 트리(tree) 자료구조

- 노드로 이루어진 자료 구조

- 계층 모델

- 그래프의 한 종류
  - '최소 연결 트리' 라고도 불림
  - 사이클(cycle)이 없는 하나의 연결 그래프(Connected Graph)
  - 또는 DAG(Directed Acyclic Graph, 방향성이 있는 비순환 그래프)의 한 종류

- 예) 파일디렉토리

트리(tree) 자료구조에서의 용어

<center>
 <img src="https://gmlwjd9405.github.io/images/data-structure-tree/tree-terms.png" alt="트리(Tree)용어" width="60%" />

</center>

- 노드(node) : 트리는 노드들의 집합으로 트리를 구성, 보통 (value) 값과 부모 자식의 정보를 가진다.
- 루트 노드(root node): 부모가 없는 노드, 트리는 하나의 루트 노드만을 가진다.
- 단말 노드(leaf node): 자식이 없는 노드, '리프',‘말단 노드’ 또는 ‘잎 노드’라고도 부른다.
- 내부(internal) 노드: 단말 노드가 아닌 노드
- 간선(edge): 노드를 연결하는 선 ('엣지', link, branch 라고도 부름)
- 형제(sibling): 같은 부모를 가지는 노드
- 노드의 크기(size): 자신을 포함한 모든 자손 노드의 개수
- 노드의 깊이(depth): 루트에서 어떤 노드에 도달하기 위해 거쳐야 하는 간선의 수
- 노드의 레벨(level): 트리의 특정 깊이를 가지는 노드의 집합
- 노드의 차수(degree): 하위 트리 개수 / 간선 수 (degree) = 각 노드가 지닌 가지의 수
- 트리의 차수(degree of tree): 트리의 최대 차수
- 트리의 높이(height): 루트 노드에서 가장 깊숙히 있는 노드의 깊이

#### 결정 트리(decision tree)

- 의사결정트리는 학습 데이터로부터 조건식을 만들고 예측할 때는 트리의 루트 노드(root node) 부터 순서대로 조건 분기를 타면서 리프 노드(leaf node)에 도달하면 예측 결과를 내는 알고리즘
- 학습 결과로 IF-THEN 형태의 규칙을 생성 (Split)
- 타깃 값이 한 개인 리프 노드를 순수 노드라고 한다.
- 모든 노드가 순수 노드가 될 때 까지 학습하면 모델이 복잡해지고 훈련 데이터에 과대적합이 된다.
- 새로운 데이터 포인트가 들어오면 해당하는 노드를 찾고, 분류라면 더 많은 클래스를 선택, 회귀라면 평균을 구한다.

특징
- 학습한 모델을 사람이 해석하기 쉽다 -> 시각화 가능
- 입력 데이터에 대한 정규화가 필요 없다
- 범주형 변수나 데이터의 누락값이 있어도 용인된다
- 특정 조건이 맞으면 과적합을 일으키는 경향이 있다
 - 트리가 깊어질 수록 데이터 수가 적어짐
 - 가지치기(pruning)로 깊이를 줄여서 방지
- 비선형 문제에는 우수하지만 선형 분리 문제는 잘 풀지 못한다
- 데이터 분포가 특정 클래스에 쏠려 있으면 잘 풀지 못한다
- 데이터의 작은 변화에도 결과가 크게 바뀌기 쉽다
- 예측 성능은 보통이다
- 배치 학습만 학습할 수 있다.


종류
- 의사 결정 포레스트, 향상된 의사결정 트리, Random Forest, Rotation Forest 등

결정트리(Decision Tree) 과대적합 제어

- 노드 생성을 미리 중단하는 사전가지치기(pre-pruning)와 트리를 만든후에 크기가 작은 노드를 삭제하는 사후가지치기(pruning)가 있다. <br>
    (sklearn은 사전가지치기만 지원)

가지치기(pruning)
- 하나의 가지 (branch)에 동일한 예측 값이 나오는 경우
 - 의사결정트리는 동일 조건에서 가장 간단한 구조여야 한다.

사전가지치기(pre-pruning)

- 트리의 최대 깊이나 리프노드의 최대 개수를 제어

- 노드가 분할하기 위한 데이터 포인트의 최소 개수를 지정

장단점

- 만들어진 모델을 쉽게 시각화할 수 있어 이해하기 쉽다. <br>
    (white box model)

- 각 특성이 개별 처리되기 때문에 데이터 스케일에 영향을 받지 않아 특성의 정규화나 표준화가 필요 없다.

- 훈련데이터 범위 밖의 포인트는 예측 할 수 없다. <br>
    (ex : 시계열 데이터)

- 가지치기를 사용함에도 불구하고 과대적합되는 경향이 있어 일반화 성능이 좋지 않다.

[DecisionTreeClassifier()](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

```
DecisionTreeClassifier(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease, min_impurity_split, class_weight, presort)
```
- criterion : 분할 품질을 측정하는 기능 (default : gini)
- splitter : 각 노드에서 분할을 선택하는 데 사용되는 전략 (default : best)
- <font  color=yellow> max_depth : 트리의 최대 깊이 <br>
(값이 클수록 모델의 복잡도가 올라간다.) </font>
- min_samples_split : 자식 노드를 분할하는데 필요한 최소 샘플 수 (default : 2)
- <font  color=yellow> min_samples_leaf : 리프 노드에 있어야 할 최소 샘플 수 (default : 1) </font>
- min_weight_fraction_leaf : min_sample_leaf와 같지만 가중치가 부여된 샘플 수에서의 비율
- max_features : 각 노드에서 분할에 사용할 특징의 최대 수
- random_state : 난수 seed 설정
- <font  color=yellow> max_leaf_nodes : 리프 노드의 최대수 </font>
- min_impurity_decrease : 최소 불순도
- min_impurity_split : 나무 성장을 멈추기 위한 임계치
- class_weight : 클래스 가중치
- presort : 데이터 정렬 필요 여부

#### Decision Tree 실습 01
붓꽃 데이터 결정트리 만들기
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=11)

# 모델 학습
model = DecisionTreeClassifier().fit(X_train, y_train)
# 평가
model.score(X_train, y_train), model.score(X_test,y_test)
```
**Graphviz 설치(Windows)**
코랩에선 설치 안해도 된다.
1. 아래 링크로 들어가서 graphviz-2.38-win32.msi 다운로드 후 설치

> https://www2.graphviz.org/Packages/stable/windows/10/msbuild/Release/Win32/

2. 시스템 환경변수 PATH 에 다음 경로 추가

> C:\Program Files (x86)\Graphviz2.38\bin

```
# 파이썬 래퍼(Wrapper) 모듈을 별도로 설치
# !pip install graphviz
```

```python
# 결정트리 규칙을 시각화
from sklearn.tree import export_graphviz
import graphviz

# export_graphviz() 를 호출하여 out_file 파라메터의 "tree.dot" 파일을 생성
export_graphviz(model, out_file='tree.dot',
                class_names=iris.target_names,
                feature_names=iris.feature_names,
                filled=True)

# "tree.dot" 파일을 graphviz 가 읽어서 주피터 노트북에 시각화
with open('tree.dot')as f:
  dot_graph = f.read()

display(graphviz.Source(dot_graph))
```
- 각 규칙에 따라 트리의 브랜치(branch) 노드와 말단 리프(leaf) 노드가 어떻게 구성되는지 시각화
- 트리를 조사할 때 많은 수의 데이터가 흐르는 경로를 찾는 것이 중요

트리 시각화 장점
- 알고리즘의 예측이 어떻게 이뤄지는지 이해가 가능
- 비전문가에게 머신러닝 알고리즘을 설명하기에 좋음

```python
# 사이킷런 0.21 버전 이후 맷플롯립 기반 트리 그래프 시각화 함수 추가
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure( figsize=(20,15))
tree.plot_tree(model,
                class_names=iris.target_names,
                feature_names=iris.feature_names,
                filled=True, impurity=False,
               rounded=True)
plt.show()
```
```python
# min_samples_split
# 자식 규칙 노드를 분할해 만들기 위한 최소한의 샘플 데이터 개수
# 모델학습
model = DecisionTreeClassifier(min_samples_split=4).fit(X_train, y_train)
# 평가
model.score(X_train, y_train), model.score(X_test, y_test)

# 결정트리 규칙 시각화
export_graphviz(model, out_file='tree.dot',
                class_names=iris.target_names,
                feature_names=iris.feature_names,
                filled=True, rounded=True, impurity=False)

# "tree.dot" 파일을 graphviz 가 읽어서 주피터 노트북에 시각화
with open('tree.dot') as f:
  dot_graph = f.read()

display( graphviz.Source(dot_graph) )
```
```python
# min_samples_leaf
# 리프 노드가 될 수 있는 샘플 데이터 건수의 최솟값
# 모델학습
model = DecisionTreeClassifier(min_samples_leaf=4).fit(X_train, y_train)

# 평가
model.score(X_train, y_train), model.score(X_test, y_test)

# 결정트리 규칙 시각화
export_graphviz(model, out_file='tree.dot',
                class_names=iris.target_names,
                feature_names=iris.feature_names,
                filled=True, rounded=True, impurity=False)

# "tree.dot" 파일을 graphviz 가 읽어서 주피터 노트북에 시각화
with open('tree.dot') as f:
  dot_graph = f.read()

display( graphviz.Source(dot_graph) )
```
