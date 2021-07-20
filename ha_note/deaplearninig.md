# 2주차
## <1>
1. 데이터를 입력하기
2. 모델 설계하기
3. 최적화하기
4. 학습하기
* 학습률이란? Learning Rate : 학습의 속도를 제어하는 상수
* MSE

<hr>

## <2>
```python
#MSE
#1
y_1 = 21
y_2 = 10
y_3 = 14
y_hat_1 = 15
y_hat_2 = 12
y_hat_3 = 18

a = (y_1 - y_hat_1)**2
b = (y_2 - y_hat_2)**2
c = (y_3 - y_hat_3)**2
print(a, b, c)

import numpy as np
n = 3
mse = (a+b+c)/n

print("MSE : {}".format(np.round(mse,4)))
```
성별 m,f,i를 0,1,2로 변경하면 선형적 문제!  
--> 원핫벡터 One-hot Vector  
```python
#1
rows=['M',0.455,0.365,0.0950] # -> [1,0,0,0.455,0.365,0.0950]
if rows[0]=='M':
  m=[1,0,0]
  rows.pop(0)
  rows=m+rows
elif rows[0]=='F':
  f=[0,1,0]
  rows.pop(0)
  rows=f+rows
else:
  i=[0,0,1]
  rows=i+rows
print(rows)
```
```python
#2 np.zeros 사용
rows = ['I', 0.455, 0.365, 0.095] # ---->> [1, 0, 0, 0.455, 0.365, 0.095]
data = np.zeros(6)
print(data)
if rows[0] == 'M':
    data[0] = 1
elif rows[0] == 'F':
    data[1] = 1
elif rows[0] == 'I':
    data[2] = 1

print('----------------------')
print(data)
print(rows)
print("=========================")
print(data)

data[3:] = rows[1:]
print(data)
```
abalone data로 실습
```python
import csv
with open('./abalone_mini.csv') as csvfile:
  csvreader=csv.reader(csvfile)
  
  rows=[]
  for row in csvreader:
    rows.append(row)
print(rows)
```

```python
#1
m=[1,0,0]
f=[0,1,0]
i=[0,0,1]
rows_s=[]
for r in rows:
  if r[0]=='M':
    r.pop(0)
    r=m+r
    rows_s.append(r)
  elif r[1]=='F':
    r.pop(0)
    r=f+r
    rows_s.append(r)
  else:
    r.pop(0)
    r=i+r
    rows_s.append(r)
print(rows_s)
```
```python
#2 
#enumerate함수 사용
data=np.zeros([5,11])
print(data)
for n,row in enumerate(rows):
  if row[0]=='M': data[n,0]=1
  elif row[0]=='F': data[n,1]=1
  elif row[0]=='I': data[n,2]=1
  data[n,3:]=row[1:]
print(data)
```

```python
#함수에 함수 넣기
def main_exec():
  #import_data()
  model_init()
  #train_and_test()

RND_MEAN = 0
RND_STD = 1
def model_init():
  global weight, bias      # 전역변수화
  weight = np.random.normal(RND_MEAN,RND_STD,size=1)
  bias = np.random.normal(RND_MEAN,RND_STD,size=1)

main_exec()
print('weight : ',weight)       # weight :  [1.53771869]
print('bias : ',bias)           # bias :  [-0.65914163]
```

<hr>

## <3>
```python
def main_exec():
  # import_data()
  model_init()
  run_train()

import random
RND_MEAN=0
RND_STD=1

input_x=10
output_y=1

def model_init():    
  global weight,bias
  weight = np.random.normal(RND_MEAN,RND_STD,size=[input_x,output_y])    #normal distribution 정규분포
  bias = np.random.normal(RND_MEAN,RND_STD,size=[output_y])

def forward_neuralnet(x):
  y_hat = np.matmul(x,weight)+ bias
  return y_hat

print(forward_neuralnet(data[:,:-1]))
```
```python
def forward_postproc(output,y):
  print('output: \n',output)
  print('y: \n',y)
  diff = output - y
  print('diff:\n',diff)
  square = np.square(diff)      #np.square : 배열 원소의 제곱값
  print('square:\n',square)
  mse = np.mean(square)       #np.mean : 평균
  print('mse :\n',mse)
  return mse

forward_postproc(forward_neuralnet(data[:,:-1]),data[:,-1:])
```
run_train() = forward_postproc() + forward_neuralnet()
```python
def run_train(x,y):
  output = forward_neuralnet(x)
  loss = forward_postproc(output,y)
  return output, loss

run_train(data[:,:-1],data[:,-1:])

#독립변수 (마지막 값이 아닌 값들은 모두 독립변수)
print(data[:,:-1])
#종속변수 (마지막 값을 종속변수라고 한다.)
print(data[:,:-1])
```
```python
def main_exec(x,y):
  import_data()
  model_init()
  run_train(x,y)

main_exec(data[:,:-1],data[:,:-1])
```

* 지금까지는 한 행에 하나의 값을 예측// -> 다수의 행에 다수의 열 : 이 문제를 해결하기 위해 차원축소..
```
#softmax
#자연상수e.. 비례의 관점에서 차이를 더 극명하게 알 수 있다.
# 너무 큰 값이 자연상수에 들어가면 수가 너무너무 커진다..  ---> 가장 큰 값을 뽑아서 뺴준다.
# 반대로, 값이 다 음수면 수가 너무너무 작아진다.. 0으로 나눌 수 없다.
# --> 요소들 중에 가장 큰 값을 뽑아준 후 분자와 분모의 모든 요소들을 나눠준다..


#_---------> softmax를 통과시킨 후 가장 높은 인덱스를 출력하는 함수 : np.argmax()
```
* y_hat / target_y / pred_y : (같은말)예측값 
* y_label / y / target_y : y값  
* tensorflow hub에 많은 모델 

#### 행렬곱을 수행하는 대표 함수!  
1. np.matmul()  : 두 배열의 행렬곱 
만약 배열이 2차원보다 클 경우, 마지막 2개의 축으로 이루어진 행렬을 나머지 축에 따라 쌓아놓은 것이라고 생각한다.   
(공식문서에서 matmul권장)  
2. np.dot()  : 두 배열의 내적곱, 만약 a가 N차원 배열이고 b가 2이상의 M차원 배열이라면, dot(a,b)는 a의 마지막 축과 b의 뒤에서 두번째 축과의 내적으로 계산된다.  

* 회귀 - DNN(Deep Nerual Network),ANN(Artificial Neural Network)  
* 이미지 - CNN(Convolution Nerual Network)
* 자연어 - RNN(Recurrent Neural Network)  
* 생성 - GAN(Generative Adversarial Network)

<hr>

# 3주차
## <1>




