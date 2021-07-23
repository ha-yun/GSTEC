# 5주차
## <1>
#### 본인의 데이터를 이용하여 모델링
* 와인품질분류 데이터
1. 데이터불러오기
2. 결측치 확인
3. 원핫벡터
4. 간단한 모델
```python
simple_layer_model = tf.keras.Sequential([
                                          tf.keras.layers.Dense(units = 1, 
                                                                activation  = 'softmax',
                                                                input_shape = (12,))
])              #ERROR loss값이 0만 나온다.
```

<hr>

## <2>, <3>
#### 최적의 파라미터 찾기
1. 데이터 불러오기
2. 데이터 분리하기
3. 파라미터 생성하기
4. 신경망 연산
5. 역전파 연산
6. 평가하기

```python
import pandas as pd
import numpy as np
df = pd.read_csv('/content/abalone_mini.csv')
df.info()
```
main_execute()
- epoch, mini batch, report, train data ratio 

load_dataset()
- 학습 데이터 불러오기   

```python
def main_execute(epoch_count = 10, mb_size = 2, report = 2, train_ratio = 0.8):
  load_dataset()
  weight_initial, bias_initial = init_param()
  losses_mean_row, accs_mean_row, final_acc = train_and_test(epoch_count, mb_size, report, train_ratio)

  return weight_initial, bias_initial, loss_mean_row, accs_mean_row, final_acc


def load_dataset():      # 데이터를 불러오고, 원핫벡터를 실행해줌
  with open('/content/abalone_mini.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    rows = []
    for row in csvreader:
      rows.append(row)

  global data, input_cnt, output_cnt
  input_cnt, output_cnt = 10, 1 
  data = np.zeros([len(rows), input_cnt + output_cnt])

  for n, row in enumerate(rows):
    if row[0] == 'M' : data[n, 0] = 1
    if row[0] == 'F' : data[n, 1] = 1
    if row[0] == 'I' : data[n, 2] = 1
    data[n, 3 : ] = row[1:]   
```

init_param()
- 파라미터 초기화

```python
def init_param():
  global weight, bias 

  weight_initial = []
  bias_initial = []
  weight = np.random.normal(RND_MEAN, RND_STD, size=[input_cnt, output_cnt])  # 초기값 설정하는 방법 1
  bias = np.zeros([output_cnt])                   # 초기값 설정하는 방법 2
  print("Initial Weight Value = \n{}".format(weight))
  print("Initial Bias Value = \n{}".format(bias))
  weight_initial.append(weight)
  bias_initial.append(bias)

  return weight_initial, bias_initial
```
arrange_data()
- 전체 인덱스 탐색
- 데이터 뒤섞기
- 학습 및 테스트 분할 인덱스 탐색
- 미니배치의 수 탐색

```python
def arrange_data(mb_size, train_ratio):
  global shuffle_map, test_begin_index

  shuffle_map = np.arange(data.shape[0])
  np.random.shuffle(shuffle_map)
  mini_batch_step_count = int(data.shape[0] * train_ratio) // mb_size
  test_begin_index = mini_batch_step_count * mb_size
  return mini_batch_step_count
```
get_test_data()
- 테스트 데이터 분할
- 독립 및 종속 변수 생성

```python
def get_test_data():
  test_data = data[shuffle_map[test_begin_index:]]
  return test_data[:, : -output_cnt], test_data[:, -output_cnt:]
```
get_train_data()
- 학습 데이터 분할
- 에폭이 진행됨에 따라 데이터 뒤섞기
- 독립 및 종속 변수 생성

```python
# 셔플링해서 데이터의 순서를 바꿔준다.
def get_train_data(mb_size, n):
  if n == 0:
    np.random.shuffle(shuffle_map[:test_begin_index])

  train_data = data[shuffle_map[mb_size * n :  mb_size * (n+1) ]]
  
  return train_data[:, : -output_cnt], train_data[:, -output_cnt:]
```

```python
def train_and_test(epoch_count, mb_size, report, train_ratio):
  mini_batch_step_count = arrange_data(mb_size, train_ratio)
  test_x, test_y = get_test_data()
  losses_mean_row = []
  accs_mean_row = []

  for epoch in range(epoch_count):
    losses = []
    accs = []
    for n in range(mini_batch_step_count):
      train_x, train_y = get_train_data(mb_size, n)
      loss, acc = run_train(train_x, train_y)
      losses.append(loss)
      accs.append(acc)

    if report > 0 and (epoch + 1) % report == 0:
      acc = run_test(test_x, test_y)
      print('Epoch {} : Train - Loss {:.3f}, Accuracy = {:.3f} / Test - Accuracy = {:.3f}'\
            .format(epoch + 1, np.mean(losses), np.mean(accs), acc))
    
    losses_mean = np.mean(losses)
    accs_mean   = np.mean(accs) * 100   #값이 더 보기 좋게 백분율

    losses_mean_row.append(losses_mean)
    accs_mean_row.append(accs_mean)

  final_acc = run_test(test_x, test_y)
  print('='*30, 'Final TEST', '='*30)
  print('\nFinal Accuracy : {:.3f}'.format(final_acc))

  return losses_mean_row, accs_mean_row, final_acc
```

forward_neuralnet() : 순전파 연산
```python
def forward_neuralnet(x):
  y_hat = np.matmul(x, weight) + bias
  return y_hat, x
```

forward_postproc() : 손실함수 연산
```python
def forward_postproc(y_hat, y):
  diff = y_hat - y
  square = np.square(diff)
  loss = np.mean(square)

  return loss
```

eval_accuracy() : 평가
```python
def eval_accuracy(y_hat, y):
  mdiff = np.mean(np.abs((y_hat - y) / y))
  return 1 - mdiff

```

backprop_neuralnet() : 파라미터 경사하강법 연산, 파라미터 갱신
```python
def backprop_neuralnet(G_output, x):
    global weight 

    x_transpose = x.transpose()
    G_w = np.matmul(x_transpose, G_output)

    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b
```

backprop_postproc() : 손실함수 미분
```python
def backprop_postproc(diff):
    M_N = diff.shape

    g_mse_square = np.prod(M_N) / np.prod(M_N)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_diff = g_mse_square * g_square_diff
    G_output = g_diff_output * G_diff
    return G_output
```

run_test()
- 순전파 연산(결과 확인)
- 평가 진행(정확도 측정)

run_train()
- 순전파 연산, 손실함수 연산, 정확도 연산
- 손실함수 미분 및 경사하강법 연산
- 파라미터 갱신
```python
def run_train(x, y):
  y_hat, aux_nn_x = forward_neuralnet(x)    #aux는 보조정보
  loss, aux_pp_y = forward_postproc(y_hat, y) #로스, 편차값
  accuracy = eval_accuracy(y_hat, y)

  G_output = backprop_postproc(aux_pp_diff)
  backprop_postproc(G_output, aux_nn_x)
  
  return loss, accuracy

def run_test(x, y):
  y_hat, _ = forward_neuralnet(x)   #test라서 반환되는 x값은 필요없다.
  accuracy = eval_accuracy(y_hat, y)

  return accuracy
  ```
