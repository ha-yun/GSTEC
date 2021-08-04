# 5주차
## <1>
* 카멜케이스 : camelCase

* 스네이크케이스 : snake_case

### Deque
```python
#deque
from collections import deque

deque_obj = deque()

print(type(deque_obj))      #<class 'collections.deque'>
print(deque_obj)            #deque([])
```
```python
deque_obj.append(50)
deque_obj.append(42)
deque_obj.append(30)
deque_obj.appendleft(100)
print(deque_obj)    #deque([100, 50, 42, 30])

deque_obj.pop()
deque_obj.popleft()
print(deque_obj)    #deque([50, 42])
```
#### Deque 사용해서 문제 풀기
```python
from collections import deque
class Deque():
  def __init__(self):
    self.deque = deque()

  def push(self, x):
    self.deque.append(x)

  def pop(self):
    if self.size() == 0:
      return -1
    else:
      return self.deque.popleft()

  def size(self):
    return len(self.deque)

  def empty(self):
    if self.size() == 0 :
      return 1
    else:
      return 0

  def front(self):
    if self.size() == 0 :
      return -1
    else:
      return self.deque[0]
  
  def back(self):
    if self.size() == 0:
      return -1
    else:
      return self.deque[-1]


def run_cmd_with_deque(my_deque, cmd):
    cmd_type = cmd[0]
    if cmd_type == "push":
        _, num = cmd 
        my_deque.push(num)
    elif cmd_type == "pop":
        print(my_deque.pop())
    elif cmd_type == "size":
        print(my_deque.size())
    elif cmd_type == "empty":
        print(my_deque.empty())
    elif cmd_type == "front":
        print(my_deque.front())
    elif cmd_type == "back":
        print(my_deque.back())
    return my_deque
```
```python
N = int(input())
my_deque = Deque()
for _ in range(N):
    command = input().split()
    my_deque = run_cmd_with_queue(my_deque, command)
    my_deque
```

<hr>

## <2>
```python
# collections deque로 풀이
from collections import deque

class StackAndQueue:
    def __init__(self, data_type="stack"):
        self.array = deque()
        self.data_type = data_type

    def push(self, num):
        self.array.append(num)

    def pop(self):
        if self.is_empty():
            return -1

        if self.is_stack():
            return self.array.pop()
        
        return self.array.popleft()

    def size(self):
        return len(self.array)

    def empty(self):
        return int(self.is_empty())

    def top(self):
        if not self.is_stack() or self.is_empty():
            return -1

        return self.array[-1]

    def front(self):
        if self.is_stack() or self.is_empty():
            return -1

        return self.array[0]

    def back(self):
        if self.is_stack() or self.is_empty():
            return -1

        return self.array[-1]
        # return self.get_last_val()

    # def get_last_val(self):
    #     return self.array[-1]

    def is_empty(self):
        return self.size() == 0

    def is_stack(self):
        return self.data_type == "stack"

def run_cmd_with_deque(command, data_obj):
    cmd_type = command[0]

    if cmd_type == "push":
        _, num = command
        data_obj.push(num)
    
    elif cmd_type == "pop":
        print(data_obj.pop())

    elif cmd_type == "empty":
        print(data_obj.empty())
    
    elif cmd_type == "size":
        print(data_obj.size())
    
    elif cmd_type == "top":
        print(data_obj.top())
    
    elif cmd_type == "front":
        print(data_obj.front())
    
    elif cmd_type == "back":
        print(data_obj.back())

data_type = input()
n = int(input())
data_obj = StackAndQueue(data_type)

for _ in range(n):
    run_cmd_with_deque(input().split(), data_obj) # ["push", "3"], ["size"]
```

### 정렬(Sort)
* 각각의 모든 요소를 이미 정렬된 앞의 배열 부분과 비교하여 적절한 위치로 옮겨 삽입한다.
### 선택정렬 (selection sort)
* 주어진 리스트 범위 내에서 (0부터 끝까지) 최솟값을 찾는다.
* 처음 위치에 최솟값을 넣는다(교환).
* 다음 인덱스(1부터 끝까지) 범위 내에서 최솟값을 찾는다.
* 처음 위치에 최솟값을 넣는다 (반복)
#### 버블정렬(Bubble Sorts)
* 인접한 두 값을 비교하여 정렬하는 방법
* 비교적 구현이 쉬움.
* 처음부터 끝까지 두 값 중 더 큰값을 오른쪽으로 바꾸면서 나아감.
* 매일 오른쪽에 제일 큰 값이 간다.

#### 위의 정렬들 다 대부분 O(N^2)만큼 걸린다.

<hr>

## <3>
```python
def insertion_sort(n_list):
  i=0
  while i != len(n_list)-1:
    if n_list[i] != min(n_list[i:]):
      n_list.append(n_list[i])
      n_list.pop(i)
    else:
      i+=1
  return n_list

def selection_sort(n_list):  
  m_list=[]
  while len(n_list)>0:
    mm=min(n_list)
    m_list.append(mm)
    n_list.remove(mm)
  return m_list

def bubble_sort(n_list):
  for i in range(len(n_list)-1):
    if n_list[i] > n_list[i+1]:
      n_list[i+1],n_list[i]=n_list[i],n_list[i+1]
  return n_list

#-------------------------------------------------

n = int(input())
num_list = []

for _ in range(n):
    num = int(input())
    num_list.append(num)

insertion_sorted_list = insertion_sort(num_list)
print(" ".join(map(str, insertion_sorted_list)))

selection_sorted_list = selection_sort(num_list)
print(" ".join(map(str, selection_sorted_list)))

bubble_sorted_list = bubble_sort(num_list)
print(" ".join(map(str, bubble_sorted_list)))
```

<hr>

# 6주차
## <1>


