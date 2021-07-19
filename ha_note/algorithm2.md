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
