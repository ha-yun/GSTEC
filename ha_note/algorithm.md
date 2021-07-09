# 2주차
## <1>
* 양지웅 강사님  

Q. 나머지와 몫이 같은 수 구하기  
숫자 N이 주어졌습니다. N으로 나누었을 때 나머지와 몫이 같은 모든 자연수의 합이 얼마인지 찾는 문제를 풀어봅시다.  
ex) N=1, 해당하는 자연수는 없기에 0으로 간주  
N=2일 경우 3, N=3일 경우 12, N=4일경우 30이 정답  
자연수 N이 주어질 때 나머지와 몫이 같은 모든 자연수의 합을 출력하시오.
```python
#N=1, 0
#N=2, 3 -> 3(2*1+1)
#N=3, 12 -> 4(3*1+1)+8(3*2+2)
#N=4, 30 -> 5(4*1+1)+10(4*2+2)+15(4*3+3)
#나머지는 나누는 수 보다 작은 값

N=int(input())
tt=0
for i in range(1,N):
  tt+=(N*i+i)
print(tt)
```
<hr>

## <2>
* 가우스합
* 1부터 n까지의 합 : (n+1)*(n)/2

Q. 휴식을 위하여  
날씨 좋은 휴일, 인근 공원에서 독서를 하기로 했지만 공원에서는 공사가 진행되고 있었습니다.  
공사 현장은 공원에서 오직 한 군데이고, 그 위치를 (a,b)라고 합니다. 공사 현장에서 R만큼의 거리 미만은 소음이 크기 때문에 독서에 적합하지 않습니다.  
또한, 공원에는 휴식과 독서에 적합한 그늘이 N개 존재합니다.   
각각의 그늘 위치는 (x_i,y_i)입니다.(i번째)  
이상의 정보에서 각 그늘이 독서에 적합한지(공사 현장에서 R이상 떨어진 그늘인지) 판별하는 코드를 작성하시오.
```python
#입력예제
#20 10 10 (a,b,R)  
#3 (N그늘의 수)
#25 10   (그늘의 위치 N개)
#20 15
#70 70

#출력 예제
#noisy
#noisy
#silent

a,b,R=map(int,input().split())
N=int(input())
for _ in range(N):
  x,y=map(int,input().split())
  if (a-x)^2+(b-y)^2<R^2:
    print('noisy')
  else:
    print('silent')
```
<hr>

## <3>
* 점근 표기법(Big-O)

Q. 약수 구하기  
입력 받은 숫자N의 모든 약수를 출력하시오.
```python
#입력예제
#100
#출력예제
#1 2 4 5 10 20 25 50 100
N=int(input())
for i in range(1,N+1):
  if N%i==0:
    print(i)
# -> O(N)

print('-------------------------------------')
#2
N=int(input())
t=[]
for i in range(1,int(N**1/2)+1):
  if N%i==0:
    t.append(i)
    t.append(N//i)
t=set(t)
print(t)


print('======================================')
#3
num=int(input())
sqrt_num=int(num**(1/2))
front_divisior_list=[]
rear_divisor_list=[]

for i in range(1,sqrt_num+1):
  if num%i==0:
    front_divisior_list.append(i)
    if i!=int(num/i):
      rear_divisor_list.append((int(num/i)))
print(front_divisior_list+rear_divisor_list[::-1])
# -> O(N**(1/2))
```
Q. 최대값, 최소값 구하기  
입력받은 정수값들 중 최대값과 최소값을 순서대로 출력하시오.  
단, 파이썬 내장 함수 max,min및 기타 numpy,pandas등의 라이브러리는 쓸 수 없다.
```python
#입력예제
#1 2 100 3 30
#출력예제
#100 1
a=list(map(int,input().split()))
n=0
max=a[n]
min=a[n]
while n<len(a):
  if a[n]>max:
    max=a[n]
  elif a[n]<min:
    min=a[n]
  n+=1
print(max,min)
# ->O(N)

print('===========================')
#2
num=list(map(int,input().split()))
max=num[0]
min=num[0]
for n in num:
  if max<n:
    max=n
  if min>n:
    min=n
print(max,min)
# ->O(N)
```
<hr>

# 3주차
## <1>
###점근표기법(Big-O notation) : 표현할 복잡도의 최대변수에 해당하는 항을 제외하고 나머지 아래 차수의 항 및 모든 항의 계수를 제외시키는 방법  
* 함수의 증가 양상을 다른 함수와의 비교로 표현하는 수론과 해석학의 방법이다.
* 알고리즘의 복잡도를 단순화할 때나 무한급수의 뒷부분을 간소화할 때 쓰인다.

### 시간복잡도
 O(1) < O(log n) < O(n^1/2) < O(n) < O(n log n) < O(n^2) < O(n^3) < O(2^n)

### 공간복잡도
The complexity of an algorithm or a computer program is the amount of memory space required to solve an instance of the computational problem as a function of characteristics of the input.   
it is the memory required by an algorithm to execute a program and produce output.

Operation|Example|Big-O|---
---|---|--- | ---
Index|I[i]|O(1)
Store|I[i]=0|O(1)
Length|len(i)|O(1)
Append|I.append(5)|O(1)
Pop|I.pop()|O(1)
Clear|I.clear()|O(1)|I=[]
Slice|I[a:b]|O(b-a)|I[:]:O(len(I)-0) = O(N)
Extend|I.extend(...)|O(len(...))
Construction|list(...)|O(len(...))
check ==,!=|I1 ==I2|O(N)
Insert|I.insert(i,v)|O(N)
Delete|del I[i] |O(N)
Containment|x in/not in I | O(N)
Copy|I.copy() |O(N)|I[:]-O(N)
Remove|I.remove(...)|O(N)
Pop|I.pop(i)|O(N)|I.pop(0) - O(N)
Extreme value|min(I)/max(I)|O(N)
Reverse|I.reverse()|O(N)
Iteration|for v in I: |O(N)
Sort|I.sort()|O(N Log N)
Multiply|k*I|O(k N)| [0,1,2,3,4]*5 > O(N**2)

Operation|Example|Big-O|-
---|---|---|---
Index|d[k]|O(1)
Store|d[k]=v|O(1)
Length|len(d)|O(1)
Delete|del d[k]|O(1)
get/setdefault|d.method|O(1)
Pop|d.pop(k)|O(1)
Pop item|d.popitem()|O(1)
Clear|d.clear()|O(1)|s={} or =dict()
View|d.keys()|O(1)|=d.values()
Construction|dict(...)|O(len(...))
Iteration|for k in d:|O(N)
<hr>

## <2>
### 스택 : Last in First out
* top, pop, push, empty

Q. 정수를 저장하는 스택을 구현한 다음, 입력으로 주어지는 명령을 처리하는 프로그램을 작성하시오. 명령은 총 다섯 가지 이다.
1. push X : 정수 X를 스택에 넣는 연산이다.
2. pop : 스택에서 가장 위에 있는 정수를 빼고, 그 수를 출력한다. 만약 스택에 들어있는 정수가 없는 경우엔 -1 출력
3. size : 스택에 들어있는 정수의 개수 출력
4. empty : 스택이 비어있으면 1, 아니면 0을 출력
5. top : 스택의 가장 위에 있는 정수 출력, 만약 스택에 들어있는 정수가 없는 경우엔 -1 출력
```python
stack=[]
N=int(input())
for i in range(N):
  P=list(input().split())
  if P[0]=='push':             #push
    stack.append(P[1])

  elif P[0]=='top':           #top
    if len(stack)==0:
      print(-1)
    else:
      print(stack[-1])

  elif P[0]=='size':           #size
    print(len(stack))

  elif P[0]=='empty':           #empty
    if len(stack)==0:
      print(1)
    else:
      print(0)

  elif P[0]=='pop':     #pop
    if len(stack)==0:
      print(-1)
    else:
      print(stack.pop())
```
```python
#append, remove, len 등을 사용하지 않고 만들어보기
N=int(input())
stack=[i for i in range(N)]
cnt=0
for i in range(N):
  P=list(input().split())
  if P[0]=='push':             #push
    stack[cnt]=P[1]
    cnt+=1

  elif P[0]=='top':           #top
    if cnt==0:
      print(-1)
    else:
      print(stack[cnt-1])

  elif P[0]=='size':           #size
    print(cnt)

  elif P[0]=='empty':           #empty
    if cnt==0:
      print(1)
    else:
      print(0)

  elif P[0]=='pop':     #pop
    if cnt==0:
      print(-1)
    else:
      print(stack[cnt-1])
      cnt-=1
```
<hr>

## <3>
```python
#스택
def push(stack,x) : 
  stack.append(x)

def pop(stack):
  if empty(stack)==1:
    return -1
  else:
    return stack.pop()
def size(stack):
  return len(stack)
def empty(stack):
  if len(stack)==0:
    return 1
  else:
    return 0
def top(stack):
  if empty(stack)==1:
    return -1
  else:
    return stack[-1]
```
```python
#저번시간 문제 풀이 210706.ipynb 참고

def run_cmd_with_stack(stack,cmd):
  cmd_type = cmd[0]
  if cmd_type=='push':
    _, num = cmd
    stack.append(int(num))          #stack : global을 쓰던가, 파라미터로 주어야 한다. 
  elif cmd_type == 'pop':
    if len(stack)>0:print(stack.pop())
    else : print(-1)
  elif cmd_type == 'size':
    print(len(stack))
  elif cmd_type == 'empty':
    if len(stack)>0 : print(0)
    else: print(1)
  elif cmd_type == 'top':
    if len(stack)>0:print(stack[-1])
    else: print(-1)
  return stack


stack=[]
n=int(input())
for _ in range(n):
  #"push 2".split() -> ["push",2]
  #"size".split()  -> ["size"]  
  command = input().split()
  stack = run_cmd_with_stack(stack,command)
```
```python
#len , append, pop 을 쓰지 않고 사용하기 풀이

def run_cmd_with_stack(stack,stack_size, cmd):
  cmd_type = cmd[0]

  if cmd_type=='push':
    _,num = cmd
    stack.append(int(num))  
    stack_size += 1 
  elif cmd_type == 'pop':
    if stack_size > 0:
      print(stack.pop()) 
      stack_size -= 1
    else : print(-1)
  elif cmd_type == 'size':
    print(stack_size)
  elif cmd_type == 'empty':
    if stack_size > 0 : print(0)
    else: print(1)
  elif cmd_type == 'top':
    if stack_size > 0 : print(stack[stack_size])
    else: print(-1)
  return stack, stack_size


stack=[]
stack_size = 0
n=int(input())
for _ in range(n):
  command = input().split()
  stack, stack_size = run_cmd_with_stack(stack,stack_size,command)

```
```python
#Class 이용
class Stack:
  def __init__(self):
    self.stack = []
    self.stack_size = 0

  def push(self, num):
    self.stack.append(int(num))
    self.stack_size += 1
  
  def pop(self):
    if self.stack_size > 0:
      self.stack_size -= 1
      return self.stack.pop()
    return -1
  
  def size(self):
    return self.stack_size

  def empty(self):
    if self.stack_size > 0:
      return 0
    else:
      return 1

  def top(self):
    return self.stack[-1]

def run_cmd_with_stack(stack,stack_size, cmd):
  cmd_type = cmd[0]
  if cmd_type=='push':
    _,num = cmd
    my_stack.push(num)
  elif cmd_type == 'pop':
    print(my_stack.pop())
  elif cmd_type == 'size':
    print(my_stack.size())
  elif cmd_type == 'empty':
    print(my_stack.empty())
  elif cmd_type == 'top':
    print(my_stack.top())
  return stack, stack_size

  
my_stack = Stack()
stack_size = 0
n=int(input())
for _ in range(n):
  command = input().split()
  my_stack, stack_size = run_cmd_with_stack(my_stack,stack_size,command)
```

