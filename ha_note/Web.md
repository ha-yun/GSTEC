# 2주차
## <1>
* 박형석 강사님  

### 프로젝트 만들기
https://docs.djangoproject.com/ko/3.2/intro/tutorial01/

pycharm startproject
```
mysite/
    manage.py
    mysite/
        __init__.py
        settings.py
        urls.py
        asgi.py
        wsgi.py
```
이 파일들은  
* The outer mysite/ root directory is a container for your project. Its name doesn’t matter to Django; you can rename it to anything you like.  
* manage.py: Django 프로젝트와 다양한 방법으로 상호작용 하는 커맨드라인의 유틸리티 입니다. 
* mysite/ 디렉토리 내부에는 프로젝트를 위한 실제 Python 패키지들이 저장됩니다.  
  이 디렉토리 내의 이름을 이용하여, (mysite.urls 와 같은 식으로) 프로젝트의 어디서나 Python 패키지들을 임포트할 수 있습니다.
* mysite/__init__.py: Python으로 하여금 이 디렉토리를 패키지처럼 다루라고 알려주는 용도의 단순한 빈 파일입니다.
* mysite/settings.py: 현재 Django 프로젝트의 환경 및 구성을 저장합니다. Django settings에서 환경 설정이 어떻게 동작하는지 확인할 수 있습니다.
* mysite/urls.py: 현재 Django project 의 URL 선언을 저장합니다. Django 로 작성된 사이트의 《목차》 라고 할 수 있습니다.
* mysite/asgi.py: An entry-point for ASGI-compatible web servers to serve your project. See ASGI를 사용하여 배포하는 방법
* mysite/wsgi.py: 현재 프로젝트를 서비스하기 위한 WSGI 호환 웹 서버의 진입점입니다. WSGI를 사용하여 배포하는 방법를 읽어보세요.

### 개발서버
$ python manage.py runserver
* pycharm은 edit configuration에서 port 변경하면 주소가 바뀐다.  
http://127.0.0.1:8000/에서 8000은 포트로 서버의 포트를 변경할 수 있다.
  
### accountapp 만들기
```python
python manage.py startapp accountapp
```
accountapp이라는 디렉토리가 생긴다.
```
accountapp/
    __init__.py
    admin.py
    apps.py
    migrations/
        __init__.py
    models.py
    tests.py
    views.py
```

* accountapp의 views.py에 view할 내용을 적어준다.
```python
from django.http import HttpResponse 
from django.shortcuts import render 

# Create your views here. 
def hello_world(request): 
    return HttpResponse('Hello World :)') 
```

* accountapp 디렉토리 안에 urls.py라는 파일을 생성해준다.
```python
from django.urls import path
urlpatterns = [
    path('hello_world/', hello_world, name="hello_world"),]
```
* 다음은 최상위 폴더 gsweb에서 accountapp.urls를 바라보게 설정해줘야한다.  
gsweb/urls.py를 열고 include()함수를 추가해준다.
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/',include('accountapp.urls')),
]
```
* include() 함수는 다른 URLconf들을 참조할 수 있도록 도와줍니다.  
  Django가 함수 include()를 만나게 되면, URL의 그 시점까지 일치하는 부분을 잘라내고, 남은 문자열 부분을 후속 처리를 위해 include 된 URLconf로 전달합니다.
* 언제 include()를 사용해야 하나요?
    - 다른 URL 패턴을 포함할 때마다 항상 include()를 사용해야 합니다. admin.site.urls가 유일한 예외입니다.
    
* path() 함수에는 2개의 필수 인수인 route 와 view, 2개의 선택 가능한 인수로 kwargs 와 name 까지 모두 4개의 인수가 전달 되었습니다.
    1. path() 인수: route, route 는 URL 패턴을 가진 문자열 입니다.  
       요청이 처리될 때, Django 는 urlpatterns 의 첫 번째 패턴부터 시작하여, 일치하는 패턴을 찾을 때 까지 요청된 URL 을 각 패턴과 리스트의 순서대로 비교합니다.   
       패턴들은 GET 이나 POST 의 매개 변수들, 혹은 도메인 이름을 검색하지 않습니다. 예를 들어, https://www.example.com/myapp/ 이 요청된 경우, URLconf 는 오직 myapp/ 부분만 바라 봅니다. https://www.example.com/myapp/?page=3, 같은 요청에도, URLconf 는 역시 myapp/ 부분만 신경씁니다.
    2. path() 인수: view, Django 에서 일치하는 패턴을 찾으면, HttpRequest 객체를 첫번째 인수로 하고, 경로로 부터 〈캡처된〉 값을 키워드 인수로하여 특정한 view 함수를 호출합니다.
    
    
## <2>
#### 깃에 올릴 때 SECRET KEY를 가려주기
1. 프로젝트 파일에 .env 텍스트 파일을 만들어주고 SECTRET_KEY를 넣어준다.
2. settings.py파일에 들어가서 .env파일의 경로를 라우팅 해준다.  
3. .env파일이 git에 배포되지 않도록 .gitignore파일을 프로젝트 파일에 넣어준다.  
4. .env파일을 .gitignore에 입력해준다.

```python
#1 .env파일에 넣어준다.
SECRET_KEY=django-insecure-$mn=iyjt7o)!2sf8$w9+@1hu0r4_%d16vito)fh$fb*qcngkpu
#2 seettings.py파일에 입력해준다.
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

local_env=open(os.path.join(BASE_DIR, '.env'))
env_list=dict()
while True:
    line=local_env.readline()
    if not line:
        break
    line=line.replace('\n','')
    start=line.find('=')
    key=line[:start]
    value=line[start+1:]
    env_list[key]=value

SECRET_KEY =env_list['SECRET_KEY']

#.gitignore에 .env를 적어준다.
.env
```

## <3>
#### 장고 Template의 extends, include 구문과 render 함수
templates폴더 안에 base.html파일을 생성한다.
accountapp의 view에서 응답해줄 때 템플릿을 가져와서 응답을 할 수 있도록 accountapp/views.py의 hello_world함수를 return render로 바꿔준다.  
```python
#accountapp/views.py
def hello_world(request):
    return render(request, 'base.html')
```
settings.py의 Templates에 'DIRS': [BASE_DIR / 'templates']를 넣어준다.
#### include, extends, block 구문을 이용한 뼈대 html 만들기
- extends, include  
  1. extends : 바탕 block을 만들어 놓고 채우는  
  2. include : 구역이 나누어져 있고 데이터를 가져와서 채워놓는  
        extends -> include  : Response view
     

```html
<!DOCTYPE html>
<html lang="en">

{% include 'head.html' %}

{% include 'header.html' %}

<hr>

{% block content %}
{% endblock %}

<hr>

{% include 'footer.html'%}

</body>
</html>
```
accountapp내부에서 따로 accountapp자체의 template을 보관할 경로를 만들어준다.  
accountapp에 template폴더를 만들고, accountapp이라는 폴더를 만들어준 후 이 안에다가 hello_world.html을 만들어준다.  
- 가독성을 높이기 위한 사전 작업  
accountapp/template/accountapp/hello_world.html
```html
{% extends 'base.html' %}

{% block content %}
    <div style="margin: 2rem; text-align: center">
    <h1>hello world list</h1>
    </div>
{% endblock %}
```
block content안에 있는 내용을 자유롭게 고치면 
* accountapp의 view에서 응답해줄 때 hello_world.html을 가져올 수 있도록 해준다.  
```python
#accountapp/views.py
def hello_world(request):
    return render(request, 'accountapp:hello_world')
```

#### style, 구글 폰트를 통해 Header, Footer 꾸미기
gsweb/templates폴더에 footer, head, header.html을 만들어준다.  
- html파일에서 style을 이용하여 css언어를 사용하기도 한다.  
* border-radius: 1rem 모서리를 둥글게 깍아줌
```html
<div style="height: 10rem; background-color: aquamarine; border-radius: 1rem; margin: 2rem">
</div>
```

# 3주차
## <1>
