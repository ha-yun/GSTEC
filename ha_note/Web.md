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
    
<hr>

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
<hr>

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

<hr>

# 3주차
## <1>
#### Static 설정 및 CSS 파일 분리
static은 자주 변경되지 않는 파일들을 말함. 앱별로 따로 관리하게 될것.  
1. 최상위 폴더에 static폴더 생성
2. static폴더안에 base.css생성
3. header와 footer를 꾸며준다면  
```html
< div class="pragmatic_header_logo" >
```
4. base.css에 header와 footer의 꾸밀 class를 적어준다.
```html
   .pragmatic_header_logo{
    margin : 2rem 0;
    font-family: 'Yomogi', cursive;
}
```
5. head.html에 base.css를 불러온다.  
(내부 소스이므로 템플릿 언어로 {% load static % }이용  
   <  hr >은 줄을 그어준다.
   
```html
{% load static %}  
< link rel="stylesheet" type="text/css" href="{% static "base.css" %}" >
```
6. django는 static을 사용하지 않는다고 전제하므로 static의 경로를 만들어줘야 한다.  
settings.py에
```html
STATICFILES_DIRS = [
    BASE_DIR / "static",
]
```
#### bootstrap
* front-end library로 css를 일일이 만지지 않아도, class만 지정하면 디자인을 사용할 수 있다. , 다양한 디자인 요소들!  
외부 소스이므로 불러와야한다. -> 외부 소스를 가져오는 위치는 head.html  
```html
{# BOOTSTRAP LINK #}  
    < link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous" >  
```

### 폰트 불러오기
* 외부에서 폰트를 불러온다면 (google font)
```html
 {# HEADER FONT #}  
    < link rel="preconnect" href="https://fonts.googleapis.com" >  
    < link rel="preconnect" href="https://fonts.gstatic.com" crossorigin >  
    < link href="https://fonts.googleapis.com/css2?family=Yomogi&display=swap" rel="stylesheet" >  
-> 사용할 폰트를 긁어오고, font-family를 입력해줘야 한다.
.pragmatic_footer_logo{  margin : 2rem 0; font-family: 'Yomogi', cursive;}
```

<hr>

## <2>
#### CSS  
* DISPLAY Attribute : block,inline,inline-block, None  
* Visibility : hidden,   
* size : px, em, rem, %      -->font size에 따라 변한다.?
* rem -> 값이 유연하게 변하는  
### css(html에 css꾸미기)   
1. inline  
2. internal style sheet < style  >
3. external style sheet href 외부스타일시트를 이용한..  
```html
<!--templates파일의 accountapp파일의 hello_world.html에 size test-->
  <style>

    .testing{
        background-color: blue;
        height: 48px;
        width: 48px;
        margin: 1rem;
        border-radius: .5rem;
    }
    </style>
<!-- internal style sheet-->
    <div class="testing" div style="display : inline-block">block</div>
    <div class="testing" div style="display : inline-block; width: 3rem; height: 3rem">block</div>
    <div class="testing" div style="display : inline-block; width: 3em; height: 3em">block</div>
    <div class="testing" div style="display : inline-block; width: 30%; height: 30%">block</div>

<!--rem, em, %, px test해보기-->

```
### DataBase model
```python
#추가로 아주 간단한 모델. .? 만들기
#모델의 활성화
#accountapp package안에 models.py에 들어간다.
from django.db import models

# Create your models here.

class HelloWorld(models.Model):
    text = models.CharField(max_length=255, null=False)

#아주 간단한 모델
#terminal에 
python manage.py makemigrations 0001_iniatial
python manage.py migrate
#를 해주면
#migration package안에 0001_iniatial.py가 생성된다.

#당신을 위해 migration들을 실행시켜주고, 자동으로 데이터베이스 스키마를 관리해주는 migrate 명령어가 있습니다
```
migrate 명령은 아직 적용되지 않은 마이그레이션을 모두 수집해 이를 실행하며(Django는 django_migrations 테이블을 두어 마이그레이션 적용 여부를 추적합니다)  
이 과정을 통해 모델에서의 변경 사항들과 데이터베이스의 스키마의 동기화가 이루어집니다.
마이그레이션은 매우 기능이 강력하여, 마치 프로젝트를 개발할 때처럼 데이터베이스나 테이블에 손대지 않고도 모델의 반복적인 변경을 가능하게 해줍니다.  
동작 중인 데이터베이스를 자료 손실 없이 업그레이드 하는 데 최적화 되어 있습니다. 튜토리얼의 나머지 부분에서 이 부분을 조금 더 살펴보겠습니다만, 지금은 모델의 변경을 만드는 세 단계의 지침을 기억하세요.

<hr>

## <3>
> migration  
> makemigrations : 모델 변경을 감지하고 변경사항을 반영할 파일 생성  
manage.py -> migrate ->database

>HTTP Protocol  : GET, POST   
* GET  
주소창?key=value(쿼리데이터)  
주소창에 추가적인 데이터를 넣는 방식  
주소창에 넣는데 한계가 있다.  
* POST  
post + body  
http body 안에다가 추가적인 데이터를 넣는다.  
  
return render(request, 'accountapp/hello_world.html', context={'text':'POST METHOD!'})  
context는 문맥.

< h3  > {{     }} <  /h3 > : 단순 변수 출력 쌍괄호
```python
#view.py
def hello_world(request):
    if request.method == "POST":
        return render(request, 'accountapp/hello_world.html', context={'text':'POST METHOD!'})
    else:
        return render(request, 'accountapp/hello_world.html', context={'text':'GET METHOD!'})
```
```html
#templates/accountapp/hello_world.html
<div style="margin: 2rem; text-align: center">
  <h1>METHOD</h1>

  <h3>{{ text }}</h3>

    <div style="margin: 2rem; text-align: center">
    <h1>METHOD</h1>
    <form action="/accounts/hello_world/"method="post">       <!--action에 요청할 post html주소 입력-->
        {% csrf_token %}                                      <!--장고에서 post를 요청할 때는 반드시 넣어주어야 정상적으로 작동-->

        <input class="btn btn-primary rounded-pill px-2 py-2" type="submit">     <!--px : x축 padding늘려주고, py : y축 padding늘려줌 5까지 지원-->
                                                                                 <!--bootstrap의 button에서 class를 가져온다, rounded-pill은 버튼을 round한 모양으로 바꿔줌-->
    </form>


<input type="text" name="hello_world_input">
```
```python
#views.py에 
    if request.method == "POST":

        temp = request.POST.get('hello_world_input')

        return render(request, 'accountapp/hello_world.html', context={'text':temp})
```

데이터베이스 모델을 하나 생성한후 text에 입력한 데이터가 모델에 저장되게 한 후 출력하게 하는 방법
#### hello_world.html
<  input type="text" name="hello_world_input"  >를 추가해준다.

```python
#views.py
from accountapp.models import HelloWorld
def hello_world(request):
    if request.method == "POST":

        temp = request.POST.get('hello_world_input')

        new_hello_world = HelloWorld()        #models.py에 만들었던 모델
        new_hello_world.text = temp
        new_hello_world.save()

        return render(request, 'accountapp/hello_world.html', context={'hello_world_output':new_hello_world})
    else:
        return render(request, 'accountapp/hello_world.html', context={'text':'GET METHOD!'})
```

#### hello_world.html
{% if hello_world_output %}     
<h3>{{ hello_world_output.text }}</h3>
{% endif %}

#### if, endif를 사용하여 hello_world_output값이 있으면 출력해준다.

#### database에 들어가서 sqlite driver를 다운로드 하면 데이터베이스를 볼 수 있다.

<hr>

# 4주차
## <1>
#### POST 통신을 이용한 DB 데이터 저장
 accountapp/templates/accountapp/hello_world.html  
 ```html
  <h1 style="font-family: 'Yomogi', cursive"
class="m-5">hello world list</h1>

    <form action="/accounts/hello_world/"method="post">
        {% csrf_token %}
        <div class="input-group mb-3 w-50 m-auto">
            <input type="text" class="form-control"
                name="hello_world_input">
            <button class="btn btn-dark" type="submit">
                Button
            </button>
        </div>

 {% if hello_world_list %}
        {% for hello_world in hello_world_list %}
            <h3>{{ hello_world.text }}</h3>
        {% endfor %}
```
accountapp/views.py 
```python
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
from django.urls import reverse

from accountapp.models import HelloWorld


def hello_world(request):
        new_hello_world.text = temp
        new_hello_world.save()


        return HttpResponseRedirect(reverse('accountapp:hello_world'))

    else:
        hello_world_list = HelloWorld.objects.all()
        return render(request, 'accountapp/hello_world.html', context={'hello_world_list': hello_world_list})

```
#### Create View
accountapp/views.py
```python

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

# Create your views here.
from django.urls import reverse
from django.urls import reverse, reverse_lazy
from django.views.generic import CreateView

from accountapp.models import HelloWorld

def hello_world(request):
        hello_world_list = HelloWorld.objects.all()
        return render(request, 'accountapp/hello_world.html', context={'hello_world_list': hello_world_list})


class AccountCreateView(CreateView):
    model = User
    form_class = UserCreationForm
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/create.html'

```

<hr>

## <2>
accountapp/urls.py
```python
from django.urls import path

from accountapp.views import hello_world
from accountapp.views import hello_world, AccountCreateView

app_name ='accountapp'

urlpatterns = [
    path('hello_world/', hello_world, name="hello_world"),

    path('create/', AccountCreateView.as_view(), name='create')
]
```
accountapp/views.py
```python
    model = User
    form_class = UserCreationForm
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/create.html'
```
accountapp/templates/accountapp/create.html
```html
{% extends 'base.html' %}

{% block content %}

    <div class="text-center">
        <div>
            <h4 class="m-5">Sign Up</h4>
        </div>
        <div>
            <form action="{% url 'accountapp:create' %}" method="post">
                {% csrf_token %}
                {{ form }}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```
accountapp/templates/accountapp/login.html
```html
{% extends 'base.html' %}

{% block content %}

    <div class="text-center">
        <div>
            <h4 class="m-5">Login</h4>
        </div>
        <div>
            <form action="{% url 'accountapp:loogin' %}" method="post">
                {% csrf_token %}
                {{ form }}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %} 

```
accountapp/urls.py
```python
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from accountapp.views import hello_world, AccountCreateView

urlpatterns = [
    path('hello_world/', hello_world, name="hello_world"),

    path('login/', LoginView.as_view(template_name='accountapp/login.html'), name='loogin'),

    path('logout/', LogoutView.as_view(), name='logout'),

    path('create/', AccountCreateView.as_view(), name='create')
]
```
gsweb/settings.py
```python

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
from django.urls import reverse_lazy

BASE_DIR = Path(__file__).resolve().parent.parent

local_env=open(os.path.join(BASE_DIR, '.env')) #base_dir가 프로젝트 경로를 의미,

# https://docs.djangoproject.com/en/3.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGIN_REDIRECT_URL = reverse_lazy('accountapp:hello_world')

LOGOUT_REDIRECT_URL = reverse_lazy('accountapp:login')
```
templates/header.html
```html
    <div class="pragmatic_header_button">
        <span>nav1</span>
        <span>nav2</span>

        <span>
            <a href="{% url 'accountapp:loogin' %}">
                Login
            </a>
        </span>
        <span>
            <a href="{% url 'accountapp:create' %}">
                Signup
            </a>
        </span>
    </div>

</div>
```

<hr>

## <3>
#### Bootstrap
accountapp/templates/accountapp/create.html
```html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5">Sign Up</h4>
        </div>
        <div>
            <form action="{% url 'accountapp:create' %}" method="post">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
```
accountapp/templates/accountapp/login.html
```html
{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5">Login</h4>
        </div>
        <div>
            <form action="{% url 'accountapp:login' %}" method="post">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
```
accountapp/urls.py
```python
urlpatterns = [
    path('hello_world/', hello_world, name="hello_world"),

    path('login/', LoginView.as_view(template_name='accountapp/login.html'), name='login'),

    path('logout/', LogoutView.as_view(), name='logout'),
```
static/base.css
```css
.mw-500{
    max-width: 500px;
    padding: 1rem 1.5rem;
} 
```
templates/header.html
```html
    <div class="pragmatic_header_button">
        <span>nav1</span>
        <span>nav2</span>
        {% if not user.is_authenticated  %}
        <span>
            <a href="{% url 'accountapp:loogin' %}">
            <a href="{% url 'accountapp:login' %}">
                Login
            </a>
        </span>
<h1>Pragmatic</h1>
                Signup
            </a>
        </span>
        {% else %}
        <span>
            <a href="{% url 'accountapp:logout' %}">
                Logout
            </a>
        </span>
        {% endif %}
    </div>

</div>
```
accountapp/templates/accountapp/login.html
```html
{% load bootstrap4 %}
```
templates/base.html
```html
<body style="font-family: 'NanumSquareR'">
```
templates/head.html
```html
{#    static link#}
    <link rel="stylesheet" type="text/css" href="{% static "base.css" %}">

        <style>
        @font-face {
            font-family: 'NanumSquareR';
            src: local('NanumSquareR'),
            url("{% static 'fonts/NanumSquareR.ttf' %}") format("opentype");
        }
        @font-face {
            font-family: 'NanumSquareEB';
            src: local('NanumSquareEB'),
            url("{% static 'fonts/NanumSquareEB.otf' %}") format("opentype");
        }
        @font-face {
            font-family: 'NanumSquareB';
            src: local('NanumSquareB'),
            url("{% static 'fonts/NanumSquareB.otf' %}") format("opentype");
        }
        @font-face {
            font-family: 'NanumSquareR';
            src: local('NanumSquareR'),
            url("{% static 'fonts/NanumSquareR.otf' %}") format("opentype");
        }
    </style>

</head> 
```
#### Detailview
accountapp/urls.py
```python
from django.contrib.auth.views import LoginView, LogoutView
from django.urls import path

from accountapp.views import hello_world, AccountCreateView, AccountDetailView

app_name ='accountapp'

    path('logout/', LogoutView.as_view(), name='logout'),

    path('create/', AccountCreateView.as_view(), name='create'),

    path('detail/<int:pk>', AccountDetailView.as_view(), name='detail')
]

```
accountapp/views.py
```python

# Create your views here.
from django.urls import reverse, reverse_lazy
from django.views.generic import CreateView
from django.views.generic import CreateView, DetailView

from accountapp.models import HelloWorld

class AccountCreateView(CreateView):
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/create.html'

class AccountDetailView(DetailView):
    model = User
    context_object_name = 'target_user'
    template_name = 'accountapp/detail.html'
```
templates/header.html
```html
        {% else %}
        <span>
            <a href="{% url 'accountapp:detail' pk=user.pk %}">
                Mypage
            </a>
        </span>
        <span>
            <a href="{% url 'accountapp:logout' %}">
                Logout
```
accountapp/templates/accountapp/detail.html
```html
{% extends 'base.html' %}
{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5">{{ target_user.username }}</h4>
        </div>
        <div>
            {{ target_user.date_joined }}
        </div>
    </div>

{% endblock %}
```
templates/head.html
```html
{#    static link#}
    <link rel="stylesheet" type="text/css" href="{% static "base.css" %}">

        <style>
    <style>
        @font-face {
            font-family: 'NanumSquareR';
            src: local('NanumSquareR'),
            url("{% static 'fonts/NanumSquareR.ttf' %}") format("opentype");
            url("{% static 'fonts/NanumSquareR.otf' %}") format("opentype");
        }
        @font-face {
            font-family: 'NanumSquareEB';
            url("{% static 'fonts/NanumSquareB.otf' %}") format("opentype");
        }
        @font-face {
            font-family: 'NanumSquareR';
            src: local('NanumSquareR'),
            url("{% static 'fonts/NanumSquareR.otf' %}") format("opentype");
            font-family: 'NanumSquareL';
            src: local('NanumSquareL'),
            url("{% static 'fonts/NanumSquareL.otf' %}") format("opentype");
        }
    </style>

```