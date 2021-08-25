# 5주차
## <1>
##### AccountUpdateView
accountapp/templates/accountapp/detail.html
```html
    <div>
        <a href="{% url 'accountapp:update'  pk=target_user.pk %}">
            Update Info
        </a>
    </div>
```
accountapp/templates/accountapp/update.html
```html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Update Info </h4>
        </div>
        <div>
            <form action="{% url 'accountapp:update'  pk=target_user.pk %}" method="post">
                {% csrf_token %}
                {% bootstrap_form form %}
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
path('update/<int:pk>', AccountUpdateView.as_view(), name='update'),
# name은 reverse url
```
URL의 이름공간 정하기¶  
튜토리얼의 프로젝트는 polls라는 앱 하나만 가지고 진행했습니다. 실제 Django 프로젝트는 앱이 몇개라도 올 수 있습니다. Django는 이 앱들의 URL을 어떻게 구별해 낼까요?   
예를 들어, polls 앱은 detail이라는 뷰를 가지고 있고, 동일한 프로젝트에 블로그를 위한 앱이 있을 수도 있습니다. Django가 {% url %} 템플릿태그를 사용할 때, 어떤 앱의 뷰에서 URL을 생성할지 알 수 있을까요?  
정답은 URLconf에 이름공간(namespace)을 추가하는 것입니다. polls/urls.py 파일에 app_name을 추가하여 어플리케이션의 이름공간을 설정할 수 있습니다.  
polls/urls.py¶

accountapp/views.py
```python
class AccountUpdateView(UpdateView):
    model = User
    form_class = UserCreationForm
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/update.html'
```

#### AccountDeleteView
accountapp/templates/accountapp/delete.html
```html
{% extends 'base.html' %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Quit </h4>
        </div>
        <div>
            <form action="{% url 'accountapp:delete' pk=target_user.pk %}" method="post">
                {% csrf_token %}
                <div class="m-5">
                    <input type="submit" class="btn btn-danger rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```
accountapp/templates/accountapp/detail.html
```html
    <div>
        <a href="{% url 'accountapp:delete' pk=target_user.pk %}">
            Quit
        </a>
    </div>
```
accountapp/urls.py
```python
    path('delete/<int:pk>', AccountDeleteView.as_view(), name='delete'),
```
accountapp/views.py
```python
class AccountDeleteView(DeleteView):
    model = User
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/delete.html'
```
#### my page를 대상 유저만 볼 수 있도록 해주기 : if문 활용
accountapp/templates/accountapp/detail.html
```html
        {% if user == target_user %}
    <div>
        <a href="{% url 'accountapp:update'  pk=target_user.pk %}">
            Update Info
            <h4 class="m-5">{{ target_user.username }}</h4>
            Quit
        </a>
    </div>
    {% endif %}
```

<hr>

## <2>
#### mypage의 update info에서 아이디와 비밀번호 모두 수정할 수 있기 때문에 아이디는 수정하지 못하도록 해주기
UserCreationForm에서 아이디입력은 비활성화
accountapp디렉토리에 forms.py파일을 하나 생성
```python
# accountapp\forms.py
class AccountCreationForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fields['username'].disabled = True
        
```
```python
# gsweb\accountapp\views.py
class AccountUpdateView(UpdateView):
    model = User
    form_class = AccountCreationForm
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/update.html'
```
def hello_world  #request를 보내고 출력하는 함수에서, request에 허락된 사람만 볼 수 있도록 해주기
```python
# gsweb\accountapp\templates\accountapp\hello_world.html
def hello_world(request):
    if request.user.is_authenticated:         ### is_authenticated
        if request.method == "POST":

            temp = request.POST.get('hello_world_input')

            new_hello_world = HelloWorld()
            new_hello_world.text = temp
            new_hello_world.save()

            return HttpResponseRedirect(reverse('accountapp:hello_world'))

        else:
            hello_world_list = HelloWorld.objects.all()
            return render(request, 'accountapp/hello_world.html', context={'hello_world_list': hello_world_list})
    else:
        return HttpResponseRedirect(reverse('accountapp:login'))
```
AccountUpdateView와 AccountDeleteView도 해당 유저아니면 들어갈 수 없도록 설정해준다.
```python
# gsweb\accountapp\views.py
class AccountUpdateView(UpdateView):
    model = User
    form_class = AccountCreationForm
    context_object_name = 'target_user'
    template_name = 'accountapp/update.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse('accountapp:login'))

    def post(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse('accountapp:login'))

class AccountDeleteView(DeleteView):
    model = User
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:ha_world')
    template_name = 'accountapp/delete.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse('accountapp:login'))

    def post(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return super().post(request, *args, **kwargs)
        else:
            return HttpResponseRedirect(reverse('accountapp:login'))
```
if request.user.is_authenticated만 설정해주면 a유저가 b유저의 update, delete창에 들어갈 수 있기 때문에 대상 유저만 들어갈 수 있도록 조건을 추가해준다.
```python
    def get(self, request, *args, **kwargs) :
        if request.user.is_authenticated and self.get_object() == request.user:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()

    def post(self, request, *args, **kwargs):
        if request.user.is_authenticated and self.get_object() == request.user:
            return super().get(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
```
#### Decorator 함수
시스템이 커질수록 관리하기 편하고, 가독성이 좋아진다.
```python
def decorator(func):
    def decorated():
        print('Start WORLD')
        func()
        print('The END.')
    return decorated

@decorator
def hello_world():
    print('Start World')
    print('Hello World!!')
    print('The End.')

hello_world()
```

<hr>

## <3>
* django에서 로그인 데코레이터를 이미 제공  
* @login_required  
    - 기본적으로 accounts / login으로 넘어간다.  
* @login_required(login_url=reverse_lazy('accountapp:login'))  
    - 추가적인 인자를 넣을 수 있다.
* login_required는 함수에만 적용가능, 클래스안의 메소드에는 적용되지 않는다.  
    - 메소드를 변환해주어야 하는데 장고에서 모두 지원해줌.  
> @method_decorator  메소드로 데코레이터를 바꿔준다.
 @method_decorator(login_required)추가적인 인자를 넣어줌  
 @method_decorator(login_required, 'get') get메소드에 적용
 
```python
# gsweb\accountapp\decorators.py 파일 생성
from django.contrib.auth.models import User
from django.http import HttpResponseForbidden


def account_ownership_required(func):
    def decorated(request, *args, **kwargs):
        target_user = User.objects.get(pk=kwargs['pk'])
        if target_user == request.user:
            return func(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
    return decorated
```
```python
# gsweb\accountapp\views.py
has_ownership = [login_required, account_ownership_required]    #로그인이 되었는지, target_user와 유저가 같은 사람인지 확인, decorator함수 이요

@method_decorator(has_ownership, 'get')
@method_decorator(has_ownership, 'post')
class AccountUpdateView(UpdateView):
    model = User
    form_class = AccountCreationForm
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/update.html'

@method_decorator(has_ownership, 'get')
@method_decorator(has_ownership, 'post')
class AccountDeleteView(DeleteView):
    model = User
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/delete.html'

```
> terminal 창  
python manage.py createsuperuser  
hyeon  
hyeon@admin.com  
비밀번호입력하고 http://127.0.0.1:8000/admin/로 접속하면 관리창이 뜬다. 

<hr>

# 6주차
## <1>
### profileapp 생성하기
```python
# profileapp 생성, 터미널창에
python manage.py startapp profileapp

# settings.py
INSTALLED_APPS = [
    '...',
    'bootstrap4',
    'accountapp',
    'profileapp',     # profileapp을 추가해준다.
]

# gsweb\urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/',include('accountapp.urls')),
    path('profiles/', include('profileapp.urls'))   # profileapp.urls 추가
]

# profileapp에 urls.py를 생성해준다.
# 기본적으로 앱을 만들 때 urls.py는 생성되지 않는다.
urlpatterns = [

]
```
* on_delete는 일대일로 연결해놓았던 유저가 삭제된다면 어떻게 할것인가라는 뜻                                                            
* models.CASCADE는 삭제(종속)한다는 의미 -> 유저객체가 사라진다면 profile도 삭제
* models.SET_NULL는 user가 삭제된다면 NULL로 유저를 변경한다는 뜻
* related_name은 어떤 이름으로 불러올건지
* upload_to 경로관련, null= 비어있어도 괜찮은지 True, False
```python
# gsweb\profileapp\models.py
class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE,
                                related_name='profile')
    image = models.ImageField(upload_to='profile/', null=True)
    nickname = models.CharField(max_length=30, unique=True, null=True)
    message = models.CharField(max_length=200, null=True)




python manage.py makemigrations
# Migrations for 'profileapp':
#  profileapp\migrations\0001_initial.py
#   - Create model Profile

python manage.py migrate
# Operations to perform:
#   Apply all migrations: accountapp, admin, auth, contenttypes, profileapp, sessions
# Running migrations:
#   Applying profileapp.0001_initial... OK
```
#### migration command
* makimigrations
  - model변경을 감지하고 변경사항을 반영할 파일 생성
* migrate
* django에서 database schema에 데이터베이스 생성, 변경?

```python
# gsweb\profileapp\forms.py
class ProfileCreationForm(ModelForm):
    class Meta:
        model = Profile
        fields = ['image', 'nickname', 'message']   # user이 없는 이유는 user은 서버에서 직접 처리


# gsweb\profileapp\views.py
class ProfileCreateView(CreateView):
    model = Profile
    form_class = ProfileCreationForm
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'profileapp/create.html'
```

- profileapp에 templates디렉토리를 만들고 그 안에 profileapp 디렉토리를 만든 후  create.html을 생성해준다.
- gsweb\profileapp\templates\profileapp\create.html
```html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Create Profile </h4>
        </div>
        <div>
            <form action="{% url 'profileapp:create' %}" 
                  enctype="multipart/form-data" method="post">      <!-- enctype -->
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```
- form tag의 enctype 속성 : <form> 태그의 enctype 속성은 폼 데이터(form data)가 서버로 제출될 때 해당 데이터가 인코딩되는 방법을 명시합니다. 
  이 속성은 <form> 요소의 method 속성값이 “post”인 경우에만 사용할 수 있습니다.
  
| 속성값 | 설명 |
| --- | --- |
|application/x-www-form-urlencoded|기본값으로, 모든 문자들은 서버로 보내기 전에 인코딩됨을 명시함.|
|multipart/form-data| 모든 문자를 인코딩하지 않음을 명시함. 이 방식은 <form> 요소가 파일이나 이미지를 서버로 전송할 때 주로 사용함.|
|text/plain|공백 문자(space)는 "+" 기호로 변환하지만, 나머지 문자는 모두 인코딩되지 않음을 명시함.|

<hr>

## <2>
```python
# profileapp/views.py
class ProfileCreateView(CreateView):
    model = Profile
    form_class = ProfileCreationForm
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'profileapp/create.html'

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)
```
```html
# accountapp/templates/accountapp/detail.html
        <div>
            {% if target_user.profile %}
                <img src="{{ target_user.profile.image.url }}"
                     class="profile_image m-4"
                     alt="profile image">
            <h2 class="NNS_B">
                {{ target_user.profile.nickname }}
                <a href="{% url 'profileapp:update' pk=target_user.profile.pk %}">
                    edit
                </a>
            </h2>
                <h5 class="m-5">{{ target_user.profile.message }}</h5>
            {% else %}
            <h2>
                <a href="{% url 'profileapp:create' %}">
                    Create Profile
                </a>
            </h2>
            {% endif %}
        </div>
```
데이터베이스안엔 이미지의 경로만 들어간다.
그래서 url경로를 추가해 준다.?
gsweb의 media폴더 안에 gsweb의 urls.py에 image를 로드할 수 있도록 경로 만들어주기, 잘사용하진 않는다.

```python
# gsweb/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/',include('accountapp.urls')),
    path('profiles/', include('profileapp.urls'))
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```
```css
# static/base.css
.profile_image{
    height: 12rem;
    width: 12rem;
    border-radius: 20rem;
    object-fit: cover;
} 
.NNS_B{
    font-family: "NanumSquareB", cursive;
}
```
```html
# 디테일 창에 profileupdate버튼 edit만들기  
# accountapp/templates/accountapp/detail.html
            <h2 class="NNS_B">
                {{ target_user.profile.nickname }}
                <a href="{% url 'profileapp:update' pk=target_user.profile.pk %}">
                    edit
                </a>
            </h2>
```
Profileupdateview  
image css  
image link  
image제출하고 데이터베이스확인, media폴더가 자동 생성됨  

enctype="multipart/form-data"
```html
# profileapp/templates/profileapp/update.html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Update Profile </h4>
        </div>
        <div>
            <form action="{% url 'profileapp:update' pk=target_profile.pk %}"
                  enctype="multipart/form-data" method="post">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %} 
```
```python
# profileapp/urls.py
app_name = 'profileapp'

urlpatterns = [
    path('create/', ProfileCreateView.as_view(), name='create')
    path('create/', ProfileCreateView.as_view(), name='create'),
    path('update/<int:pk>', ProfileUpdateView.as_view(), name='update'),
]
```
```python
# profileapp/views.py
class ProfileUpdateView(UpdateView):
    model = Profile
    context_object_name = 'target_profile'
    form_class = ProfileCreationForm
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'profileapp/update.html' 
```

<hr>

## <3>
detail창의 edit버튼을 로그인 당사자가 아닌 다른 사람이 들어가지 못하도록 if문 사용
```html
                    {% if target_user == user%}
                    <a href="{% url 'profileapp:update' pk=target_user.profile.pk %}">
                        edit
                    </a>
                    {% endif %}
```
프로필업데이트뷰에서 successurl을 디테일 창으로 넘어가게 하려면 해당 유저의 디테일 창으로 가게 해야 하고, 이를 위해서는 동적url이기 때문에 따로 함수를 만들어준다.  
- 프로필크리에이트뷰도 넣어준다.

```python
    def get_success_url(self):
        return reverse('accountapp:detail', kwargs={'pk':self.object.user.pk})   # self.object는 target.profile과 같다.
        # kwargs 안에는 dict가 들어감

# accountapp의 view.py
class AccountUpdateView(UpdateView):
    model = User
    form_class = AccountCreationForm
    context_object_name = 'target_user'
    success_url = reverse_lazy('accountapp:hello_world')
    template_name = 'accountapp/update.html'

    def get_success_url(self):
        return reverse('accountapp:detail',kwargs={'pk':self.object.pk})    # self.object는 target_user이다!
```
reverse는 함수형 view에서 사용, reverselazy 구분
로그아웃하고 디테일창에 들어갈 수 없도록 하기 위해 decorator 사용해주기  
메소드 변환해주는게 메소드 데코레이터
변환할 데코레이터를 넣어준다.

```python
# profileapp / decorators.py
def profile_ownership_required(func):
    def decorated(request, *args, **kwargs):
        target_profile = Profile.objects.get(pk=kwargs['pk'])
        if target_profile.user == request.user:
            return func(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()

# profileapp/views.py
@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class ProfileCreateView(CreateView):


@method_decorator(profile_ownership_required,'get')
@method_decorator(profile_ownership_required,'post')
class ProfileUpdateView(UpdateView):

```
https://fonts.google.com/icons  
아이콘이 필요할 땐 구글!
```html
# gsweb/templates/head.html   
   
    {# MATERIAL ICONS lINK #}
    <link href="https://fonts.googleapis.com/css2?family=Material+Icons"
      rel="stylesheet">
     

# gsweb\accountapp\templates\accountapp\detail.html
                    <a href="{% url 'profileapp:update' pk=target_user.profile.pk %}" 
                    class="material-icons round_button">      # material-icons 는 구글아이콘 사용하는 방법
                        edit                # round-button은 디자인을 위한 class지정
                    </a>
```
```css
#  gsweb/static/base.css
.round_button{
    color: cornflowerblue;
    text-decoration: none;
    box-shadow: 0 0 3px darkslateblue;
    border-radius: 10rem;
    padding: .3rem;
}
.round_button:hover{        # hover는 마우스를 올려놨을 때
    color: red;
}
```
```html
# gsweb\accountapp\templates\accountapp\detail.html
        <div>
        <a href="{% url 'accountapp:update'  pk=target_user.pk %}"
        class="material-icons round_button mx-2">
            settings    # google icon에서 가져오려면 이름을 구글 아이콘에 있는 걸로 바꿔주면 바뀐다!
        </a>
        
        <a href="{% url 'accountapp:delete' pk=target_user.pk %}"
        class="material-icons round_button mx-2">
            close
        </a>
    </div>

# div층을 합쳐준다. 한라인에 보이도록,
# class에 mx-2는 부트스트랩의 마진으로 x축으로 2를 의미
```
magic grid https://jsfiddle.net/eolaojo/4pov0rdf/

```html
app_name = 'articleapp'

urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),

]
# TemplateView

# gsweb\articleapp\templates\articleapp\list.html
{% extends 'base.html' %}

{% block content %}

    <style>     # magic grid의 css파일을 넣어주기! style사용
        .container div {
          width: 280px;
          height: 500px;
          background-color: antiquewhite;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 8px;
        }

    .container .item1 { height: 200px; }
    .container .item4 { height: 800px; }
    .container .item6 { height: 600px; }
    .container .item11 { height: 400px; }
    </style>


    <!DOCTYPE html>       # 매직그리드에서 가져오기
<div class="container">
  <div class="item1">1</div>
  <div class="item2">2</div>
  <div class="item3">3</div>
  <div class="item4">4</div>
  <div class="item5">5</div>
  <div class="item6">6</div>
  <div class="item7">7</div>
  <div class="item8">8</div>
  <div class="item9">9</div>
  <div class="item10">10</div>
  <div class="item11">11</div>
  <div class="item12">12</div>
  <div class="item13">13</div>
</div>

{% endblock %}
```

<hr>

# 7주차
## <1>
static구문을 사용하기 위해서는 load static구문을 사용해주어야 한다.
```js
# static안에 js디렉토리생성, magicgrid.js 생성
# https://github.com/e-oj/Magic-Grid 에서 Magic-Grid/dist/magic-grid.cjs.js 파일내용을 copy해서 넣어준 후
module.exports = MagicGrid;  # 마지막에 있는 모듈은 장고에서는 쓰지 않기 때문에 지워준다. 다른 웹프레임워크에서 사용해주는 것.,

# https://jsfiddle.net/eolaojo/4pov0rdf/의 JavaScript + No-Library (pure JS)코드를 copy해서 밑에 추가해준다.
let magicGrid = new MagicGrid({
  container: '.container',
  animate: true,
  gutter: 30,
  static: true,
  useMin: true
});

magicGrid.listen();
```
```html
# gsweb\articleapp\templates\articleapp\list.html
{% extends 'base.html' %}
{% load static %}
{% block content %}

    <script src="{% static 'js/magicgrid.js' %}">
    </script>
```
lorem picsum : 로렘입섬을 오마주한 웹서비스 임의의 이미지를 갖다줌
```html
# gsweb\articleapp\templates\articleapp\list.html
{% extends 'base.html' %}
{% load static %}
{% block content %}

    <style>
        .container div {
          width: 280px;
          background-color: lightseagreen;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 1rem;
        }
    .container img{
        width: 100%;
        border-radius: 1rem;
    }
    </style>


    <!DOCTYPE html>
<div class="container my-4">
  <div class="item1"><img src="https://picsum.photos/200/300" alt=""></div>   # 로렘픽섬의 주소를 불러온다. 맨 마지막 숫자는 높이
  <div class="item2"><img src="https://picsum.photos/200/333" alt=""></div>
  <div class="item3"><img src="https://picsum.photos/200/500" alt=""></div>
  <div class="item4"><img src="https://picsum.photos/200/100" alt=""></div>
  <div class="item5"><img src="https://picsum.photos/200/300" alt=""></div>
  <div class="item6"><img src="https://picsum.photos/200/200" alt=""></div>
  <div class="item7"><img src="https://picsum.photos/200/300" alt=""></div>
  <div class="item8"><img src="https://picsum.photos/200/400" alt=""></div>
  <div class="item9"><img src="https://picsum.photos/200/300" alt=""></div>
  <div class="item10"><img src="https://picsum.photos/200/700" alt=""></div>
  <div class="item11"><img src="https://picsum.photos/200/200" alt=""></div>
  <div class="item12"><img src="https://picsum.photos/200/600" alt=""></div>
  <div class="item13"><img src="https://picsum.photos/200/200" alt=""></div>
</div>

    <script src="{% static 'js/magicgrid.js' %}">
    </script>
{% endblock %}
```
```js
# gsweb2\static\js\magicgrid.js
var masonrys = document.getElementsByTagName("img") # document = list.html
for (let i=0; i < masonrys.length; i++){
  masonrys[i].addEventListener('load', function (){
    magicGrid.positionItems();
  }, false)
}
# EventListener 이벤트가 일어나는 걸 감시하는, 어떤 이벤트가 일어났을 때 어떤 function을 실행하겠다라고 적음 (magicGrid.positionitems)
# => 로드 이벤트가 일어날 때 위치를 재조정한다!
```
```python
# gsweb\articleapp\models.py
class Article(models.Model):
    writer = models.ForeignKey(User, on_delete=models.SET_NULL,
                               related_name='article',
                               null=True)
    title = models.CharField(max_length=200, null=True)
    image = models.ImageField(upload_to='article/', null=True)
    content = models.TextField(null=True)

    created_at = models.DateField(auto_now_add=True, null=True)
```
model을 만들었기 때문에  
python manage.py makemigrations
python manage.py migrate

```python
# articleapp/forms.py
from django.forms import ModelForm

from articleapp.models import Article


class ArticleCreationForm(ModelForm):
    class Meta:
        model = Article
        fields = ['title', 'image', 'content']
```

<hr>

## <2>
```python
# gsweb\articleapp\views.py
from django.urls import reverse_lazy
from django.views.generic import CreateView

from articleapp.forms import ArticleCreationForm
from articleapp.models import Article


class ArticleCreateView(CreateView):
    model = Article
    form_class = ArticleCreationForm
    success_url = reverse_lazy('articleapp:list')
    template_name = 'articleapp/create.html'

# gsweb\articleapp\urls.py
app_name = 'articleapp'

urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('create/', ArticleCreateView.as_view(), name='create'),
]

```
```html
# gsweb\articleapp\templates\atricleapp\create.html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Create Article </h4>
        </div>
        <div>
            <form action="{% url 'articleapp:create' %}"
                  method="post" enctype="multipart/form-data">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit"
                           class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```
```python
# gsweb\articleapp\views.py
class ArticleDetailView(DetailView):
    model = Article
    context_object_name = 'target_article'
    template_name = 'articleapp/detail.html'

# gsweb\articleapp\urls.py

urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('create/', ArticleCreateView.as_view(), name='create'),
    path('detail/<int:pk>', ArticleDetailView.as_view(), name='detail'),
]

```
```html
# gsweb\articleapp\templates\articleapp\detail.html
{% extends 'base.html' %}

{% block content %}

    <div class="container">   {# bootstrap의 class#}
        <div>

            {#      제목, 글쓴이, 작성일    #}
            <h1>{{ target_article.title }}</h1>
            <h3>{{ target_article.writer.profile.nickname }}</h3>
            <p>{{ target_article.created_at }}</p>
        </div>
        <div>
            {#      게시글 대표이미지, 글 내용     #}
            <img src="{{ target_article.image.url }}" alt="">
        </div>
    </div>


{% endblock %}
```
```python
# gsweb\articleapp\views.py
class ArticleUpdateView(UpdateView):
    model = Article
    form_class = ArticleCreationForm
    context_object_name = 'target_article'
    template_name = 'articleapp/update.html'

    def get_success_url(self):
        return reverse('articleapp:detail',kwargs={'pk':self.object.pk})

# gsweb\articleapp\urls.py
urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('create/', ArticleCreateView.as_view(), name='create'),
    path('detail/<int:pk>', ArticleDetailView.as_view(), name='detail'),
    path('update/<int:pk>', ArticleUpdateView.as_view(), name='update'),
]
```
```html
# gsweb\articleapp\templates\articleapp\detail.html
            <div>
                <a href="{% url 'articleapp:update' pk=target_article.pk %}"
                class="btn btn-success rounded-pill px-5">
                    Update
                </a>
            </div>
```
```html
# gsweb\articleapp\templates\articleapp\update.html
{% extends 'base.html' %}
{% load bootstrap4 %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Update Article </h4>
        </div>
        <div>
            <form action="{% url 'articleapp:update' pk=target_article.pk %}"
                  method="post" enctype="multipart/form-data">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit"
                           class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```
```python
# gsweb\articleapp\views.py
class ArticleDeleteView(DeleteView):
    model = Article
    context_object_name = 'target_article'
    success_url = reverse_lazy('articleapp:list')
    template_name = 'articleapp/delete.html'

# gsweb\articleapp\urls.py
urlpatterns = [
    path('list/', TemplateView.as_view(template_name='articleapp/list.html'), name='list'),
    path('create/', ArticleCreateView.as_view(), name='create'),
    path('detail/<int:pk>', ArticleDetailView.as_view(), name='detail'),
    path('update/<int:pk>', ArticleUpdateView.as_view(), name='update'),
    path('delete/<int:pk>', ArticleDeleteView.as_view(), name='delete'),
]

```
```html
# gsweb\articleapp\templates\articleapp\detail.html

                <a href="{% url 'articleapp:delete' pk=target_article.pk %}"
                class="btn btn-danger rounded-pill px-5">
                    Delete
                </a>

# gsweb\articleapp\templates\articleapp\delete.html
{% extends 'base.html' %}

{% block content %}

    <div class="text-center mw-500 m-auto">
        <div>
            <h4 class="m-5"> Delete Article : {{ target_article.title }} </h4>
        </div>
        <div>
            <form action="{% url 'articleapp:delete' pk=target_article.pk %}"
                  method="post">
                {% csrf_token %}
                <div class="m-5">
                    <input type="submit"
                           class="btn btn-danger rounded-pill px-5"> 
                </div>
            </form>
        </div>
    </div>

{% endblock %}
```

<hr>

## <3>
로그인이 되어 있찌 않아도 article/detail페이지에 들어갈 수 있기 때문에 decorator을 다시 사용하여 막아준다.  
create에서 바로 detail창으로 들어가도록 설정  
decorator를 사용
```python
# gsweb\articleapp\views.py
@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class ArticleCreateView(CreateView):
    model = Article
    form_class = ArticleCreationForm
    template_name = 'articleapp/create.html'
    def form_valid(self, form):
        form.instance.writer = self.request.user
        return super().form_valid(form)
    def get_success_url(self):
        return reverse('articleapp:detail',kwargs={'pk':self.object.pk})

# gsweb\articleapp\decorators.py
from django.http import HttpResponseForbidden

from articleapp.models import Article


def article_ownership_required(func):
    def decorated(request, *args, **kwargs):
        target_article = Article.objects.get(pk=kwargs['pk'])
        if target_article.writer == request.user:
            return func(request, *args, **kwargs)
        else:
            return HttpResponseForbidden()
    return decorated


# gsweb\articleapp\views.py
@method_decorator(article_ownership_required,'get')
@method_decorator(article_ownership_required,'post')
class ArticleUpdateView(UpdateView):
    model = Article
    form_class = ArticleCreationForm
    context_object_name = 'target_article'
    template_name = 'articleapp/update.html'

    def get_success_url(self):
        return reverse('articleapp:detail',kwargs={'pk':self.object.pk})

@method_decorator(article_ownership_required,'get')
@method_decorator(article_ownership_required,'post')
class ArticleDeleteView(DeleteView):
    model = Article
    context_object_name = 'target_article'
    success_url = reverse_lazy('articleapp:list')
    template_name = 'articleapp/delete.html'
```

```html
# gsweb\articleapp\templates\articleapp\detail.html
{% extends 'base.html' %}

{% block content %}

    <div class="container text-center">
        <div class="my-5">

            {#      제목, 글쓴이, 작성일    #}
            <h1 class="NNS_B">{{ target_article.title }}</h1>
            <h3>{{ target_article.writer.profile.nickname }}</h3>
            <p>{{ target_article.created_at }}</p>
        </div>
        <hr>
        <div class="my-5">
            {#      게시글 대표이미지, 글 내용     #}
            <img src="{{ target_article.image.url }}"
                 class="article_image"
                 alt="">
            <div class="article_content">
                {{ target_article.content }}
            </div>
            {% if target_article.writer == user %}
            <div>
                <a href="{% url 'articleapp:update' pk=target_article.pk %}"
                class="btn btn-success rounded-pill px-5">
                    Update
                </a>
                <a href="{% url 'articleapp:delete' pk=target_article.pk %}"
                class="btn btn-danger rounded-pill px-5">
                    Delete
                </a>
            </div>
            {% endif %}
        </div>
    </div>




{% endblock %}
```

```css
# gsweb\static\base.css

.article_image{
    border-radius: 2rem;
    width: 70%;
    box-shadow: 0 0 .5rem grey;
}
.article_content{
    font-size: 1.1rem;
    text-align: left;
    margin: 2rem;
}
```
### page ination  
generate page of objects  
page 짜르기..  
```python
# gsweb\articleapp\views.py
class ArticleListView(ListView):
    model = Article
    context_object_name = 'article_list'
    template_name = 'articleapp/list.html'
    paginate_by = 20

# gsweb\articleapp\urls.py
urlpatterns = [
    path('list/', ArticleListView.as_view(), name='list'),
    path('create/', ArticleCreateView.as_view(), name='create'),
    path('detail/<int:pk>', ArticleDetailView.as_view(), name='detail'),
    path('update/<int:pk>', ArticleUpdateView.as_view(), name='update'),
    path('delete/<int:pk>', ArticleDeleteView.as_view(), name='delete'),
]
```

```html
# gsweb\articleapp\templates\articleapp\list.html
{% extends 'base.html' %}
{% load static %}
{% block content %}

    <style>
        .container div {
          width: 280px;
          background-color: lightseagreen;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 1rem;
            flex-direction: column;
        }
    .container img{
        width: 100%;
        border-radius: 1rem;
    }
    </style>


    <!DOCTYPE html>
    <div class="container my-4">

        {% for article in article_list %}
            <div>
                <a href="{% url 'articleapp:detail' pk=article.pk %}">
                    <img src="{{ article.image.url }}"
                         alt="">
                </a>
            <span>{{ article.title }}</span>
            </div>
        {% endfor %}

    </div>

    <script src="{% static 'js/magicgrid.js' %}"></script>
    <div class="text-center my-5">
        <a href="{% url 'articleapp:create' %}"
        class="btn btn-dark rounded-pill material-icons">
            brush
        </a>
        <a href="{% url 'articleapp:create' %}"
        class="btn btn-outline-dark rounded-pill px-5">
            Create Article
        </a>
    </div>
{% endblock %}
```

<hr>

# 8주차
## <1>
add pagination
```html
# articleapp/templates/articleapp/list.html
   {% include 'snippets/pagination.html' %}

# templates/snippets/pagination.html
    <div class="text-center my-4">
        {% if page_obj.has_previous %}
            <a href="?page={{ page_obj.previous_page_number}}"
            class="btn btn-secondary rounded-pill">
                {{ page_obj.previous_page_number }}
            </a>
        {% endif %}
        <a href="#"
        class="btn btn-dark rounded-pill">
            {{ page_obj.number }}
        </a>
        {% if page_obj.has_next %}
            <a href="?page={{ page_obj.next_page_number }}"
            class="btn btn-secondary rounded-pill">
                {{page_obj.next_page_number}}
            </a>
        {% endif %}
    </div>
```
터미널에  
python manage.py startapp commentapp
```python
# commentapp/urls.py 파일 생성
urlpatterns = [
] 

# gsweb/urls.py
    path('comments/',include('commentapp.urls')),


# commentapp/models.py
class Comment(models.Model):
    article = models.ForeignKey(Article, on_delete=models.SET_NULL,
                                related_name='comment', null=True)
    writer = models.ForeignKey(User, on_delete=models.SET_NULL,
                               related_name='comment', null=True)

    content = models.TextField(null=False)

    created_at = models.DateTimeField(auto_now_add=True)

# commentapp/forms.py 파일 생성
class CommentCreationForm(ModelForm):
    class Meta:
        model = Comment
        fields = ['content'] 
```
터미널에  
python manage.py makemigrations  
python manage.py migrate

###  add comment createview
```python
# commentapp/views.py
class CommentCreateView(CreateView):
    model = Comment
    form_class = CommentCreationForm
    template_name = 'commentapp/create.html'

    def get_success_url(self):
        return reverse('articleapp:detail', kwargs={'pk':self.object.article.pk}) 

# commentapp/urls.py
app_name = 'commentapp'

urlpatterns = [
    path('create/', CommentCreateView.as_view(), name='create'),
] 
```

```html
# commentapp/templates/commentapp/create.html
{% load bootstrap4 %}
    <div class="text-center mw-500 m-auto">
        <div class="m-5">
            <h4>Create Comment</h4>
        </div>
        <div>
            <form action="{% url 'commentapp:create' %}" method="post">
                {% csrf_token %}
                {% bootstrap_form form %}
                <div class="m-5">
                    <input type="submit" class="btn btn-dark rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

```

```python
# articleapp/views.py
class ArticleDetailView(DetailView, FormMixin):
    model = Article
    form_class = CommentCreationForm
    context_object_name = 'target_article'
    template_name = 'articleapp/detail.html'
```

```html
#  articleapp/templates/articleapp/detail.html
        <hr>

        <div class="text-center my-4">
            {% include 'commentapp/create.html'%}
        </div>
  
```

<hr>

## <2>
 commentcreateview form_valid
 ```python
 # commentapp/views.py
     def form_valid(self, form):
        form.instance.writer = self.request.user
        form.instance.article_id = self.request.POST.get('article_pk')
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('articleapp:detail', kwargs={'pk':self.object.article.pk}) 

```

```html
# commentapp/templates/commentapp/create.html
                <input type="hidden"
                    name="article_pk"
                    value="{{ target_article.pk }}">
```
add CommentDeleteView
```python
# commentapp/views.py
class CommentDeleteView(DeleteView):
    model = Comment
    context_object_name = 'target_comment'
    template_name = 'commentapp/delete.html'

    def get_success_url(self):
        return reverse('articleapp:detail', kwargs={'pk':self.object.article.pk}
                       
# commentapp/urls.py
    path('delete/<int:pk>', CommentDeleteView.as_view(), name='delete'),

```

```html
# commentapp/templates/commentapp/delete.html
{% extends 'base.html' %}


{% block content %}

    <div class="text-center mw-500 m-auto">
        <div class="m-5">
            <h4>Delete Comment </h4>
        </div>
        <div>
            <form action="{% url 'commentapp:delete' pk=target_comment.pk %}" method="post">
                {% csrf_token %}
                <div class="m-5">
                    <input type="submit" class="btn btn-danger rounded-pill px-5">
                </div>
            </form>
        </div>
    </div>

{% endblock %} 
```

comment-box
```html
# articleapp/templates/articleapp/detail.html
            {% for comment in target_article.comment.all %}
                <div class="comment-box">
                    <div>
                        <span class="NNS_B" style="font-size: 1.3rem">{{ comment.writer.profile.nickname }}</span>
                        <span>{{ comment.created_at }}</span>
                    </div>
                    <div>
                        <p>
                            {{ comment.content }}
                        </p>
                    </div>
                </div>

            {% endfor %}
```

```css
# static/base.css
.comment-box{
    text-align: left;
    border: solid;
    border-color: #3c3cd5;
    border-radius: 1rem;
    padding: 1rem;
    margin: 1rem;
}
```

comment decorators
```python
# commentapp/views.py
@method_decorator(login_required,'get')
@method_decorator(login_required,'post')

@method_decorator(comment_ownership_required, 'get')
@method_decorator(comment_ownership_required, 'post')

# commentapp/decorators.py
from django.http import HttpResponseForbidden

from commentapp.models import Comment


def comment_ownership_required(func):
    def decorated(request, *args, **kwargs):
        target_comment = Comment.objects.get(pk=kwargs['pk'])
        if target_comment.writer == request.user:
            return func(request,*args, **kwargs)
        else:
            return HttpResponseForbidden()
    return decorated 
```

 comment delete button
 ```html
# articleapp/templates/articleapp/detail.html
                    {% if comment.writer == user %}
                    <div style="text-align: right">
                        <a href="{% url 'commentapp:delete' pk=comment.pk %}"
                        class="btn btn-danger rounded-pill px-5">
                            Delete
                        </a>
                    </div>
                    {% endif %}
```

<hr>

## <3>
ngrok 다운, ngrok.exe파일을 최상위 폴더로 옮겨준다.  
terminal창에 ngrok http 8000
```python
# gsweb/settings.py
ALLOWED_HOSTS = ["*"]
```
```html
# gsweb\templates\head.html
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    # articleapp/templates/articleapp/list.html
    <style>
        .container{
            padding: 0;
            margin: 0 auto;
        }
        .container div {
          width: 45%;
          background-color: lightseagreen;
          max-width: 250px;
            box-shadow: 0 0 .5rem cadetblue;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 1rem;
            flex-direction: column;
        }
```

```js
# static/js/magicgrid.js
let magicGrid = new MagicGrid({
  container: '.container',
  animate: true,
  gutter: 12,
  static: true,
  useMin: true
});

```
```css
# static/base.css
@media screen and(max-width: 500px){
    html{
        /* default font-size = 16px */
        font-size: 13px;
    }
}
```
terminal  
python manage.py startapp projectapp  

```python
# gsweb/settings.py
    'projectapp',

# gsweb/urls.py
    path('projects/', include('projectapp.urls')),
```

<hr>
