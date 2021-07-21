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
