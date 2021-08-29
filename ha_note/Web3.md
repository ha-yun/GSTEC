# 9주차
## <1>
```python
# projectapp/models.py

class Project(models.Model):
    name = models.CharField(max_length=20, null=False)
    description = models.CharField(max_length=200, null=True)
    image = models.ImageField(upload_to='project/', null=False)

    created_at = models.DateTimeField(auto_now_add=True)

```

terminal에  
python manage.py makemigrations  
python manage.py migrate
```python
# projectapp/forms.py
from django.forms import ModelForm

from projectapp.models import Project


class ProjectCreationForm(ModelForm):
    class Meta:
        model = Project
        fields = ['name', 'image', 'description']


# projectapp/views.py
class ProjectCreateView(CreateView):
    model = Project
    form_class = ProjectCreationForm
    success_url = reverse_lazy('articleapp:list')
    template_name = 'projectapp/create.html'

# projectapp/urls.py
from projectapp.views import ProjectCreateView

app_name = 'projectapp'

urlpatterns = [
    path('create/', ProjectCreateView.as_view(), name='create'),
]
```

```html
# projectapp/templates/projectapp/create.html
{% extends 'base.html' %}
{% load bootstrap4 %}

{% block content %}

    <div class="text-center mw-500 m-auto">
        <div class="m-5">
            <h4>Create Project</h4>
        </div>
        <div>
            <form action="{% url 'projectapp:create' %}"
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
# projectapp/views.py

class ProjectDetailView(DetailView):
    model = Project
    context_object_name = 'target_project'
    template_name = 'projectapp/detail.html'


# projectapp/urls.py
    path('detail/<int:pk>', ProjectDetailView.as_view(), name='detail'),
```

```html
# projectapp/templates/projectapp/detail.html
{% extends 'base.html' %}

{% block content %}

    <div class="text-center mw-500 m-auto">
        <div class="m-5">
            <img src="{{ target_project.image.url }}"
                 class="profile_image m-4"
                 alt="profile image">
            <h2 class="NNS_B">
                {{ target_project.name }}
            </h2>
            <h5>{{ target_project.description }}</h5>
        </div>
    </div>


{% endblock %} 

```

```python
# projectapp/views.py
@method_decorator(login_required, 'get')
@method_decorator(login_required, 'post')
class ProjectCreateView(CreateView):
    model = Project
    form_class = ProjectCreationForm
    template_name = 'projectapp/create.html'

    def get_success_url(self):
        return reverse('projectapp:detail', kwargs={'pk': self.object.pk})


# projectapp/views.py
class ProjectListView(ListView):
    model = Project
    context_object_name = 'project_list'
    template_name = 'projectapp/list.html'
    paginate_by = 20

# projectapp/urls.py
    path('list/', ProjectListView.as_view(), name='list'),


```

```html
# projectapp/templates/projectapp/list.html
{% extends 'base.html' %}
{% load static %}

{% block content %}

    <style>
        .container {
            padding: 0;
            margin: 0 auto;
        }

        .container div {
          width: 45%;
            max-width: 200px;
            box-shadow: 0 0 .5rem grey;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 1rem;
            flex-direction: column;
        }

        .container img {
            width: 100%;
            border-radius: 1rem;
        }

    </style>


    <div class="container my-4">

        {% for project in project_list %}

            <div>
                <a href="{% url 'projectapp:detail' pk=project.pk %}">
                    <img src="{{ project.image.url }}"
                         alt="">
                </a>
                <span class="m-2 NNS_B">{{ project.name }}</span>
            </div>

        {% endfor %}

    </div>

    <script src="{% static 'js/magicgrid.js' %}"></script>

    {% include 'snippets/pagination.html' %}

    <div class="text-center my-5">
        <a href="{% url 'projectapp:create' %}"
           class="btn btn-outline-dark rounded-pill px-5">
            Create Project
        </a>
    </div>


{% endblock %}
```

```html
# templates/header.html
        <span>
            <a href="{% url 'articleapp:list' %}">
                Articles
            </a>
        </span>
        <span>
            <a href="{% url 'projectapp:list' %}">
                Projects
            </a>
        </span>


# projectapp/templates/projectapp/list.html
        .container div {
          background-color: lightseagreen;
            box-shadow: 0 0 .5rem cadetblue;
          display: flex;
          justify-content: center;
          align-items: center;
          border-radius: 1rem;
            flex-direction: column;
        }
    .container img{
        width: 7rem;
        height: 7rem;
        object-fit: cover;
        border-radius: 1rem;
    }
```

<hr>

## <2>
```html
# projectapp/templates/projectapp/list.html
                <span class="m-2 NNS_B">{{ project.name | truncatechars:10 }}</span>

# In View Using Mixin
# projectapp/templates/projectapp/detail.html
    <div>
        {% include 'snippets/list_fragment.html' with article_list=object_list %}
    </div>

```

```python
# projectapp/views.py
class ProjectDetailView(DetailView, MultipleObjectMixin):
    model = Project
    context_object_name = 'target_project'
    template_name = 'projectapp/detail.html'

    paginate_by = 20

    def get_context_data(self, **kwargs):
        article_list = Article.objects.filter(project=self.object)
        return super().get_context_data(object_list=article_list, **kwargs)


```

```html
# templates/snippets/list_fragment.html
{% load static %}

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

    {% include 'snippets/pagination.html' %}

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


# accountapp/templates/accountapp/detail.html
    <div>
        {% include 'snippets/list_fragment.html' with article_list=object_list %}
    </div>
```

```python
# accountapp/views.py
class AccountDetailView(DetailView, MultipleObjectMixin):
    model = User
    context_object_name = 'target_user'
    template_name = 'accountapp/detail.html'

    paginate_by = 20

    def get_context_data(self, **kwargs):
        article_list = Article.objects.filter(writer=self.object)
        return super().get_context_data(object_list=article_list, **kwargs)

```

<hr>

# 10주차
## <1>
subscribe의 model만들기  
Meta클래스는 모델폼을 사용하면서 사용..?  
Meta클래스는 외부적인 옵션을 지정해주는 unique_together  

model을 변경하였으니 터미널에서  
python manage.py makemigrations  
python manage.py migrate
```python
# subscribeapp/models.py
class Subscription(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='subscription', null=False)
    project = models.ForeignKey(Project, on_delete=models.CASCADE,
                                related_name='subscription', null=False)

    class Meta:
        unique_together = ['user','project']
```
subscription view
```python
# subscribeapp/views.py
@method_decorator(login_required, 'get')
class SubscriptionView(RedirectView):

    def get(self, request, *args, **kwargs):
        user = request.user
        project = Project.objects.get(pk=kwargs['project_pk'])
        subscription = Subscription.objects.filter(user=user,
                                                   project=project)
        if subscription.exists():
            subscription.delete()
        else:
            Subscription(user=user, project=project).save()

        return super().get(request, *args, **kwargs)

    def get_redirect_url(self, *args, **kwargs):
        return reverse('projectapp:detail', kwargs={'pk' : kwargs['project_pk']}) 


# subscribeapp/urls.py

app_name = 'subscribeapp'

urlpatterns=[
    path('subscribe/<int:project_pk>', SubscriptionView.as_view(), name='subscribe'),
] 
```
```html
# projectapp/templates/projectapp/detail.html
    <div class="text-center">
        <a href="{% url 'subscribeapp:subscribe' project_pk=target_project.pk %}"
            class="btn btn-primary rounded-pill px-5">
            Subscribe
        </a>
    </div>
```
```python
# projectapp/views.py
class ProjectDetailView(DetailView, MultipleObjectMixin):
    model = Project
    context_object_name = 'target_project'
    template_name = 'projectapp/detail.html'

    paginate_by = 20

    def get_context_data(self, **kwargs):
        user = self.request.user
        project = self.object

        if user.is_authenticated:
            subscription = Subscription.objects.filter(user=user,
                                                       project=project)
        else:
            subscription = None

        article_list = Article.objects.filter(project=self.object)
        return super().get_context_data(object_list=article_list,
                                        subscription=subscription,
                                        **kwargs)
```
```html
# projectapp/templates/projectapp/detail.html
    {% if user.is_authenticated %}
    <div class="text-center">
        {% if not subscription %}
        <a href="{% url 'subscribeapp:subscribe' project_pk=target_project.pk %}"
           class="btn btn-primary rounded-pill px-5">
            Subscribe
        </a>
        {% else %}
        <a href="{% url 'subscribeapp:subscribe' project_pk=target_project.pk %}"
           class="btn btn-secondary rounded-pill px-5">
            Unsubscribe
        </a>
        {% endif %}
    </div>
    {% endif %}
```
field lookup : project__in
```python
# subscribeapp/views.py

class SubscriptionListView(ListView):
    model = Article
    context_object_name = 'article_list'
    template_name = 'subscribeapp/list.html'

    paginate_by = 20

    def get_queryset(self):
        project_list = Subscription.objects.filter(user=self.request.user).values_list('project')
        article_list = Article.objects.filter(project__in=project_list)
        return article_list


# subscribeapp/urls.py
path('list/', SubscriptionListView.as_view(), name='list'),
```
```html
# subscribeapp/templates/subscribeapp/list.html
{% extends 'base.html' %}

{% block content %}

    <div>
        {% include 'snippets/list_fragment.html' %}
    </div>

{% endblock %} 


# templates/header.html
       <span>
            <a href="{% url 'subscribeapp:list' %}">
                Subscription
            </a>
        </span>
```

<hr>

## <2>
1. 장고 message  
2. DB transaction
terminal에   
python manage.py startapp likeapp
   
```python
# gsweb/settings.py
'likeapp',

# gsweb/urls.py
    path('like/', include('likeapp.urls')),

# subscribeapp/views.py
@method_decorator(login_required, 'get')
class SubscriptionListView(ListView):
```
```html
# templates/header.html

        {% if user.is_authenticated %}
        <span>
            <a href="{% url 'subscribeapp:list'%}">
                Subscription
            </a>
        </span>
        {% endif %}
```
```python
# likeapp/models.py
class LikeRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='like_record', null=False)
    article = models.ForeignKey(Article, on_delete=models.CASCADE,
                                related_name='like_record', null=False)

    class Meta:
        unique_together = ['user', 'article']


# articleapp/models.py
    like = models.IntegerField(default=0) 
```
python manage.py makemigrations  
python manage.py migrate
```python
# likeapp/views.py
@method_decorator(login_required, 'get')
class LikeArticleView(RedirectView):

    def get(self, request, *args, **kwargs):
        user = request.user
        article = Article.objects.get(pk=kwargs['article_pk'])

        like_record = LikeRecord.objects.filter(user=user, article=article)
        if like_record.exists():
            return HttpResponseRedirect(reverse('articleapp:detail', kwargs={'pk':kwargs['article_pk']}))
        else:
            LikeRecord(user=user, article=article).save()
        article.like += 1
        article.save()
        return super().get(request, *args, **kwargs)

    def  get_redirect_url(self, *args, **kwargs):
        return reverse('articleapp:detail', kwargs={'pk':kwargs['article_pk']}) 

# likeapp/urls.py
app_name='likeapp'

urlpatterns=[
    path('article/<int:article_pk>', LikeArticleView.as_view(), name='article_like'),
] 
```
```html
# articleapp/templates/articleapp/detail.html
            <div>
                <a href="{% url 'likeapp:article_like' article_pk=target_article.pk%}"
                class="material-icons">
                    favorite
                </a>
                <span>
                    {{ target_article.like }}
                </span>
            </div>


# articleapp/templates/articleapp/detail.html
            <div class="my-3">
                <a href="{% url 'likeapp:article_like' article_pk=target_article.pk%}"
                class="material-icons">
                class="material-icons"
                 style="vertical-align: middle; font-size: 1.5rem; color: cornflowerblue;">
                    favorite
                </a>
                <span>
                <span style="vertical-align: middle; font-size: 1.2rem">

```
```css
# static/base.css
a{
    text-decoration: none;
    color: darkslateblue;
}

a:hover{
    color: blueviolet;
} 
```

<hr>

## <3>
django message framework  
https://docs.djangoproject.com/en/3.2/ref/contrib/messages/

DB Transaction  
transaction : wrap multiple interaction into one  
transaction화 시키려면 장고에서 지원하는 데코레이터를 사용해야 한다..
```python
# likeapp/views.py
        like_record = LikeRecord.objects.filter(user=user,
                                                article=article)
        if like_record.exists():
            # 좋아요가 반영 X
            messages.add_message(request, messages.ERROR, '좋아요는 한번만 가능합니다.')
            return HttpResponseRedirect(reverse('articleapp:detail',
                                                kwargs={'pk': kwargs['article_pk']}))
        else:
            LikeRecord(user=user, article=article).save()
        article.like += 1
        article.save()
        # 좋아요가 반영 O
        messages.add_message(request, messages.SUCCESS, '좋아요가 반영되었습니다.')
        return super().get(request, *args, **kwargs)
```
```html
# templates/base.html
{% for message in messages %}
    <div class="text-center">
        <div class="btn btn-success rounded-pill px-5">
            {{ message }}
        </div>
    </div>
{% endfor %}

```
```python
# gisweb_1/settings.py
from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.ERROR: 'danger',
}

```
```html
# templates/base.html
        <div class="btn btn-{{ message.tags }} rounded-pill px-5">
            {{ message }}
        </div>
```
```python
# likeapp/views.py
@transaction.atomic
def db_transaction(user, article):
    article.like += 1
    article.save()
    
    like_record = LikeRecord.objects.filter(user=user, article=article)
    if like_record.exists():
        raise ValidationError('Like already_exists')
    else:
        LikeRecord(user=user, article=article).save()


@method_decorator(login_required, 'get')
class LikeArticleView(RedirectView):
    def get(self, request, *args, **kwargs):
        user = request.user
        article = Article.objects.get(pk=kwargs['article_pk'])

        try:
            db_transaction(user, article)
            # 좋아요 반영 o
            messages.add_message(request, messages.SUCCESS, '좋아요!!')
        except ValidationError:
            # 좋아요 반영 x
            messages.add_message(request, messages.ERROR, '좋아요는 한번만 눌러주세요!')
        return super().get(request, *args, **kwargs)

    def  get_redirect_url(self, *args, **kwargs):
        return reverse('articleapp:detail', kwargs={'pk':kwargs['article_pk']})
```
WYSIWYG : What You See Is What You Get  
https://yabwe.github.io/medium-editor/

```python
# articleapp/forms.py
class ArticleCreationForm(ModelForm):
    content = forms.CharField(widget=forms.Textarea(attrs={'class' : 'editable',
                                                           'style' : 'text-align: left;'
                                                                     'min-height: 10rem;'}))
```
```html
# articleapp/templates/articleapp/create.html
    <script src="//cdn.jsdelivr.net/npm/medium-editor@latest/dist/js/medium-editor.min.js"></script>
    <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/medium-editor@latest/dist/css/medium-editor.min.css" type="text/css" media="screen" charset="utf-8">

    <script>var editor = new MediumEditor('.editable');</script>


# articleapp/templates/articleapp/detail.html
                {{ target_article.content | safe}}
```

<hr>

# 11주차