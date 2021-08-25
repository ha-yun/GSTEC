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