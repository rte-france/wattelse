from django.urls import path
from gpt import views

app_name = "gpt"
urlpatterns = [
    path("", views.main_page, name="main_page"),
    path("query_gpt/", views.query_gpt, name="query_gpt"),
]
