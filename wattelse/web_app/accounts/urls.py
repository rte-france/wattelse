from django.urls import path
from . import views

app_name = "accounts"
urlpatterns = [
    path("", views.root_page, name="root_page"),
    path("login/", views.login, name="login"),
    path("register/", views.register, name="register"),
    path("change_password/", views.change_password, name="change_password"),
    path("logout/", views.logout, name="logout"),
]
