from django.urls import path
from . import views

app_name = 'chatbot'
urlpatterns = [
    path('', views.chatbot, name='chatbot'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout, name='logout'),
	path('reset/', views.reset, name='reset'),
    path('delete/', views.delete, name='delete'),
    path('upload/', views.upload, name='upload'),
    path('file_viewer/<str:file_name>', views.file_viewer, name='file_viewer'),
]