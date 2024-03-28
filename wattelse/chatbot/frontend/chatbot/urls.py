from django.urls import path
from . import views, simple_views

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
	path('add_user_to_group/', views.add_user_to_group, name='add_user_to_group'),
	path('remove_user_from_group/', views.remove_user_from_group, name='remove_user_from_group'),
    path('llm/', simple_views.basic_chat, name='basic_chat'),
]