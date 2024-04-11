#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.urls import path
from . import views, simple_views, feedback_views

app_name = 'chatbot'
urlpatterns = [
    path('', views.chatbot, name='chatbot'),
	path('query_rag/', views.query_rag, name='query_rag'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout, name='logout'),
    path('delete/', views.delete, name='delete'),
    path('upload/', views.upload, name='upload'),
    path('file_viewer/<str:file_name>', views.file_viewer, name='file_viewer'),
    path('add_user_to_group/', views.add_user_to_group, name='add_user_to_group'),
    path('remove_user_from_group/', views.remove_user_from_group, name='remove_user_from_group'),
    path('llm/', simple_views.basic_chat, name='basic_chat'),
    path('send_short_feedback/', feedback_views.manage_short_feedback, name="manage_short_feedback"),
    path('send_long_feedback/', feedback_views.manage_long_feedback, name="manage_long_feedback"),
    path('dashboard/', views.dashboard, name="dashboard"),
]
