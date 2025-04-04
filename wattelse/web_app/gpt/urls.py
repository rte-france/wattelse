from django.urls import path
from gpt import views

app_name = "gpt"
urlpatterns = [
    path("", views.main_page, name="main_page"),
    path("query_gpt/", views.query_gpt, name="query_gpt"),
    path(
        "save_assistant_message/",
        views.save_assistant_message,
        name="save_assistant_message",
    ),
    path(
        "get_conversation_messages/",
        views.get_conversation_messages,
        name="get_conversation_messages",
    ),
    path("handle_vote/", views.handle_vote, name="handle_vote"),
]
