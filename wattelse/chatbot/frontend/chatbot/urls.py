#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.urls import path
from . import secureGPT_views, views

app_name = "chatbot"
urlpatterns = [
    # Web pages
    path("", views.default_page, name="default_page"),
    path("doc/", views.rag_page, name="main_page"),
    path("login/", views.login, name="login"),
    path("register/", views.register, name="register"),
    path("logout/", views.logout, name="logout"),
    path("faq/", views.faq_page, name="faq"),
    # Chatbot
    path("query_rag/", views.query_rag, name="query_rag"),
    path("save_interaction/", views.save_interaction, name="save_interaction"),
    path(
        "send_short_feedback/",
        views.manage_short_feedback,
        name="manage_short_feedback",
    ),
    path(
        "send_long_feedback/", views.manage_long_feedback, name="manage_long_feedback"
    ),
    # Feedback count
    path(
        "get_questions_count_since_last_feedback/",
        views.get_questions_count_since_last_feedback,
        name="get_questions_count_since_last_feedback",
    ),
    # Files
    path("upload/", views.upload, name="upload"),
    path("delete/", views.delete, name="delete"),
    path("file_viewer/<str:file_name>", views.file_viewer, name="file_viewer"),
    # Users
    path("add_user_to_group/", views.add_user_to_group, name="add_user_to_group"),
    path(
        "remove_user_from_group/",
        views.remove_user_from_group,
        name="remove_user_from_group",
    ),
    path("admin_change_group/", views.admin_change_group, name="admin_change_group"),
    path(
        "manage_superuser_permission/",
        views.manage_superuser_permission,
        name="manage_superuser_permission",
    ),
    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),
    # GPT chat
    path("gpt/", secureGPT_views.gpt_page, name="gpt_chat"),
    path("query_gpt/", secureGPT_views.query_gpt, name="query_gpt"),
    # Conversations history management
    path(
        "get_conversation_messages/",
        secureGPT_views.get_conversation_messages,
        name="get_conversation_history",
    ),
    # FAQ
    path("add_to_faq/", views.add_faq_item, name="add_to_faq"),
    path("delete_from_faq/", views.delete_faq_item, name="delete_from_faq"),
]
