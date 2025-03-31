#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.urls import path
from . import views

app_name = "chatbot"
urlpatterns = [
    # Web pages
    path("", views.default_page, name="default_page"),
    path("doc/", views.rag_page, name="main_page"),
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
    path(
        "change_active_group/", views.change_active_group, name="0change_active_group"
    ),
    path(
        "manage_superuser_permission/",
        views.manage_superuser_permission,
        name="manage_superuser_permission",
    ),
    path(
        "update_group_system_prompt/",
        views.update_group_system_prompt,
        name="update_group_system_prompt",
    ),
    # Dashboard
    path("dashboard/", views.dashboard, name="dashboard"),
    # Conversations history management
    path(
        "get_conversation_messages/",
        views.get_conversation_messages,
        name="get_conversation_history",
    ),
]
