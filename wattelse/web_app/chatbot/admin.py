#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import Chat, GroupProfile, UserProfile


class ChatAdmin(admin.ModelAdmin):
    list_display = (
        "user_id",
        "group_id",
        "conversation_id",
        "message",
        "response",
        "question_timestamp",
        "answer_timestamp",
        "answer_delay",
        "short_feedback",
        "long_feedback",
        "rag_config",
    )
    list_filter = (
        "user_id",
        "group_id",
        "conversation_id",
        "question_timestamp",
        "answer_timestamp",
        "answer_delay",
        "short_feedback",
    )


class GroupProfileAdmin(admin.ModelAdmin):
    list_display = (
        "group_id",
        "system_prompt",
    )
    list_filter = ("group_id",)


class GroupProfileAdmin(admin.ModelAdmin):
    list_display = (
        "group",
        "rag_config",
        "system_prompt",
    )
    list_filter = ("group_id",)


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "active_group")


# Register your models here.
admin.site.register(Chat, ChatAdmin)
admin.site.register(GroupProfile, GroupProfileAdmin)
admin.site.register(UserProfile, UserProfileAdmin)
