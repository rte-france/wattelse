#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import Chat, GPTChat, GroupSystemPrompt


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


class GroupSystemPromptAdmin(admin.ModelAdmin):
    list_display = (
        "group_id",
        "system_prompt",
    )
    list_filter = ("group_id",)


class GroupSystemPromptAdmin(admin.ModelAdmin):
    list_display = (
        "group_id",
        "system_prompt",
    )
    list_filter = ("group_id",)


# Register your models here.
admin.site.register(Chat, ChatAdmin)
admin.site.register(GPTChat, ChatAdmin)
admin.site.register(GroupSystemPrompt, GroupSystemPromptAdmin)
