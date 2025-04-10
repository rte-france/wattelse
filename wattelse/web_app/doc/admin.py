#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import RAGConversation, RAGMessage


class RAGConversationAdmin(admin.ModelAdmin):
    list_display = ("user", "title", "created_at", "updated_at")


class RAGMessageAdmin(admin.ModelAdmin):
    list_display = (
        "group",
        "content",
        "role",
        "created_at",
        "rating",
        "faq_answer",
        "rag_config",
        "relevant_extracts",
    )


admin.site.register(RAGConversation, RAGConversationAdmin)
admin.site.register(RAGMessage, RAGMessageAdmin)


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
