#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import Chat


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
    )
    list_filter = (
        "user_id",
        "group_id",
        "conversation_id",
        "question_timestamp",
        "answer_timestamp",
        "answer_delay",
    )


# Register your models here.
admin.site.register(Chat, ChatAdmin)
