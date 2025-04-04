#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import uuid
from django.db import models
from django.contrib.auth.models import User, Group


class RAGConversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
        ]


class RAGMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        RAGConversation, related_name="messages", on_delete=models.CASCADE
    )
    role = models.CharField(
        max_length=10,
        choices=[("user", "User"), ("assistant", "Assistant")],
    )
    content = models.TextField()
    group = models.ForeignKey(Group, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    rating = models.BooleanField(null=True, blank=True)
    faq_answer = models.TextField(default="")
    group_system_prompt = models.TextField(default="")
    rag_config = models.CharField(max_length=100, null=False, blank=False)
    relevant_extracts = models.JSONField(default=list)
