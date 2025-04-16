import uuid
from django.db import models
from django.contrib.auth.models import User


class Conversation(models.Model):
    """
    Represents a chat conversation between a user and the assistant.
    Each conversation contains multiple messages and belongs to a specific user.
    """

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


class Message(models.Model):
    """
    Represents an individual message within a conversation.
    Messages can be from either the user or the assistant
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, related_name="messages", on_delete=models.CASCADE
    )
    role = models.CharField(
        max_length=10,
        choices=[("user", "User"), ("assistant", "Assistant")],
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    rating = models.BooleanField(null=True, blank=True)
    model = models.CharField(max_length=255, blank=True)
