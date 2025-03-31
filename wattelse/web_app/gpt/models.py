import uuid
from django.db import models
from django.contrib.auth.models import User


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(
                fields=["user", "created_at"]
            ),  # For queries that filter by user and sort by date
        ]


class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation, related_name="messages", on_delete=models.CASCADE
    )
    role = models.CharField(
        max_length=10,
        choices=[("user", "User"), ("assistant", "Assistant")],
    )
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    # class Meta:
    #     indexes = [
    #         models.Index(
    #             fields=["conversation", "timestamp"]
    #         ),  # For retrieving messages in a conversation by time
    #         models.Index(fields=["role"]),  # If you often query by role
    #     ]
