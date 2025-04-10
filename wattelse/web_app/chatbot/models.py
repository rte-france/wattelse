#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.db import models
from django.contrib.auth.models import User, Group
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver


class Chat(models.Model):
    """
    Class used to represent an interaction with the chatbot for RAG
    """

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    group_id = models.TextField()
    conversation_id = models.UUIDField()
    message = models.TextField()
    response = models.TextField()
    question_timestamp = models.DateTimeField()
    answer_timestamp = models.DateTimeField(auto_now_add=True)
    short_feedback = models.TextField(default="")
    long_feedback = models.TextField(default="")
    answer_delay = models.DurationField(null=True, blank=True)  # Optional fields
    relevant_extracts = models.JSONField(default=list)
    group_system_prompt = models.TextField(default="")
    rag_config = models.CharField(max_length=100, null=False, blank=False)

    def __str__(self):
        return f"{self.user.username}: {self.message}"


class GPTChat(models.Model):
    """
    Class used to represent an interaction with the chatbot for pure generation without RAG
    """

    # NB. So far, we log the exact same information as for RAG
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    group_id = models.TextField()
    conversation_id = models.UUIDField()
    message = models.TextField()
    response = models.TextField()
    question_timestamp = models.DateTimeField()
    answer_timestamp = models.DateTimeField(auto_now_add=True)
    short_feedback = models.TextField(default="")
    long_feedback = models.TextField(default="")
    answer_delay = models.DurationField(null=True, blank=True)  # Optional fields

    def __str__(self):
        return f"{self.user.username}: {self.message}"


class GroupProfile(models.Model):
    group = models.OneToOneField(Group, on_delete=models.CASCADE, primary_key=True)
    rag_config = models.CharField(
        max_length=100, default="azure_20241216", null=False, blank=False
    )
    system_prompt = models.TextField(null=True, blank=True, default="")


class SuperUserPermissions(models.Model):
    """
    Dummy model for managing users permissions.
    """

    class Meta:
        # No database table creation or deletion
        managed = False

        # disable "add", "change", "delete" and "view" default permissions
        default_permissions = ()

        # (codename, name)
        permissions = (
            # User related permissions
            ("can_upload_documents", "Can upload documents"),
            ("can_remove_documents", "Can remove documents"),
            ("can_manage_users", "Can manage users"),
            # Group related permissions
            ("can_edit_group_system_prompt", "Can edit group system prompt"),
        )


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    active_group = models.ForeignKey(
        Group, on_delete=models.SET_NULL, null=True, blank=True
    )


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()


@receiver(m2m_changed, sender=User.groups.through)
def update_user_profile_active_group(
    sender, instance, action, reverse, model, pk_set, **kwargs
):
    if action == "post_add":
        user_profile = instance.userprofile
        # Check if the active_group is already set
        if user_profile.active_group is None:
            # Get the first group added (assuming only one group is added at a time)
            group = Group.objects.get(pk=list(pk_set)[0])
            user_profile.active_group = group
            user_profile.save()
