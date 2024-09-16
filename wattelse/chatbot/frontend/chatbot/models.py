#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.db import models
from django.contrib.auth.models import User

# User is not an object per se, 
# but rather a model provided by Django itself. It's part of Django's built-in authentication system

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE) #why is this a FK ?
    # ForeignKey is a field used to define a many-to-one relationship.
    # single user can be associated with multiple chat instances, but each chat is associated with only one user.
    group_id = models.TextField()
    conversation_id = models.UUIDField()
    message = models.TextField()
    response = models.TextField()
    question_timestamp = models.DateTimeField()
    answer_timestamp = models.DateTimeField(auto_now_add=True)
    short_feedback = models.TextField(default="")
    suggested_update = models.TextField(default="")
    answer_delay = models.DurationField(null=True, blank=True)  # Optional fields

    def __str__(self):
        return f'{self.user.username}: {self.message}'

    def calculate_answer_delay(self):
        if self.answer_timestamp and self.question_timestamp:
            return self.answer_timestamp - self.question_timestamp
        return None

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs) # required to have self.*_timestamp set
        self.answer_delay = self.calculate_answer_delay()
        super().save(*args, **kwargs) # required to save self.answer_delay


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
            ("can_upload_documents", "Can upload documents"),
            ("can_remove_documents", "Can remove documents"),
            ("can_manage_users", "Can manage users"),
           ("can_download_updates", "Can download updates"),
        )
        
        
class Update(models.Model):
    """
    Model for storing user suggestions for updates.
    """
    extract_id = models.CharField(max_length=50)
    previous_version= models.TextField(default="")
    updated_extract = models.TextField(default="")
    document_name = models.CharField(max_length=100)
    update_timestamp = models.DateTimeField()
    user = models.ForeignKey(User, default="User deleted", on_delete=models.SET_DEFAULT)
    group_id = models.TextField(null=True)
    chat = models.ForeignKey(Chat, null=True, blank=True, on_delete=models.SET_NULL)



    def __str__(self):
        return f'{self.extract_id}: {self.document_name}'




    