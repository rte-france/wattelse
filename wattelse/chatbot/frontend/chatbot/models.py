#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.db import models
from django.contrib.auth.models import User

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username}: {self.message}'
    
        
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
        )
