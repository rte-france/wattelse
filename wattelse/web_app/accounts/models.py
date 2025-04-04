from django.db import models
from django.contrib.auth.models import User, Group
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    active_group = models.ForeignKey(
        Group, on_delete=models.SET_NULL, null=True, blank=True
    )


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
