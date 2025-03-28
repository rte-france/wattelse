from django.db import migrations


def create_user_profiles(apps, schema_editor):
    # Use the historical versions of models to ensure compatibility during migrations
    User = apps.get_model("auth", "User")
    UserProfile = apps.get_model("chatbot", "UserProfile")

    users = User.objects.all()
    for user in users:
        if not UserProfile.objects.filter(
            user=user
        ).exists():  # Ensure profile is not already created
            # Create the UserProfile instance
            profile = UserProfile.objects.create(user=user)

            # Get the first group of the user, if it exists
            user_groups = user.groups.all()
            if user_groups.exists():
                profile.active_group = user_groups.first()
                profile.save()


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0006_userprofile"),
    ]

    operations = [
        migrations.RunPython(create_user_profiles),
    ]
