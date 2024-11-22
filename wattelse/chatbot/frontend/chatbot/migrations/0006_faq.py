# Generated by Django 5.1.1 on 2024-11-13 16:45

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0005_chat_relevant_extracts"),
    ]

    operations = [
        migrations.CreateModel(
            name="FAQ",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("question", models.TextField()),
                ("answer", models.TextField()),
                ("group_id", models.TextField()),
            ],
        ),
    ]
