#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Generated by Django 5.1.1 on 2024-11-20 16:59

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chatbot", "0004_gptchat"),
    ]

    operations = [
        migrations.CreateModel(
            name="GroupSystemPrompt",
            fields=[
                ("group_id", models.TextField(primary_key=True, serialize=False)),
                ("system_prompt", models.TextField()),
            ],
        ),
        migrations.AlterModelOptions(
            name="superuserpermissions",
            options={
                "default_permissions": (),
                "managed": False,
                "permissions": (
                    ("can_upload_documents", "Can upload documents"),
                    ("can_remove_documents", "Can remove documents"),
                    ("can_manage_users", "Can manage users"),
                    ("can_edit_group_system_prompt", "Can edit group system prompt"),
                ),
            },
        ),
        migrations.AddField(
            model_name="chat",
            name="group_system_prompt",
            field=models.TextField(default=""),
        ),
        migrations.AddField(
            model_name="chat",
            name="relevant_extracts",
            field=models.JSONField(default=list),
        ),
    ]
