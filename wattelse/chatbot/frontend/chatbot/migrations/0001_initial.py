#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Generated by Django 5.0.2 on 2024-04-08 08:29

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='SuperUserPermissions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            options={
                'permissions': (('can_upload_documents', 'Can upload documents'), ('can_remove_documents', 'Can remove documents'), ('can_manage_users', 'Can manage users')),
                'managed': False,
                'default_permissions': (),
            },
        ),
        migrations.CreateModel(
            name='Chat',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('group_id', models.TextField()),
                ('conversation_id', models.UUIDField()),
                ('message', models.TextField()),
                ('response', models.TextField()),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('short_feedback', models.TextField(default='')),
                ('long_feedback', models.TextField(default='')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
