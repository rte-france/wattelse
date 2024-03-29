#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

# Generated by Django 5.0.2 on 2024-03-22 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='SuperUserPermissions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            options={
                'permissions': (('can_upload_documents', 'Can upload documents'), ('can_remove_documents', 'Can remove documents'), ('can_add_users', 'Can add users')),
                'managed': False,
                'default_permissions': (),
            },
        ),
    ]
