from django.contrib import admin

# Register your models here.
#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import GroupProfile, UserProfile


class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "active_group")


class GroupProfileAdmin(admin.ModelAdmin):
    list_display = ("group", "rag_config", "system_prompt")
    list_filter = ("group_id",)


# Register your models here.
admin.site.register(GroupProfile, GroupProfileAdmin)
admin.site.register(UserProfile, UserProfileAdmin)
