#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.contrib import admin
from .models import Chat, Update # Import your Chat model

admin.site.register(Chat) 
admin.site.register(Update)
# Register your models here.


