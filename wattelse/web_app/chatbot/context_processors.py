#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.conf import settings


def versioning(request):
    return {
        "STATIC_VERSION": settings.STATIC_VERSION,
    }
