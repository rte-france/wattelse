from django.conf import settings


def versioning(request):
    return {
        "STATIC_VERSION": settings.STATIC_VERSION,
    }
