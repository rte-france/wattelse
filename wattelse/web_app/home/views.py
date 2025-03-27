from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET


@login_required
@require_GET
def main_page(request):
    """
    WattElse home page where user can select an app.
    """
    return render(request, "home/home.html")
