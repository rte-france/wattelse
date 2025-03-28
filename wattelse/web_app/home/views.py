from django.shortcuts import render
from django.contrib.auth.decorators import login_required


@login_required
def home(request):
    """Home page where user can select an app."""
    return render(request, "home/home.html")
