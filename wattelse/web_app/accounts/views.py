#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.


from django.shortcuts import render, redirect
from django.urls import reverse

from django.contrib import auth
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required


from loguru import logger


def root_page(request):
    return redirect(reverse("home:main_page"))


def login(request):
    """Main function for login page.
    If request method is GET : render login.html
    If request method is POST : log the user in and redirect to chatbot.html
    """
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = auth.authenticate(request, username=username, password=password)
        # If user exists: check group, login and redirect to chatbot
        if user is not None:
            auth.login(request, user)
            logger.info(f"[User: {request.user.username}] logged in")
            next_page = request.GET.get("next")
            if next_page:
                return redirect(next_page)
            else:
                return redirect(reverse("home:main_page"))
        # Else return error
        else:
            error_message = "Nom d'utilisateur ou mot de passe invalides"
            return render(
                request, "accounts/login.html", {"error_message": error_message}
            )
    else:
        return render(request, "accounts/login.html")


def register(request):
    """Main function for register page.
    If request method is GET : render register.html
    If request method is POST : create a new user and print an new_user_created webpage
    """
    if request.method == "POST":
        username = request.POST.get("username")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        # Check username is not already taken
        if User.objects.filter(username=username).exists():
            error_message = "Nom d'utilisateur indisponible"
            return render(
                request, "accounts/register.html", {"error_message": error_message}
            )

        # Check both password are the same
        if password1 == password2:
            try:
                user = User.objects.create_user(username, password=password1)
                user.save()
                return redirect(reverse("home:main_page"))
            except:
                error_message = "Erreur lors de la  création du compte"
                return render(
                    request, "accounts/register.html", {"error_message": error_message}
                )
        else:
            error_message = "Mots de passe non identiques"
            return render(
                request, "accounts/register.html", {"error_message": error_message}
            )
    return render(request, "accounts/register.html")


@login_required
def change_password(request):
    """Main function for change password page.
    If request method is GET : render change_password.html
    If request method is POST : change user password and print an password_changed webpage
    """
    if request.method == "POST":
        user = request.user
        password1 = request.POST.get("new_password1")
        password2 = request.POST.get("new_password2")

        # Check both password are the same
        if password1 == password2 and password1 != "":
            try:
                user.set_password(password1)
                user.save()
                update_session_auth_hash(request, user)
                return render(request, "accounts/password_changed.html")
            except:
                error_message = "Erreur lors du changement du mot de passe"
                return render(
                    request,
                    "accounts/change_password.html",
                    {"error_message": error_message},
                )
        else:
            error_message = "Mots de passe non identiques"
            return render(
                request,
                "accounts/change_password.html",
                {"error_message": error_message},
            )
    else:
        return render(request, "accounts/change_password.html")


def logout(request):
    """Log a user out and redirect to login page"""
    logger.info(f"[User: {request.user.username}] logged out")
    auth.logout(request)
    return redirect(reverse("accounts:login"))
