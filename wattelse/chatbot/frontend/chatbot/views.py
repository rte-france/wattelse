#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import io
import uuid
import json
import socket
import tempfile

import mammoth
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404, StreamingHttpResponse

from django.contrib import auth
from django.contrib.auth.models import User, Group
from loguru import logger
from pathlib import Path

from xlsx2html import xlsx2html

from wattelse.api.rag_orchestrator.rag_client import RAGAPIError
from wattelse.chatbot.backend import DATA_DIR
from .models import Chat

from .utils import (
    get_user_group_id,
    get_group_usernames_list,
    new_user_created,
    get_conversation_history,
    streaming_generator,
    insert_feedback,
    RAG_API,
)


def main_page(request):
    """
    Main function for chatbot interface.
    Render chatbot.html webpage with associated context.
    """
    # If user is not authenticated, redirect to login
    if not request.user.is_authenticated:
        return redirect("/login")

    # Get user group_id
    user_group_id = get_user_group_id(request.user)

    # Generate a new conversation_id
    conversation_id = str(uuid.uuid4())

    # Get list of available documents
    try:
        available_docs = RAG_API.list_available_docs(user_group_id)
    except RAGAPIError:
        return redirect("/login")

    # Get user permissions
    can_upload_documents = request.user.has_perm("chatbot.can_upload_documents")
    can_remove_documents = request.user.has_perm("chatbot.can_remove_documents")
    can_manage_users = request.user.has_perm("chatbot.can_manage_users")

    # If can manage users, find usernames of its group
    if can_manage_users:
        group_usernames_list = get_group_usernames_list(user_group_id)
        # Remove admin so it cannot be deleted
        try:
            group_usernames_list.remove("admin")
        except ValueError:
            pass  # admin may not be in the list depending on how users have been defined
    else:
        group_usernames_list = None

    # Special case for admin
    if request.user.is_superuser:
        admin_group_selection = [group.name for group in Group.objects.filter()]
    else:
        admin_group_selection = None
    return render(
        request, "chatbot/chatbot.html",
        {
            "conversation_id": conversation_id,
            "available_docs": available_docs,
            "can_upload_documents": can_upload_documents,
            "can_remove_documents": can_remove_documents,
            "can_manage_users": can_manage_users,
            "user_group": user_group_id,
            "group_usernames_list": group_usernames_list,
            "admin_group_selection": admin_group_selection,
        }
    )


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
            # If user doesn't belong to a group, return error
            user_group_id = get_user_group_id(user)
            if user_group_id is None:
                error_message = "Vous n'appartenez à aucun groupe."
                return render(request, "chatbot/login.html", {"error_message": error_message})
            else:
                auth.login(request, user)
                logger.info(f"[User: {request.user.username}] logged in")
                RAG_API.create_session(user_group_id)
                return redirect("/")
        # Else return error
        else:
            error_message = "Nom d'utilisateur ou mot de passe invalides"
            return render(request, "chatbot/login.html", {"error_message": error_message})
    else:
        return render(request, "chatbot/login.html")


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
            return render(request, "chatbot/register.html", {"error_message": error_message})

        # Check both password are the same
        if password1 == password2:
            try:
                user = User.objects.create_user(username, password=password1)
                user.save()
                return new_user_created(request, username=user.username)
            except:
                error_message = "Erreur lors de la  création du compte"
                return render(request, "chatbot/register.html", {"error_message": error_message})
        else:
            error_message = "Mots de passe non identiques"
            return render(request, "chatbot/register.html", {"error_message": error_message})
    return render(request, "chatbot/register.html")


def logout(request):
    """Log a user out and redirect to login page"""
    logger.info(f"[User: {request.user.username}] logged out")
    auth.logout(request)
    return redirect("/login")


def query_rag(request):
    """
    Main function for query RAG calls.
    Call RAGOrchestratorAPI with `stream=True` and streams the response to frontend.
    First chunk contains `relevant_extracts` data,
    other chunks contain streamed answer tokens.
    """
    if request.method == "POST":
        # Get user group_id
        user_group_id = get_user_group_id(request.user)

        # Get conversation id
        conversation_id = uuid.UUID(request.POST.get("conversation_id"))

        # Get user chat history
        history = get_conversation_history(request.user, conversation_id)

        # Get posted message
        message = request.POST.get("message", None)

        if not message:
            logger.warning(f"[User: {request.user.username}] No user message received")
            error_message = "Veuillez saisir une question"
            return JsonResponse({"error_message": error_message}, status=500)
        logger.info(f"[User: {request.user.username}] Query: {message}")

        # Select documents for RAG
        selected_docs = request.POST.get("selected_docs", None)
        selected_docs = json.loads(selected_docs)
        logger.debug(f"[User: {request.user.username}] Selected docs: {selected_docs}")
        if not selected_docs:
            logger.warning(f"[User: {request.user.username}] No selected docs received, using all available docs")
            selected_docs = []

        # Query RAG and stream response
        try:
            response = RAG_API.query_rag(
                user_group_id,
                message,
                history=history,
                selected_files=selected_docs,
                stream=True,
            )

            return StreamingHttpResponse(streaming_generator(response), status=200, content_type='text/event-stream')

        except RAGAPIError as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse({"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500)
    else:
        raise Http404()


def save_interaction(request):
    """Function called to save query and response in DB once response streaming is finished."""
    if request.method == "POST":
        # Save query and response in DB
        chat = Chat(
            user=request.user,
            group_id=get_user_group_id(request.user),
            conversation_id=request.POST.get("conversation_id", ""),
            message=request.POST.get("message", ""),
            response=request.POST.get("answer", ""),
        )
        chat.save()
        return HttpResponse(status=200)
    else:
        raise Http404()


def manage_short_feedback(request):
    """
    Function that collects short feedback sent from the user interface about the last
    interaction (matching between the user question and the bot answer).
    """
    return insert_feedback(request, short=True)


def manage_long_feedback(request):
    """
    Function that collects long feedback sent from the user interface about the last
    interaction (matching between the user question and the bot answer).
    """
    return insert_feedback(request, short=False)


def upload(request):
    """Main function for delete interface.
    If request method is POST : make a call to RAGOrchestratorClient to upload the specified documents
    """
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            logger.warning(f"[User: {request.user.username}] No file to be uploaded, action ignored")
            return JsonResponse({"error_message": "No file received!"}, status=500)
        else:
            user_group_id = get_user_group_id(request.user)
            logger.debug(f"[User: {request.user.username}] Received file: {uploaded_file.name}")

            # Create a temporary directory
            # TODO: investigate in memory temp file, probably a better option
            with tempfile.TemporaryDirectory() as temp_dir:
                # Construct the full path for the uploaded file within the temporary directory
                temp_file_path = Path(temp_dir) / Path(uploaded_file.name)

                # Open the temporary file for writing
                with open(temp_file_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                try:
                    # Use the temporary file path for upload
                    RAG_API.upload_files(user_group_id, [temp_file_path])
                except RAGAPIError as e:
                    logger.error(f"[User: {request.user.username}] {e}")
                    return JsonResponse({"error_message": f"Erreur de téléchargement de {uploaded_file.name}\n{e}"},
                                        status=500)

            # Returns the list of updated available documents
            return JsonResponse({"available_docs": RAG_API.list_available_docs(user_group_id)}, status=200)


def delete(request):
    """Main function for delete interface.
    If request method is POST : make a call to RAGOrchestratorClient to delete the specified documents
    """
    if request.method == "POST":
        # Select documents for removal
        selected_docs = request.POST.get("selected_docs", None)
        logger.debug(f"[User: {request.user.username}] Docs selected for removal: {selected_docs}")
        if not selected_docs:
            logger.warning(f"[User: {request.user.username}] No docs selected for removal received, action ignored")
            return JsonResponse({"warning_message": "No document removed"}, status=202)
        else:
            user_group_id = get_user_group_id(request.user)
            try:
                rag_response = RAG_API.remove_documents(user_group_id, json.loads(selected_docs))
            except RAGAPIError as e:
                logger.error(f"[User: {request.user.username}] Error in deleting documents {selected_docs}: {e}")
                return JsonResponse({"error_message": f"Erreur pour supprimer les documents {selected_docs}"},
                                    status=500)
            # Returns the list of updated available documents
            return JsonResponse({"available_docs": RAG_API.list_available_docs(user_group_id)}, status=200)


def file_viewer(request, file_name: str):
    """
    Main function to render a file. The url to access this function should be :
    file_viewer/file_name.suffix
    It will render the file if the user belongs to the right group and if the file format is supported.
    """
    # TODO: manage more file type
    # FIXME: to be rethought in case of distributed deployment (here we suppose RAG backend and Django on the same server...)
    file_path = DATA_DIR / get_user_group_id(request.user) / file_name
    if file_path.exists():
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            content_type = 'application/pdf'
            with open(file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type=content_type)
                response['Content-Disposition'] = f'inline; filename="{file_path.name}"'
                return response
        elif suffix == ".docx":
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value  # The generated HTML
                return HttpResponse(html)
        elif suffix == ".xlsx":
            xlsx_file = open(file_path, 'rb')
            out_file = io.StringIO()
            xlsx2html(xlsx_file, out_file, locale='en')
            out_file.seek(0)
            return HttpResponse(out_file.read())
    else:
        raise Http404()


def add_user_to_group(request):
    """
    Function to add a new user to a group.
    The superuser send a POST request with data `new_username`.
    If `new_username` exists and doesn't belong to any group, add it to superuser group.
    Else return error to frontend.
    """
    if request.method == "POST":
        # Get superuser group object
        superuser = request.user
        superuser_group_id = get_user_group_id(superuser)
        superuser_group = Group.objects.get(name=superuser_group_id)

        # Get new_username user object if it exists
        new_username = request.POST.get("new_username", None)
        if User.objects.filter(username=new_username).exists():
            new_user = User.objects.get(username=new_username)
        else:
            logger.error(f"[User: {request.user.username}] Username {new_username} not found")
            error_message = f"Le nom d'utilisateur {new_username} n'a pas été trouvé"
            return JsonResponse({"error_message": error_message}, status=500)

        # If new_user already in a group then return error status code
        if get_user_group_id(new_user) is not None:
            logger.error(
                f"[User: {request.user.username}] User with username {new_username} already belongs to a group")
            error_message = f"L'utilisateur {new_username} appartient déjà à un groupe"
            return JsonResponse({"error_message": error_message}, status=500)

        # If new_user has no group then add it to superuser group
        else:
            logger.info(f"[User: {request.user.username}] Adding {new_username} to group {superuser_group_id}")
            new_user.groups.add(superuser_group)
            return HttpResponse(status=201)
    else:
        raise Http404()


def remove_user_from_group(request):
    """
    Function to remove a user from a group.
    The superuser send a POST request with data `username_to_delete`.
    If `username_to_delete` exists, remove it from superuser group.
    Else return error to frontend.
    """
    if request.method == "POST":
        # Get superuser group object
        superuser = request.user
        superuser_group_id = get_user_group_id(superuser)
        superuser_group = Group.objects.get(name=superuser_group_id)

        # Get username_to_remove user object if it exists
        username_to_remove = request.POST.get("username_to_delete", None)
        if User.objects.filter(username=username_to_remove).exists():
            user_to_remove = User.objects.get(username=username_to_remove)
        else:
            error_message = f"Le nom d'utilisateur {username_to_remove} n'a pas été trouvé"
            return JsonResponse({"error_message": error_message}, status=500)

        # Send an error if a user tries to remove himself
        if request.user.username == username_to_remove:
            error_message = f"Vous ne pouvez pas vous supprimer du groupe"
            return JsonResponse({"error_message": error_message}, status=500)

        # Remove user_to_remove
        logger.info(f"[User: {request.user.username}] Removing {username_to_remove} from group {superuser_group_id}")
        user_to_remove.groups.remove(superuser_group)
        return HttpResponse(status=201)
    else:
        raise Http404()


def admin_change_group(request):
    """Special function for admins to change group using web interface"""
    if request.method == "POST":
        print(request.POST)
        if request.user.is_superuser:
            request.user.groups.clear()
            new_group = Group.objects.get(name=request.POST.get("new_group"))
            request.user.groups.add(new_group)
            return redirect("/")
        else:
            raise Http404()
    else:
        raise Http404()


def dashboard(request):
    return redirect(f"http://{socket.gethostbyname(socket.gethostname())}:9090")
