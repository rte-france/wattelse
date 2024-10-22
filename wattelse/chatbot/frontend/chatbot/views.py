#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import io
import uuid
import json
import socket
import tempfile
from datetime import datetime, timedelta

import mammoth
import pytz
from django.db.models import Q
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404, StreamingHttpResponse

from django.contrib import auth
from django.contrib.auth.models import User, Group, Permission
from django.contrib.contenttypes.models import ContentType
from loguru import logger
from pathlib import Path

from xlsx2html import xlsx2html

from wattelse.api.rag_orchestrator.rag_client import RAGAPIError
from wattelse.chatbot.backend import DATA_DIR
from .models import Chat, SuperUserPermissions

from .utils import (
    get_user_group_id,
    get_group_usernames_list,
    new_user_created,
    get_conversation_history,
    streaming_generator,
    insert_feedback,
    RAG_API,
    get_chat_model,
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
        group_usernames_dict = get_group_usernames_list(user_group_id)
    else:
        group_usernames_dict = None

    # Special case for admin
    if request.user.is_superuser:
        admin_group_selection = [group.name for group in Group.objects.filter()]
    else:
        admin_group_selection = None
    return render(
        request,
        "chatbot/chatbot.html",
        {
            "conversation_id": conversation_id,
            "available_docs": available_docs,
            "can_upload_documents": can_upload_documents,
            "can_remove_documents": can_remove_documents,
            "can_manage_users": can_manage_users,
            "user_group": user_group_id,
            "group_usernames_dict": group_usernames_dict,
            "admin_group_selection": admin_group_selection,
        },
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
                return render(
                    request, "chatbot/login.html", {"error_message": error_message}
                )
            else:
                auth.login(request, user)
                logger.info(f"[User: {request.user.username}] logged in")
                RAG_API.create_session(user_group_id)
                return redirect("/")
        # Else return error
        else:
            error_message = "Nom d'utilisateur ou mot de passe invalides"
            return render(
                request, "chatbot/login.html", {"error_message": error_message}
            )
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
            return render(
                request, "chatbot/register.html", {"error_message": error_message}
            )

        # Check both password are the same
        if password1 == password2:
            try:
                user = User.objects.create_user(username, password=password1)
                user.save()
                return new_user_created(request, username=user.username)
            except:
                error_message = "Erreur lors de la  création du compte"
                return render(
                    request, "chatbot/register.html", {"error_message": error_message}
                )
        else:
            error_message = "Mots de passe non identiques"
            return render(
                request, "chatbot/register.html", {"error_message": error_message}
            )
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
        # Get request data
        data = json.loads(request.body)

        # Get user group_id
        user_group_id = get_user_group_id(request.user)

        # Get conversation id
        conversation_id = uuid.UUID(data.get("conversation_id"))

        # Get user chat history
        history = get_conversation_history(
            request.user, conversation_id, ChatModel=Chat
        )

        # Get posted message
        message = data.get("message", None)

        # Check message is not empty
        if not message:
            logger.warning(f"[User: {request.user.username}] No user message received")
            return JsonResponse({"message": "Aucune question reçue"}, status=500)

        # Log message
        logger.info(f"[User: {request.user.username}] Query: {message}")

        # Select documents for RAG
        selected_docs = data.get("selected_docs", None)
        logger.info(f"[User: {request.user.username}] Selected docs: {selected_docs}")
        if not selected_docs:
            logger.warning(
                f"[User: {request.user.username}] No selected docs received, using all available docs"
            )
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

            return StreamingHttpResponse(
                streaming_generator(response),
                status=200,
                content_type="text/event-stream",
            )

        except RAGAPIError as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500
            )
    else:
        raise Http404()


def save_interaction(request):
    """Function called to save query and response in DB once response streaming is finished."""
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)

        # Transform timestamps to datetime objects
        question_timestamp = data.get("question_timestamp", None)
        question_timestamp = datetime.fromisoformat(
            question_timestamp.replace("Z", "+00:00")
        )

        # Transform delay to timedelta
        answer_delay = data.get("answer_delay", None)
        answer_delay = timedelta(milliseconds=answer_delay)

        # Get database to use based on source path
        ChatModel = get_chat_model(data.get("source_path"))

        # Save interaction
        try:
            chat = ChatModel(
                user=request.user,
                group_id=get_user_group_id(request.user),
                conversation_id=data.get("conversation_id", ""),
                message=data.get("message", ""),
                response=data.get("answer", ""),
                question_timestamp=question_timestamp,
                answer_delay=answer_delay,
            )
            chat.save()

            # No need to show a pop up message to the user
            return HttpResponse(status=200)

        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {
                    "message": "Erreur serveur : échec lors de l'enregistrement de la conversation"
                },
                status=500,
            )
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
    if request.method == "POST":
        # Get the uploaded file
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            logger.warning(
                f"[User: {request.user.username}] No file to be uploaded, action ignored"
            )
            return JsonResponse({"error_message": "No file received!"}, status=500)
        else:
            user_group_id = get_user_group_id(request.user)
            logger.debug(
                f"[User: {request.user.username}] Received file: {uploaded_file.name}"
            )

            # Create a temporary directory
            # TODO: investigate in memory temp file, probably a better option
            with tempfile.TemporaryDirectory() as temp_dir:
                # Construct the full path for the uploaded file within the temporary directory
                temp_file_path = Path(temp_dir) / Path(uploaded_file.name)

                # Open the temporary file for writing
                with open(temp_file_path, "wb") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                try:
                    # Use the temporary file path for upload
                    RAG_API.upload_files(user_group_id, [temp_file_path])
                except RAGAPIError as e:
                    logger.error(f"[User: {request.user.username}] {e}")
                    return JsonResponse(
                        {
                            "error_message": f"Erreur de téléchargement de {uploaded_file.name}\n{e}"
                        },
                        status=500,
                    )

            # Returns the list of updated available documents
            return JsonResponse(
                {"available_docs": RAG_API.list_available_docs(user_group_id)},
                status=200,
            )


def delete(request):
    """Main function for delete interface.
    If request method is POST : make a call to RAGOrchestratorClient to delete the specified documents.
    """
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)

        # Select documents for removal
        selected_docs = data.get("selected_docs", None)
        logger.info(
            f"[User: {request.user.username}] Docs selected for removal: {selected_docs}"
        )
        if not selected_docs:
            logger.warning(
                f"[User: {request.user.username}] No docs selected for removal received, action ignored"
            )
            return JsonResponse({"message": "Aucun document sélectionné"}, status=500)
        else:
            user_group_id = get_user_group_id(request.user)
            try:
                RAG_API.remove_documents(user_group_id, selected_docs)
                return JsonResponse(
                    {
                        "available_docs": RAG_API.list_available_docs(user_group_id),
                        "message": "Documents supprimés",
                    },
                    status=200,
                )
            except Exception as e:
                logger.error(f"[User: {request.user.username}] {e}")
                return JsonResponse(
                    {
                        "message": f"Erreur serveur : échec lors de la suppression des documents"
                    },
                    status=500,
                )
    else:
        raise Http404()


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
            content_type = "application/pdf"
            with open(file_path, "rb") as f:
                response = HttpResponse(f.read(), content_type=content_type)
                response["Content-Disposition"] = f'inline; filename="{file_path.name}"'
                return response
        elif suffix == ".docx":
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value  # The generated HTML
                return HttpResponse(html)
        elif suffix == ".xlsx":
            xlsx_file = open(file_path, "rb")
            out_file = io.StringIO()
            xlsx2html(xlsx_file, out_file, locale="en")
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
        data = json.loads(request.body)
        new_username = data.get("new_username", None)
        if User.objects.filter(username=new_username).exists():
            new_user = User.objects.get(username=new_username)
        else:
            logger.error(
                f"[User: {request.user.username}] Username {new_username} not found"
            )
            return JsonResponse(
                {"message": f"{new_username} non trouvé"},
                status=500,
            )

        # If new_user already in a group then return error status code
        if get_user_group_id(new_user) is not None:
            logger.error(
                f"[User: {request.user.username}] Username {new_username} already belongs to a group"
            )
            return JsonResponse(
                {"message": f"{new_username} appartient déjà à un groupe"},
                status=500,
            )

        # If new_user has no group then add it to superuser group
        try:
            new_user.groups.add(superuser_group)
            logger.info(
                f"[User: {request.user.username}] Added {new_username} to group {superuser_group_id}"
            )
            return JsonResponse(
                {"message": f"{new_username} ajouté au groupe"},
            )
        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {"message": "Erreur serveur : échec lors de l'ajout de l'utilisateur"},
                status=500,
            )
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
        data = json.loads(request.body)
        username_to_remove = data.get("username_to_delete", None)
        if User.objects.filter(username=username_to_remove).exists():
            user_to_remove = User.objects.get(username=username_to_remove)
        else:
            return JsonResponse(
                {"message": f"{username_to_remove} non trouvé"},
                status=500,
            )

        # Send an error if a user tries to remove himself
        if request.user.username == username_to_remove:
            return JsonResponse({"message": "Impossible de vous supprimer"}, status=500)

        # Remove user_to_remove
        try:
            user_to_remove.groups.remove(superuser_group)
            logger.info(
                f"[User: {request.user.username}] Removed {username_to_remove} from group {superuser_group_id}"
            )
            return JsonResponse({"message": f"{username_to_remove} supprimé"})
        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {"message": "Erreur serveur : échec lors de la suppression"}, status=500
            )
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


def get_questions_count_since_last_feedback(request):
    """
    Counts the number of entries without feedback since the user's last feedback
    Returns:
        int: The number of entries without feedback since the last feedback.
    """
    if request.method == "GET":

        # Get request data
        source_path = request.GET.get("source_path")

        # Get database to use based on source path
        ChatModel = get_chat_model(source_path)

        try:
            last_feedback = (
                ChatModel.objects.filter(
                    ~Q(short_feedback__exact=""),
                    user=request.user,
                    short_feedback__isnull=False,
                )
                .order_by("-question_timestamp")
                .first()
            )
            if last_feedback:
                last_feedback_date = last_feedback.question_timestamp
            else:
                return JsonResponse({"count": 9999})  # arbitrary big value
        except ChatModel.DoesNotExist:
            return JsonResponse({"count": 9999})  # arbitrary big value

        count = ChatModel.objects.filter(
            user=request.user, question_timestamp__gt=last_feedback_date
        ).count()
        return JsonResponse({"count": count})
    else:
        raise Http404()


def manage_superuser_permission(request):
    """
    Function to allow a superuser to add or remove superuser permissions
    for users of its group. The request should contain a JSON body with the
    `upgrade` parameter: true to upgrade user and false to downgrade.
    """
    if request.method == "POST":
        # Get response data
        data = json.loads(request.body)
        user = User.objects.get(username=data.get("username", None))
        upgrade = data.get("upgrade", False)

        # Get permissions
        content_type = ContentType.objects.get_for_model(SuperUserPermissions)
        permissions = Permission.objects.filter(content_type=content_type)

        # Assign permissions to user
        try:
            for permission in permissions:
                if upgrade:
                    logger.info(
                        f"[User: {user.username}] Added permission: {permission.name}"
                    )
                    user.user_permissions.add(permission)
                else:
                    user.user_permissions.remove(permission)
                    logger.info(
                        f"[User: {user.username}] Removed permission: {permission.name}"
                    )
            return JsonResponse({"message": f"Modifications validées"})

        # Return error message if something goes wrong with permissions assignment
        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {"message": "Erreur serveur : échec lors de la modification"},
                status=500,
            )
    else:
        raise Http404()
