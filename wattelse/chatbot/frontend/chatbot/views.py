import io
import json
import tempfile
from typing import Dict, Tuple

import mammoth
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404

from django.contrib import auth
from django.contrib.auth.models import User
from loguru import logger
from pathlib import Path

from django.utils import timezone
from xlsx2html import xlsx2html

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient
from wattelse.chatbot import DATA_DIR
from .models import Chat

# Mapping table between user login and RAG clients
rag_dict: Dict[str, RAGOrchestratorClient] = {}


class WattElseError(Exception):
    """Generic Error for the Django interface of the chatbot application"""
    pass


def chatbot(request):
    """Main function for chatbot interface.
    If request method is GET : render chatbot.html
    If request method is POST : make a call to RAGOrchestratorClient and return response as json
    """
    # If user is not authenticated, redirect to login
    if not request.user.is_authenticated:
        return redirect("/login")

    # Get user chat history
    chats = Chat.objects.filter(user=request.user)
    rag_client = rag_dict.get(request.user.get_username())
    if not is_active_session(rag_client):
        # session expired for some reason
        return redirect("/login")

    # Get list of available documents
    available_docs = rag_client.list_available_docs()

    # If request method is POST, call RAG API and return response to query
    # else render template
    if request.method == "POST":
        message = request.POST.get("message", None)
        if not message:
            raise WattElseError("No user message received")
        logger.info(f"User: {request.user.username} - Query: {message}")

        # Select documents for RAG
        selected_docs = request.POST.get("selected_docs", None)
        logger.debug(f"Selected docs: {selected_docs}")
        if not selected_docs:
            logger.warning("No selected docs received, using all available docs")
            selected_docs = []
        rag_client.select_documents_by_name(json.loads(selected_docs))

        # Query RAG
        response = rag_client.query_rag(message)
        # separate text answer and relevant extracts
        answer = response["answer"]
        relevant_extracts = response["relevant_extracts"]

        # Update url in relevant_extracts to make it openable accessible from the web page
        if relevant_extracts:
            for extract in relevant_extracts:
                page_number = int(extract["metadata"].get("page", "0")) + 1
                extract["metadata"][
                    "url"] = f'file_viewer/{extract["metadata"]["file_name"]}#page={page_number}'

        # Save query and response in DB
        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({"message": message, "answer": answer, "relevant_extracts": relevant_extracts})
    else:
        # Get user permissions
        can_upload_documents = request.user.has_perm("chatbot.can_upload_documents")
        can_remove_documents = request.user.has_perm("chatbot.can_remove_documents")
        can_add_users = request.user.has_perm("chatbot.can_add_users")
        return render(
            request, "chatbot/chatbot.html",
            {
                "chats": chats,
                "available_docs": available_docs,
                "can_upload_documents": can_upload_documents,
                "can_remove_documents": can_remove_documents,
                "can_add_users": can_add_users,
            }
            )


def file_viewer(request, file_name: str):
    """
    Main function to render a PDF file. The url to access this function should be :
    file_viewer/file_name.pdf
    It will render the file if the user belongs to the right group and if the file format is supported
    """
    # TODO: manage more file type
    file_path = DATA_DIR / request.user.groups.all()[0].name / file_name
    if file_path.exists():
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value  # The generated HTML
                messages = result.messages  # Any messages, such as warnings during conversion
                return HttpResponse(html)
        elif suffix == ".xlsx":
            xlsx_file = open(file_path, 'rb')
            out_file = io.StringIO()
            xlsx2html(xlsx_file, out_file, locale='en')
            out_file.seek(0)
            return HttpResponse(out_file.read())
        elif suffix == ".pdf":
            content_type = 'application/pdf'
            with open(file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type=content_type)
                response['Content-Disposition'] = f'inline; filename="{file_path.name}"'
                return response
    else:
        raise Http404()

def delete(request):
    """Main function for delete interface.
    If request method is POST : make a call to RAGOrchestratorClient to delete the specified documents
    """
    if request.method == "POST":
        # Select documents for removal
        selected_docs = request.POST.get("selected_docs", None)
        logger.debug(f"Docs selected for removal: {selected_docs}")
        if not selected_docs:
            logger.warning("No docs selected for removal received, action ignored")
            return JsonResponse({"status": "No document removed"})
        else:
            rag_client = rag_dict[request.user.get_username()]
            if not is_active_session(rag_client):
                # session expired for some reason
                return redirect("/login")
            rag_response = rag_client.remove_documents(json.loads(selected_docs))
            # Returns the list of updated available documents
            return JsonResponse({"available_docs": rag_client.list_available_docs()})


def upload(request):
    """Main function for delete interface.
    If request method is POST : make a call to RAGOrchestratorClient to upload the specified documents
    """
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.FILES.get('file')

        if not uploaded_file:
            logger.warning("No file to be uploaded, action ignored")
            return JsonResponse({"status": "No file received!"})
        else:
            logger.debug(f"Received file: {uploaded_file.name}")
            rag_client = rag_dict[request.user.get_username()]
            if not is_active_session(rag_client):
                # session expired for some reason
                return redirect("/login")

            # Create a temporary directory
            # TODO: investigate in memory temp file, probably a better option
            with tempfile.TemporaryDirectory() as temp_dir:
                # Construct the full path for the uploaded file within the temporary directory
                temp_file_path = Path(temp_dir) / Path(uploaded_file.name)

                # Open the temporary file for writing
                with open(temp_file_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

                # Use the temporary file path for upload
                rag_client.upload_files([temp_file_path])

            # Returns the list of updated available documents
            return JsonResponse({"available_docs": rag_client.list_available_docs()})


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
            group = get_user_group(user)
            if group is None:
                error_message = "Vous n'appartenez à aucun groupe."
                return render(request, "chatbot/login.html", {"error_message": error_message})
            else:
                auth.login(request, user)
                rag_dict[user.get_username()] = RAGOrchestratorClient(user.get_username(), group)
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

def new_user_created(request, username=None):
    """
    Webpage rendered when a new user is created.
    It warns the user that no group is associated yet and need to contact an administrator.
    """
    if username is None:
        return redirect("/login")
    else:
        return render(request, "chatbot/new_user_created.html", {"username": username})


def is_active_session(rag_client: RAGOrchestratorClient) -> bool:
    """Return true is the rag_client backend session ID is in the list of current sessions as reported by the backend
    (the backend performs some automatic cleaning, that's why we need to check...)"""
    if not rag_client:
        return False
    elif rag_client.session_id in rag_client.get_current_sessions():
        return True
    else: # clean Django sessions as well
        for k, v in rag_dict.items():
            if v == rag_client:
                rag_dict.pop(k)
                break
        return False


def logout(request):
    """Log a user out and redirect to login page"""
    auth.logout(request)
    return redirect("/login")


def reset(request):
    """Reset chat history from DB"""
    if request.method == "POST":
        Chat.objects.filter(user=request.user).delete()
    # Need to return an empty HttpResponse
    return HttpResponse()


def get_filename_parts(filename: str) -> Tuple[str, str]:
    """
  This function splits a filename into its prefix and suffix.

  Args:
      filename: The filename as a string.

  Returns:
      A tuple containing the prefix (without the dot) and the suffix (including the dot).
  """
    dot_index = filename.rfind(".")
    if dot_index == -1:
        prefix = filename
        suffix = ""
    else:
        prefix = filename[:dot_index]
        suffix = filename[dot_index:]
    return prefix, suffix

def get_user_group(user: User) -> str:
    """
    Given a user, return the name of the group it belongs to.
    If user doesn't belong to a group, return None.

    A user should belong to only 1 group.
    If it belongs to more than 1 group, return the first group.
    """
    group_list = user.groups.all()
    logger.info(f"Group list for user {user.get_username()} : {group_list}")
    if len(group_list)==0:
        return None
    else:
        return group_list[0].name