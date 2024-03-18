import json
import tempfile
from typing import Dict, Tuple

from django.core.files.uploadedfile import InMemoryUploadedFile
from django.forms import forms
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404

from django.contrib import auth
from django.contrib.auth.models import User
from loguru import logger
from pathlib import Path

from .models import Chat

from django.utils import timezone

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient
from wattelse.chatbot import DATA_DIR

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
    if not rag_client:
        # session expired for some reason
        return redirect("/login")

    # Get RAG session ID
    session_id = rag_client.session_id

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
                extract["metadata"]["url"] = f'pdf_viewer/{extract["metadata"]["file_name"]}#page={extract["metadata"].get("page", "0")+1}'

        # Save query and response in DB
        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({"message": message, "answer": answer, "relevant_extracts": relevant_extracts})
    else:
        return render(request, "chatbot/chatbot.html",
                      {"chats": chats, "session_id": session_id, "available_docs": available_docs})


def pdf_viewer(request, pdf_name: str):
    """
    Main function to render a PDF file. The url to access this function should be :
    pdf_viewer/pdf_file_name.pdf
    It will render the pdf file if the user belongs to the right group.
    """
    #TODO: manage more file type
    pdf_path = DATA_DIR / request.user.groups.all()[0].name / pdf_name
    if pdf_path.exists():
        with open(pdf_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='application/pdf')
            response['Content-Disposition'] = 'inline; filename="{}.pdf"'.format(pdf_name)
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
        # If user exists, login and redirect to chatbot
        if user is not None:
            auth.login(request, user)
            rag_dict[user.get_username()] = RAGOrchestratorClient(user.get_username())
            return redirect("/")
        # Else return error
        else:
            error_message = "Invalid username or password"
            return render(request, "chatbot/login.html", {"error_message": error_message})
    else:
        return render(request, "chatbot/login.html")


def register(request):
    """Main function for register page.
    If request method is GET : render register.html
    If request method is POST : create a new user, login and redirect to chatbot
    """
    if request.method == "POST":
        username = request.POST.get("username")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        # Check both password are the same
        if password1 == password2:
            try:
                user = User.objects.create_user(username, password=password1)
                user.save()
                auth.login(request, user)
                rag_dict[username] = RAGOrchestratorClient(username)
                return redirect("/")
            except:
                error_message = "Error creating account"
                return render(request, "chatbot/register.html", {"error_message": error_message})
        else:
            error_message = "Password dont match"
            return render(request, "chatbot/register.html", {"error_message": error_message})
    return render(request, "chatbot/register.html")


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
