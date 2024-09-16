#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404, StreamingHttpResponse

from django.contrib.auth.models import User, Group
from django.contrib import auth
from .forms import  RegistrationForm, LoginForm
from django.contrib.auth import authenticate, login as auth_login
from .models import Chat, Update

from loguru import logger

import pytz
import mammoth
import csv

import io
import uuid
import json
import socket
import tempfile
from datetime import datetime

from pathlib import Path
from uuid import UUID
from xlsx2html import xlsx2html

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient, RAGAPIError 
from wattelse.chatbot.backend import DATA_DIR

# Create your views here.

from .utils import (
    get_user_group_id,
    get_group_usernames_list,
    new_user_created,
    get_conversation_history,
    streaming_generator,
    insert_feedback,
    RAG_API
)

def main_page(request):
    """
    Main function for chatbot interface.
    Render chatbot.html webpage with associated context.
    """
    # If user is not authenticated, redirect to login
    if not request.user.is_authenticated: #see this on django documentation
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
    can_download_updates = request.user.has_perm("chatbot.can_download_updates")

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
        request, "chatbot/main_chatbot.html",
        {
            "conversation_id": conversation_id,
            "available_docs": available_docs,
            "can_upload_documents": can_upload_documents,
            "can_remove_documents": can_remove_documents,
            "can_manage_users": can_manage_users,
            "can_download_updates": can_download_updates,
            "user_group": user_group_id,
            "group_usernames_list": group_usernames_list,
            "admin_group_selection": admin_group_selection,
        }
    )
    

def login(request):
    """Main function for login page.
    If request method is GET: render login.html
    If request method is POST: log the user in and redirect to chatbot.html
    """
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                # Check if the user belongs to a group
                user_group_id = get_user_group_id(user)
                if user_group_id is None:
                    error_message = "You do not belong to any group."
                    return render(request, "chatbot/login.html", {"form": form, "error_message": error_message})
                else:
                    auth_login(request, user)
                    logger.info(f"[User: {user.username}] logged in")
                    RAG_API.create_session(user_group_id)
                    return redirect("/")
            else:
                error_message = "Invalide username or password."
                return render(request, "chatbot/login.html", {"form": form, "error_message": error_message})
    else:
        form = LoginForm()

    return render(request, "chatbot/login.html", {"form": form})



def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return new_user_created(request, username=form.cleaned_data['username'])
        else:
            return render(request, "chatbot/register.html", {"form": form, "error_message": form.errors})
    else:
        form = RegistrationForm()
    return render(request, "chatbot/register.html", {"form": form})


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
            error_message = "Please raise a question"
            return JsonResponse({"error_message": error_message}, status=500)
        logger.info(f"[User: {request.user.username}] Query: {message}")

        # Select documents for RAG
        selected_docs = request.POST.get("selected_docs", None)
        selected_docs = json.loads(selected_docs)
        logger.info(f"[User: {request.user.username}] Selected docs: {selected_docs}")
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
        question_ts_s = request.POST.get("question_timestamp", "")
        question_ts = datetime.strptime(question_ts_s, "%Y-%m-%dT%H:%M:%S.%fZ") if question_ts_s else None
        # Convert to non-naive timedate (required for Django)
        question_ts = pytz.utc.localize(question_ts)
        chat = Chat(
            user=request.user,
            group_id=get_user_group_id(request.user),
            conversation_id=request.POST.get("conversation_id", ""),
            message=request.POST.get("message", ""),
            response=request.POST.get("answer", ""),
            question_timestamp=question_ts,
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

def suggest_update(request):
    """
    Function that collects short feedback sent from the user interface about the last
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

def update_extract(request):
    if request.method == "POST":
        user_group_id = get_user_group_id(request.user)
        relevant_unique_ids = request.POST.get("unique_ids", None)
        wrong_answer=request.POST.get("wrong_answer", "")
        correction = request.POST.get("correction", "")
        
        unique_ids = json.loads(relevant_unique_ids.replace('\\', '\\\\'))

        
        
        # Query update RAG and use LLM to generate the new extract
        try:
            response = RAG_API.update_extract(
                user_group_id,
                unique_ids,
                wrong_answer,
                correction
            )

            return JsonResponse(response, status=200)
        except RAGAPIError as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse({"error_message": f"Error during the extract update: {e}"}, status=500)
    else:
        raise Http404()

def save_modification(request):
    """Function called to save the modifications in the DB once the update is finished."""
    if request.method == "POST":
        # Save query and response in DB
        update_ts_s = request.POST.get("update_timestamp", "")
        update_ts = datetime.strptime(update_ts_s, "%Y-%m-%dT%H:%M:%S.%fZ") if update_ts_s else None
        # Convert to non-naive timedate (required for Django)
        update_ts = pytz.utc.localize(update_ts)
        chat_id = request.POST.get("chat_id", "")
        chat = Chat.objects.filter(pk = chat_id).first()
        
        update = Update(
            extract_id = request.POST.get("extract_id", ""),
            previous_version= request.POST.get("previous_version", ""),
            updated_extract = request.POST.get("updated_extract", ""),
            document_name = request.POST.get("document_name", ""),
            update_timestamp = update_ts,
            user = request.user,
            group_id=get_user_group_id(request.user),
            chat = chat
        )
        update.save()
        return HttpResponse(status=200)
    else:
        raise Http404()
    
    
def get_chat_id(request):
    if request.method == "POST":
        user_question = request.POST.get("user_question", "")
        bot_answer = request.POST.get("bot_answer", "")
        chat_id = Chat.objects.filter(
            user=request.user, 
            message=user_question, 
            response=bot_answer).order_by('-question_timestamp').first().id
        
        response = {"chat_id": chat_id}
        return JsonResponse(response, status=200)
        
    
    
        

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
            error_message = f"Username {new_username} was not found"
            return JsonResponse({"error_message": error_message}, status=500)

        # If new_user already in a group then return error status code
        if get_user_group_id(new_user) is not None:
            logger.error(
                f"[User: {request.user.username}] User with username {new_username} already belongs to a group")
            error_message = f"User {new_username} already belongs to a group"
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
            error_message = f"Username {username_to_remove} was not found"
            return JsonResponse({"error_message": error_message}, status=500)

        # Send an error if a user tries to remove himself
        if request.user.username == username_to_remove:
            error_message = f"You cannot remove yourself from the group"
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
    
def download_updates(request):
    if request.method == 'POST':
        selected_file = request.POST.get('selected_file', '[]')
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')
        
        logger.info(f'Received POST request with start_date: "{start_date}", end_date: "{end_date}", file_name: "{selected_file}"')

        
             # Debugging: print retrieved values
        print(f"Selected file: {selected_file}")
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        
        # Parse JSON and handle dates here
        
        logger.debug(f"[User: {request.user.username}] File selected for update track: {selected_file}")
        if not selected_file:
            logger.warning(f"[User: {request.user.username}] No file selected for update tracking received, action ignored")
            return JsonResponse({"warning_message": "No file downloaded"}, status=202)
        else:
            # Initialize the query set
            updates = Update.objects.all()
            
            start_date = datetime.strptime(start_date, '%d-%m-%Y')
            end_date = datetime.strptime(end_date, '%d-%m-%Y')

            # Filter updates based on the provided date range and file name
            updates = Update.objects.filter(
                update_timestamp__range=[start_date, end_date],
                document_name=selected_file
            )

          
            # Check if there are any updates
            if updates.count() == 0:
                # Return a response indicating no updates found
                return JsonResponse({"error_message":'No updates found for the specified criteria.'},
                                        status=202)
            else:

                # Create CSV response
                # Convert dates to string format for the filename
                start_date_str = start_date.strftime('%d-%m-%Y')
                end_date_str = end_date.strftime('%d-%m-%Y')
                selected_file_name = selected_file.split(".")[0]
                
                response = HttpResponse(content_type='text/csv; charset=utf-8')
                filename = '{}_updates_{}_{}.csv'.format(selected_file_name,start_date_str, end_date_str)
                print("Selected file name: '{}'".format(selected_file_name))

                response['Content-Disposition'] = 'attachment; filename={}'.format(filename)
                # Debugging: Print headers to ensure correct assignment
                print("Content-Disposition header: '{}'".format(response['Content-Disposition']))
                response.write(u'\ufeff'.encode('utf8'))


                writer = csv.writer(response, delimiter=';')

               # Write CSV headers
                field_names = [field.name for field in Update._meta.fields]
                writer.writerow(field_names)

                # Write data rows
                for update in updates:
                    writer.writerow([getattr(update, field) for field in field_names])

                return response
