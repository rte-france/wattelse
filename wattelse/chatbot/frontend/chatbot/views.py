from typing import Dict

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from django.contrib import auth
from django.contrib.auth.models import User
from loguru import logger

from .models import Chat

from django.utils import timezone

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

# Mapping table between user login and RAG clients
rag_dict : Dict[str, RAGOrchestratorClient] = {}

class WattElseError(Exception):
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
    session_id = rag_dict[request.user.username].session_id

    # Get list of available documents
    available_docs = rag_dict[request.user.username].list_available_docs()

    # If request method is POST, call RAG API and return response to query
    # Else render template
    if request.method == "POST":
        message = request.POST.get("message", None)
        if not message:
            raise WattElseError("No user message received")
        logger.info(f"User: {request.user.username} - Query: {message}")

        # Select documents for RAG
        selected_docs = request.POST.get("selected_docs", None)
        if not selected_docs:
            logger.warning("No selected docs received, using all available docs")
            selected_docs = available_docs
        rag_dict[request.user.get_username()].select_documents_by_name(selected_docs)

        # Query RAG
        response = rag_dict[request.user.get_username()].query_rag(message)
        # separate text answer and relevant extracts
        answer = response["answer"]
        relevant_extracts = response["relevant_extracts"]

        # Update url in relevant_extracts to make it openable accessible from the web page
        # TODO à adapter avec la bonne URL!! (cf Guillaume)
        for extract in relevant_extracts:
            extract["metadata"]["url"] = f'http://{extract["metadata"]["file_name"]}#{extract["metadata"]["page"]}'

        # Save query and response in DB
        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({"message": message, "answer": answer, "relevant_extracts": relevant_extracts})
    else:
        return render(request, "chatbot/chatbot.html",
                      {"chats": chats, "session_id": session_id, "available_docs": available_docs})


def login(request):
    """Main function for login page.
    If request method is GET : render login.html
    If request method is POST : log the user in and redirect to chatbot.html
    """
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
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
        username = request.POST["username"]
        password1 = request.POST["password1"]
        password2 = request.POST["password2"]

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
