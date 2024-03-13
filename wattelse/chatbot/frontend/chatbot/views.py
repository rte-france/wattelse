from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import openai

from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat

from django.utils import timezone

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient

RAG_DICT = {}

def chatbot(request):
    """Main function for chatbot interface.
    If request method is GET : render chatbot.html
    If request method is POST : make a call to RAGOrchestratorClient and return response as json
    """
    # If user is not authenticated, redirect to login
    if not request.user.is_authenticated:
        return redirect('/login')
    
    # Get user chat history
    chats = Chat.objects.filter(user=request.user)

    # If request method is POST, call RAG API and return response to query
    # Else render template
    if request.method == 'POST':
        message = request.POST.get('message')

        response = RAG_DICT[request.user.get_username()].query_rag(message)

        # Save query and response in DB
        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()

        return JsonResponse({'message': message, 'response': response})
    else:
        return render(request, 'chatbot/chatbot.html', {'chats': chats})


def login(request):
    """Main function for login page.
    If request method is GET : render login.html
    If request method is POST : log the user in and redirect to chatbot.html
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        # If user exists, login and redirect to chatbot
        if user is not None:
            auth.login(request, user)
            RAG_DICT[user.get_username()] = RAGOrchestratorClient(user.get_username())
            return redirect('/')
        # Else return error
        else:
            error_message = 'Invalid username or password'
            return render(request, 'chatbot/login.html', {'error_message': error_message})
    else:
        return render(request, 'chatbot/login.html')

def register(request):
    """Main function for register page.
    If request method is GET : render register.html
    If request method is POST : create a new user, login and redirect to chatbot
    """
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        # Check voth password are the same
        if password1 == password2:
            try:
                user = User.objects.create_user(username, password=password1)
                user.save()
                auth.login(request, user)
                RAG_DICT[username] = RAGOrchestratorClient(username)
                return redirect('/')
            except:
                error_message = 'Error creating account'
                return render(request, 'chatbot/register.html', {'error_message': error_message})
        else:
            error_message = 'Password dont match'
            return render(request, 'chatbot/register.html', {'error_message': error_message})
    return render(request, 'chatbot/register.html')

def logout(request):
    """Log a user out and redirect to login page"""
    auth.logout(request)
    return redirect('/login')

def reset(request):
    """Reset chat history from DB"""
    if request.method == 'POST':
        Chat.objects.filter(user=request.user).delete()
    # Need to return an empty HttpResponse
    return HttpResponse()
