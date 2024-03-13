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
    if not request.user.is_authenticated:
        return redirect('/login')
    
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
        message = request.POST.get('message')

        response = RAG_DICT[request.user.get_username()].query_rag(message)

        chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
        chat.save()
        return JsonResponse({'message': message, 'response': response})
    return render(request, 'chatbot/chatbot.html', {'chats': chats})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            RAG_DICT[user.get_username()] = RAGOrchestratorClient(user.get_username())
            return redirect('/')
        else:
            error_message = 'Invalid username or password'
            return render(request, 'chatbot/login.html', {'error_message': error_message})
    else:
        return render(request, 'chatbot/login.html')

def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, password1)
                user.save()
                auth.login(request, user)
                print(username)
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
    auth.logout(request)
    return redirect('/login')

def reset(request):
    if request.method == 'POST':
        Chat.objects.filter(user=request.user).delete()
    return HttpResponse()
