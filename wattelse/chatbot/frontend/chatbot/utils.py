#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import uuid
import tempfile
import pandas as pd

from pathlib import Path
from typing import List, Dict
from loguru import logger

from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Group
from django.http import HttpResponse, Http404, JsonResponse

from .models import Chat

from wattelse.api.rag_orchestrator.rag_client import RAGOrchestratorClient, RAGAPIError

# RAG API
RAG_API = RAGOrchestratorClient()

# Separator to split json object from streaming chunks
SPECIAL_SEPARATOR = "¤¤¤¤¤"

# Feedback identifiers in the database
GREAT = "great"
OK = "ok"
MISSING = "missing_info"
WRONG = "wrong"

# Long feedback FAQ file
FAQ_FILE_PATTERN = "_FAQ.xlsx"


def get_user_group_id(user: User) -> str:
    """
    Given a user, return the id of the group it belongs to.
    If user doesn't belong to a group, return None.

    A user should belong to only 1 group.
    If it belongs to more than 1 group, return the first group.
    """
    group_list = user.groups.all()
    logger.trace(f"Group list for user {user.get_username()} : {group_list}")
    if len(group_list) == 0:
        return None
    else:
        return group_list[0].name


def is_superuser(user: User) -> bool:
    """
    Check if user is a superuser, i.e. can
    add/remove docs and manage users.
    """

    return (
        user.has_perm("chatbot.can_upload_documents")
        and user.has_perm("chatbot.can_remove_documents")
        and user.has_perm("chatbot.can_manage_users")
    )


def get_group_usernames_list(group_id: str) -> dict[str, bool]:
    """
    Returns a dictionnary with usernames as keys and
    whether they are superuser as values.
    """
    group = Group.objects.get(name=group_id)
    users_list = User.objects.filter(groups=group)
    users_dict = {user.username: is_superuser(user) for user in users_list}
    # Sort dictionnary so superusers are first and alphabetically sorted
    users_dict = dict(
        sorted(users_dict.items(), key=lambda item: (not item[1], item[0]))
    )
    return users_dict


def get_conversation_history(
    user: User, conversation_id: uuid.UUID
) -> List[Dict[str, str]] | None:
    """
    Return chat history following OpenAI API format :
    [
        {"role": "user", "content": "message1"},
        {"role": "assistant", "content": "message2"},
        {"role": "user", "content": "message3"},
        {...}
    ]
    If no history is found return None.
    """
    history = []
    history_query = Chat.objects.filter(user=user, conversation_id=conversation_id)
    for chat in history_query:
        history.append({"role": "user", "content": chat.message})
        history.append({"role": "assistant", "content": chat.response})
    return None if len(history) == 0 else history


def new_user_created(request, username=None):
    """
    Webpage rendered when a new user is created.
    It warns the user that no group is associated yet and need to contact an administrator.
    """
    if username is None:
        return redirect("/login")
    else:
        return render(request, "chatbot/new_user_created.html", {"username": username})


def streaming_generator(data_stream):
    """Generator to decode the chunks received from RAGOrchestratorClient"""
    with data_stream as r:
        for chunk in r.iter_content(chunk_size=None):
            # Add delimiter `\n` at the end because if streaming is too fast,
            # frontend can receive multiple chunks in one pass, so we need to split them.
            yield chunk.decode("utf-8") + SPECIAL_SEPARATOR


def streaming_generator_llm(data_stream):
    """Generator to decode the chunks received from RAGOrchestratorClient"""
    for chunk in data_stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            yield token


def insert_feedback(request, short: bool):
    """
    Function that collects feedback sent from the user interface about the last
    interaction (matching between the user question and the bot answer).
    """
    if request.method == "POST":
        # Get request data
        data = json.loads(request.body)

        # Get user info
        user = request.user

        # Get feedback info from the request
        feedback = data.get("feedback", None)
        if short:
            feedback = to_short_feedback(feedback)
        user_message = data.get("user_message", None)
        bot_message = data.get("bot_message", None)

        # Find the matching Chat object based on user, message, and response
        if Chat.objects.filter(
            user=user, message=user_message, response=bot_message
        ).exists():
            chat_message = (
                Chat.objects.filter(
                    user=user, message=user_message, response=bot_message
                )
                .order_by("-question_timestamp")
                .first()
            )  # in case multiple chat messages match, take the newest
        else:
            return JsonResponse({"message": "Conversation non trouvée"}, status=500)

        # Handle feedback and save it in the database
        if short:
            chat_message.short_feedback = feedback
        else:
            if feedback:
                chat_message.long_feedback = feedback
                update_FAQ(chat_message)
        try:
            chat_message.save()
            logger.info(
                f'[User: {request.user.username}] Feedback: "{feedback}" for question "{chat_message.message}"'
            )
            return JsonResponse({"message": "Feedback enregistré"})
        except Exception as e:
            logger.error(e)
            return JsonResponse(
                {
                    "message": "Erreur serveur : échec lors de l'enregistrement du feedback"
                },
                status=500,
            )
    else:
        raise Http404()


def update_FAQ(chat_message: Chat):
    """Update a FAQ file with the long feedback"""
    # Retrieve current FAQ file if it exists
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the downloaded file within the temporary directory
        temp_file_path = Path(temp_dir) / FAQ_FILE_PATTERN

        try:
            RAG_API.download_to_dir(
                chat_message.group_id, FAQ_FILE_PATTERN, temp_file_path
            )
            df = pd.read_excel(temp_file_path)
        except RAGAPIError:
            # No file available, create new one
            df = pd.DataFrame(columns=["question", "answer"])

        # Update the feedback data and file
        df.loc[len(df)] = {
            "question": chat_message.message,
            "answer": chat_message.long_feedback,
        }
        df.to_excel(temp_file_path, index=False)

        # Uploads the file and updates its embeddings in the collection
        RAG_API.remove_documents(chat_message.group_id, [FAQ_FILE_PATTERN])
        RAG_API.upload_files(chat_message.group_id, [temp_file_path])


def to_short_feedback(feedback: str) -> str:
    """
    Function that converts the identifiers of the short feedback sent from the user interface into
    another identifier stored in the database.
    """
    if "rating-great" in feedback:
        return GREAT
    if "rating-ok" in feedback:
        return OK
    if "rating-missing" in feedback:
        return MISSING
    if "rating-wrong" in feedback:
        return WRONG
    return feedback
