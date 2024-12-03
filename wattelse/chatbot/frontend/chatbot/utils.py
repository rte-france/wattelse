#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import uuid
import tempfile
import datetime
import pandas as pd

from pathlib import Path
from typing import List, Dict, Type

from loguru import logger

from django.db import models
from django.shortcuts import render, redirect
from django.contrib.auth.models import User, Group
from django.http import Http404, JsonResponse

from .models import Chat, GPTChat, GroupSystemPrompt, UserProfile

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


ChatModels = {
    "/doc/": Chat,
    "/gpt/": GPTChat,
}


def get_chat_model(source_path: str) -> Type[GPTChat | Chat]:
    """Return the database class associated to the current page (RAG vs secureGPT)
    allowed source_path : "/doc/" and "/gpt/"
    """
    # Check if source_path is valid
    if not source_path:
        raise Exception("Invalid source path")
    db_model = ChatModels.get(source_path, None)
    if not db_model:
        raise Exception(
            f"No database class model associated to the source path {source_path}"
        )
    return db_model


def get_user_active_group_id(user: User) -> str | None:
    """
    Given a user, return the id of its active group.
    If user doesn't belong to a group, return None.
    """
    active_group = UserProfile.objects.get(user=user).active_group
    if active_group:
        return active_group.name
    else:
        None


def get_user_groups(user: User) -> list[str] | None:
    """
    Given a user, return the list of group_id he can switch to.
    If user is admin, return all groups.
    """
    # If admin, return all groups else return user groups
    if user.is_superuser:
        user_groups = Group.objects.all()
    else:
        user_groups = user.groups.all()

    # If groups are found, return the list of group_id, else None
    if user_groups.exists():
        return [group.name for group in user_groups]
    else:
        return None


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


def can_edit_group_system_prompt(group_id: str) -> bool:
    """
    Check if a group can edit the group system prompt.
    """
    group = Group.objects.get(name=group_id)
    return group.permissions.filter(codename="can_edit_group_system_prompt").exists()


def get_group_usernames_list(group_id: str) -> dict[str, bool]:
    """
    Returns a dictionnary with usernames as keys and
    whether they are superuser as values.
    """
    group = Group.objects.get(name=group_id)
    users_list = User.objects.filter(groups=group).exclude(is_superuser=True)
    users_dict = {user.username: is_superuser(user) for user in users_list}
    # Sort dictionnary so superusers are first and alphabetically sorted
    users_dict = dict(
        sorted(users_dict.items(), key=lambda item: (not item[1], item[0]))
    )
    return users_dict


def get_conversation_history(
    user: User,
    conversation_id: uuid.UUID,
    ChatModel: models.Model,
    n: int = 0,
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
    Use `n` parameter to only return the last `n` messages.
    Default `n` value is 0 to return all messages.
    """
    history = []
    history_query = ChatModel.objects.filter(user=user, conversation_id=conversation_id)
    for chat in history_query:
        history.append({"role": "user", "content": chat.message})
        history.append({"role": "assistant", "content": chat.response})
    return None if len(history) == 0 else history[-n:]


def get_user_conversation_ids(
    user: User, ChatModel: models.Model, number: int = 100
) -> list[uuid.UUID]:
    """
    Return the list of the user conversation ids for its current active group.
    Use `number` to limit the maximum number of returned ids.
    """
    # Get user active group id
    group_id = get_user_active_group_id(user)

    # Get list of all conversation ids for the user, ordered by question timestamp
    conversation_ids = (
        ChatModel.objects.filter(user=user, group_id=group_id)
        .order_by("-question_timestamp")
        .values_list("conversation_id", flat=True)
    )

    # Remove duplicates while preserving order
    return list(dict.fromkeys(conversation_ids))[:number]


def get_conversation_first_message(
    conversation_id, ChatModel: models.Model
) -> tuple[str, datetime.datetime]:
    """
    Returns the first message of a conversation associated with its timestamp.
    Used as a title that will be displayed to the user in the conversation selection pannel.
    """
    first_message = ChatModel.objects.filter(conversation_id=conversation_id).first()
    return first_message.message, first_message.question_timestamp


def get_user_conversation_history(user: User, ChatModel: models.Model) -> dict:
    """
    Returns a dictionnary containing the user's conversations in the following format :
    {
        "today": [list of conversations today old],
        "last_week": [list of conversations one week old],
        "archive": [all other conversations],
    }
    """
    # Get conversation ids
    conversation_ids = get_user_conversation_ids(
        user,
        ChatModel=ChatModel,
    )

    # Sort conversations into date categories for better user expericence
    sorted_conversations = {"today": [], "last_week": [], "others": []}

    today = datetime.datetime.now()

    for conversation_id in conversation_ids:
        # Get conversation title and timestamp
        title, timestamp = get_conversation_first_message(conversation_id, ChatModel)

        # If created today
        if timestamp.date() == today.date():
            sorted_conversations["today"].append(
                {"id": conversation_id, "title": title}
            )

        # If created in 7 last days
        elif timestamp.date() >= (today - datetime.timedelta(days=7)).date():
            sorted_conversations["last_week"].append(
                {"id": conversation_id, "title": title}
            )

        # If created longer ago
        else:
            sorted_conversations["others"].append(
                {"id": conversation_id, "title": title}
            )

    return sorted_conversations


def get_group_system_prompt(group_id: str) -> str:
    """
    Gets the group system prompt of a group.
    Returns empty string if not found.
    """
    group_system_prompt = GroupSystemPrompt.objects.filter(group_id=group_id).first()
    return group_system_prompt.system_prompt if group_system_prompt else ""


def get_group_system_prompt(group_id: str) -> str:
    """
    Gets the group system prompt of a group.
    Returns empty string if not found.
    """
    group_system_prompt = GroupSystemPrompt.objects.filter(group_id=group_id).first()
    return group_system_prompt.system_prompt if group_system_prompt else ""


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
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


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

        # Get database to use based on source path
        chat_model = get_chat_model(data.get("source_path"))

        # Find the matching Chat object based on user, message, and response
        if chat_model.objects.filter(
            user=user, message=user_message, response=bot_message
        ).exists():
            chat_message = (
                chat_model.objects.filter(
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
