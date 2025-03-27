#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import json
import os
import tempfile
import datetime
import pandas as pd

from pathlib import Path

from django.contrib.auth.models import User, Group

from .models import RAGConversation, RAGMessage
from accounts.models import GroupProfile, UserProfile

from wattelse.api.rag_orchestrator.client import RAGOrchestratorClient, RAGAPIError

# RAG API
RAG_API = RAGOrchestratorClient(
    client_id="wattelse",
    client_secret=os.getenv("WATTELSE_CLIENT_SECRET", None),
)

# Feedback identifiers in the database
# GREAT = "great"
# OK = "ok"
# MISSING = "missing_info"
# WRONG = "wrong"

# FAQ file
FAQ_FILE_PATTERN = "_FAQ.xlsx"

LLM_MAPPING = {
    "wattelse-gpt35": "gpt-3.5",
    "wattelse-gpt4": "gpt-4",
    "wattelse-gpt4o-mini-sweden": "gpt-4o-mini",
    "wattelse-gpt4o-sweden": "gpt-4o",
    "wattelse-gpt4o-mini-sweden-dev": "gpt-4o-mini-dev",
    "wattelse-gpt4o-sweden-dev": "gpt-4o-dev",
    "wattelse-mistral-Large-2411": "Mistral-large",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "wattelse-Phi-4": "phi-4",
}


def get_user_active_group(user: User) -> Group | None:
    """
    Given a user, return its active group.
    If user doesn't belong to a group, return None.
    """
    active_group = UserProfile.objects.get(user=user).active_group
    if active_group:
        return active_group
    else:
        return None


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
        group_ids_list = [group.name for group in user_groups]
        group_ids_list.sort(key=str.lower)
        return group_ids_list
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
    Returns a dict of users having `group_id` as active group in the following format:
    {
        "username": is_superuser(),
        ...
    }
    with is_superuser() being True if the user is admin of its group.
    """
    # Get group object
    group = Group.objects.get(name=group_id)

    # Filter users having group_id as active group
    users_list = UserProfile.objects.filter(active_group=group).exclude(
        user__is_superuser=True
    )
    users_dict = {user.user.username: is_superuser(user.user) for user in users_list}

    # Sort dictionnary so superusers are first and alphabetically sorted
    users_dict = dict(
        sorted(users_dict.items(), key=lambda item: (not item[1], item[0]))
    )
    return users_dict


def conversation_messages(
    conversation: RAGConversation,
    n: int = 0,
) -> list[dict[str, str]] | None:
    """
    Return chat history following OpenAI API format + include message_id:
    [
        {"role": "user", "content": "message1", "id": "message_id"},
        {"role": "assistant", "content": "message2", "id": "message_id"},
        {"role": "user", "content": "message3", "id": "message_id"},
        {...}
    ]
    If no history is found return None.
    Use `n` parameter to only return the last `n` messages.
    Default `n` value is 0 to return all messages.
    """
    messages = conversation.messages.order_by("created_at")

    if n > 0:
        messages = messages[:n]

    history = [
        {"role": message.role, "content": message.content, "id": str(message.id)}
        for message in messages
    ]

    return history if history else None


def get_user_conversations(user: User) -> dict[str, list]:
    """
    Returns a dictionnary containing the user's conversations in the following format :
    {
        "today": [{"id": conversation_id, "title": conversation_title}, ...],
        "last_week": [{"id": conversation_id, "title": conversation_title}, ...],
        "archive": [{"id": conversation_id, "title": conversation_title}, ...],
    }
    """
    # Get conversation ids ordered by updated_at
    user_conversations = RAGConversation.objects.filter(user=user).order_by(
        "-updated_at"
    )

    # Sort conversations into date categories for better user expericence
    sorted_conversations = {"today": [], "last_week": [], "others": []}

    today = datetime.datetime.now()

    for conversation in user_conversations:
        # If created today
        if conversation.updated_at.date() == today.date():
            sorted_conversations["today"].append(
                {"id": conversation.id, "title": conversation.title}
            )

        # If created in 7 last days
        elif conversation.updated_at.date() >= today.date() - datetime.timedelta(
            days=7
        ):
            sorted_conversations["last_week"].append(
                {"id": conversation.id, "title": conversation.title}
            )
        # If created longer ago
        else:
            sorted_conversations["others"].append(
                {"id": conversation.id, "title": conversation.title}
            )

    return sorted_conversations


def get_group_system_prompt(group: Group) -> str:
    """
    Gets the group system prompt of a group.
    Returns empty string if not found.
    """
    group_system_prompt = GroupProfile.objects.filter(group=group).first()
    return group_system_prompt.system_prompt if group_system_prompt else ""


def get_group_rag_config_name(group: Group) -> str:
    """
    Gets the group RAG config file path of a group.
    Returns None if not found.
    """
    group_profile, _ = GroupProfile.objects.get_or_create(group=group)
    return group_profile.rag_config


def streaming_generator_rag(data_stream):
    """
    Generator to decode the chunks received from RAGOrchestratorClient.
    First chunk contains relevant extracts, remaining chunks are plain text.
    """
    first_chunk = True
    with data_stream as r:
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            # First chunk is a JSON object for relevant extracts
            if first_chunk:
                yield chunk
                first_chunk = False
            # Remaining chunks are plain text
            else:
                chunk = json.loads(chunk)
                yield chunk["answer"]


def update_FAQ(user_message: RAGMessage, faq_answer: str):
    """
    Update a FAQ file with the faq answer provided by a user.
    """
    # Retrieve current FAQ file if it exists
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the downloaded file within the temporary directory
        temp_file_path = Path(temp_dir) / FAQ_FILE_PATTERN

        try:
            RAG_API.download_to_dir(
                user_message.group.name, FAQ_FILE_PATTERN, temp_file_path
            )
            df = pd.read_excel(temp_file_path)
        except RAGAPIError:
            # No file available, create new one
            df = pd.DataFrame(columns=["question", "answer"])

        # Update the feedback data and file
        df.loc[len(df)] = {
            "question": user_message.content,
            "answer": faq_answer,
        }
        df.to_excel(temp_file_path, index=False)

        # Uploads the file and updates its embeddings in the collection
        RAG_API.remove_documents(user_message.group.name, [FAQ_FILE_PATTERN])
        RAG_API.upload_files(user_message.group.name, [temp_file_path])


# def to_short_feedback(feedback: str) -> str:
#     """
#     Function that converts the identifiers of the short feedback sent from the user interface into
#     another identifier stored in the database.
#     """
#     if "rating-great" in feedback:
#         return GREAT
#     if "rating-ok" in feedback:
#         return OK
#     if "rating-missing" in feedback:
#         return MISSING
#     if "rating-wrong" in feedback:
#         return WRONG
#     return feedback
