#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import tempfile
from pathlib import Path

import pandas as pd
from django.http import HttpResponse, Http404, JsonResponse

from wattelse.api.rag_orchestrator.rag_client import RAGAPIError
from .models import Chat
from .views import RAG_API

# Feedback identifiers in the database
GREAT = "great"
OK = "ok"
MISSING = "missing_info"
WRONG = "wrong"

FAQ_FILE_PATTERN = "_FAQ.xlsx"

def insert_feedback(request, short: bool):
    """
    Function that collects feedback sent from the user interface about the last
    interaction (matching between the user question and the bot answer).
    """
    if request.method == "POST":
        # Get user info
        user = request.user

        # Get feedback info from the request
        feedback = request.POST.get("feedback", None)
        if short:
            feedback = to_short_feedback(feedback)
        user_message = request.POST.get("user_message", None)
        bot_message = request.POST.get("bot_message", None)

        # Try to find the matching Chat object based on user, message, and response
        try:
            chat_message = (Chat.objects.filter(user=user, message=user_message, response=bot_message)
                            .order_by('-timestamp').first()) # in case multiple chat messages match, take the newest
            if short:
                chat_message.short_feedback = feedback
            else:
                if feedback:
                    chat_message.long_feedback = feedback
                    update_FAQ(chat_message)
            chat_message.save()
            return HttpResponse(status=200)
        except Chat.DoesNotExist:
            # Handle case where message not found (log error, display message, etc.)
            error_message = f"Chat message not found for user: {user}, message: {user_message}, response: {bot_message}"
            return JsonResponse({"error_message": error_message}, status=500)
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


def to_short_feedback(feedback: str) -> str:
    """
    Function that converts the identifiers of the short feedback sent from the user interface into
    another identifier stored in the database.
    """
    if feedback == "rating-great":
        return GREAT
    if feedback == "rating-ok":
        return OK
    if feedback == "rating-missing":
        return MISSING
    if feedback == "rating-wrong":
        return WRONG
    return feedback

def update_FAQ(chat_message: Chat):
    """Update a FAQ file with the long feedback"""
    # Retrieve current FAQ file if it exists
    with tempfile.TemporaryDirectory() as temp_dir:
        # Construct the full path for the downloaded file within the temporary directory
        temp_file_path = Path(temp_dir) / FAQ_FILE_PATTERN

        try:
            RAG_API.download_to_dir(chat_message.group_id, FAQ_FILE_PATTERN, temp_file_path)
            df = pd.read_excel(temp_file_path)
        except RAGAPIError:
            # No file available, create new one
            df = pd.DataFrame(columns=["question", "answer"])

        # Update the feedback data and file
        df.loc[len(df)] =  {"question": chat_message.message, "answer": chat_message.long_feedback}
        df.to_excel(temp_file_path, index=False)

        # Uploads the file and updates its embeddings in the collection
        RAG_API.remove_documents(chat_message.group_id, [FAQ_FILE_PATTERN])
        RAG_API.upload_files(chat_message.group_id, [temp_file_path])