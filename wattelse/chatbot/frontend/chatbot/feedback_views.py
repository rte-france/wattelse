#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
from django.http import HttpResponse, Http404, JsonResponse

from .models import Chat

# Feedback identifiers in the database
GREAT = "great"
OK = "ok"
MISSING = "missing_info"
WRONG = "wrong"


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
            chat_message = Chat.objects.get(user=user, message=user_message, response=bot_message)
            if short:
                chat_message.short_feedback = feedback
            else:
                chat_message.long_feedback = feedback
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
