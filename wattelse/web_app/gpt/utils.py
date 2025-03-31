import datetime
import uuid
from django.contrib.auth.models import User

from .models import Conversation, Message


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


def get_user_conversation_history(user: User) -> dict:
    """
    Returns a dictionnary containing the user's conversations in the following format :
    {
        "today": [list of conversations today old],
        "last_week": [list of conversations one week old],
        "archive": [all other conversations],
    }
    """
    # Get conversation ids
    user_conversations = Conversation.objects.filter(user=user)

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


def get_conversation_history(
    user: User,
    conversation_id: uuid.UUID,
    n: int = 0,
) -> list[dict[str, str]] | None:
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
    conversation = Conversation.objects.get(id=conversation_id, user=user)

    messages = conversation.messages.order_by("timestamp")

    if n > 0:
        messages = messages[:n]

    history = [
        {"role": message.role, "content": message.content} for message in messages
    ]

    return history if history else None


def streaming_generator_llm(data_stream):
    """Generator to decode the chunks received from RAGOrchestratorClient"""
    for chunk in data_stream:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
