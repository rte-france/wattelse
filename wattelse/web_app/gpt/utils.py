import datetime
from django.contrib.auth.models import User

from .models import Conversation


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
    user_conversations = Conversation.objects.filter(user=user).order_by("-updated_at")

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


def conversation_messages(
    conversation: Conversation,
    n: int = 0,
    include_id: bool = True,
) -> list[dict[str, str]] | None:
    """
    Return chat history following OpenAI API format, optionally including message_id:
    [
        {"role": "user", "content": "message1", "id": "message_id"},
        {"role": "assistant", "content": "message2", "id": "message_id"},
        ...
    ]
    If no history is found return None.
    Use `n` parameter to only return the last `n` messages.
    Use `include_id` to control whether to include message IDs.
    """
    messages = conversation.messages.order_by("created_at")

    if n > 0:
        messages = messages[:n]

    history = [
        {
            "role": message.role,
            "content": message.content,
            **({"id": str(message.id)} if include_id else {}),
        }
        for message in messages
    ]

    return history if history else None


def streaming_generator_llm(data_stream):
    """
    Generator to decode the chunks received from LLM streaming response.
    """
    for chunk in data_stream:
        if len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
