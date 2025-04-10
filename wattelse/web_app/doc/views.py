#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

import io
import uuid
import json
import tempfile
import mammoth
from loguru import logger
from pathlib import Path
from xlsx2html import xlsx2html

from django.urls import reverse

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, Http404, StreamingHttpResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_GET, require_POST
from django.contrib.auth.models import User, Group, Permission
from django.contrib.contenttypes.models import ContentType


from wattelse.api.rag_orchestrator.client import RAGAPIError
from wattelse.rag_backend import DATA_DIR

from .models import RAGConversation, RAGMessage
from accounts.models import GroupProfile, SuperUserPermissions, UserProfile

from .utils import (
    conversation_messages,
    get_user_active_group,
    get_user_groups,
    get_group_usernames_list,
    get_group_system_prompt,
    get_group_rag_config_name,
    streaming_generator_rag,
    is_superuser,
    RAG_API,
    get_user_conversations,
    can_edit_group_system_prompt,
    LLM_MAPPING,
    update_FAQ,
)

# Max number of messages to take
MAX_MESSAGES = 12


@login_required
@require_GET
def main_page(request):
    """
    WattElse DOC page.
    Render DOC main page with associated context.
    """
    # Get user active group
    user_group = get_user_active_group(request.user)

    # If user doesn't belong to a group, return error
    if user_group is None:
        return render(request, "doc/no_group.html")

    # Create RAG session for group
    rag_config_name = get_group_rag_config_name(user_group)
    RAG_API.create_session(user_group, config=rag_config_name)

    # Get user group list for group change
    user_group_list = get_user_groups(request.user)

    # Get group system prompt
    group_system_prompt = get_group_system_prompt(user_group)

    # Get user conversations history
    conversations = get_user_conversations(request.user)

    # Get list of available documents
    try:
        available_docs = RAG_API.list_available_docs(user_group)
    except RAGAPIError:
        return redirect("/login")

    # Get user permissions
    can_upload_documents = request.user.has_perm("accounts.can_upload_documents")
    can_remove_documents = request.user.has_perm("accounts.can_remove_documents")
    can_manage_users = request.user.has_perm("accounts.can_manage_users")

    # Check if group is allowed to edit system prompt
    group_can_edit_system_prompt = can_edit_group_system_prompt(user_group)

    # Only admin can edit system prompt, other users have read access by default
    user_can_edit_system_prompt = is_superuser(request.user)

    # If can manage users, find usernames of its group
    if can_manage_users:
        group_usernames_dict = get_group_usernames_list(user_group)
    else:
        group_usernames_dict = None

    # Special case for admin
    if request.user.is_superuser:
        admin_group_selection = [group.name for group in Group.objects.filter()]
    else:
        admin_group_selection = None
    return render(
        request,
        "doc/doc.html",
        {
            "available_docs": available_docs,
            "can_upload_documents": can_upload_documents,
            "can_remove_documents": can_remove_documents,
            "can_manage_users": can_manage_users,
            "group_can_edit_system_prompt": group_can_edit_system_prompt,
            "user_can_edit_system_prompt": user_can_edit_system_prompt,
            "user_group": user_group.name,
            "user_group_list": user_group_list,
            "group_system_prompt": group_system_prompt,
            "group_usernames_dict": group_usernames_dict,
            "admin_group_selection": admin_group_selection,
            "conversations": conversations,
            "is_wattelse_doc": True,
            "llm_name": LLM_MAPPING[RAG_API.get_rag_llm_model(user_group)],
        },
    )


@login_required
@require_POST
def query_rag(request):
    """
    Main function for RAG calls.
    Saves user message to database and calls 'RAGOrchestratorAPI'
    with 'stream=True' to stream the response to frontend.
    First chunk contains 'relevant_extracts' data, other
    chunks contain streamed answer tokens.
    """
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")
    selected_docs = data.get("selected_docs", None)

    # Get user group_id
    user_active_group = get_user_active_group(request.user)

    # Get or create conversation
    conversation, _ = RAGConversation.objects.get_or_create(
        id=conversation_id,
        user=request.user,
        defaults={"title": content},
    )

    # Add user message to database
    message = RAGMessage(
        id=message_id,
        conversation=conversation,
        role="user",
        content=content,
        group=user_active_group,
    )
    message.save()

    # Get user chat history
    history = conversation_messages(conversation, n=MAX_MESSAGES)

    # Get group system prompt
    group_system_prompt = get_group_system_prompt(user_active_group)

    # Select documents for RAG
    if not selected_docs:
        selected_docs = []

    # Query RAG and stream response
    try:
        response = RAG_API.query_rag(
            user_active_group.name,
            content,
            history=history,
            group_system_prompt=group_system_prompt,
            selected_files=selected_docs,
            stream=True,
        )

        return StreamingHttpResponse(
            streaming_generator_rag(response),
            status=200,
            content_type="text/plain",
        )

    except RAGAPIError as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {"error_message": f"Erreur lors de la requête au RAG: {e}"}, status=500
        )


@login_required
@require_POST
def save_assistant_message(request):
    """
    Saves assistant message to database.
    This function is called after the streaming response from 'query_rag' has ended.
    """
    # Get request data
    data = json.loads(request.body)
    conversation_id = uuid.UUID(data.get("conversation_id"))
    message_id = uuid.UUID(data.get("message_id"))
    content = data.get("content")
    relevant_extracts = {
        i: extract for i, extract in enumerate(data.get("relevant_extracts", ""))
    }

    # Get user related data
    group = get_user_active_group(request.user)
    group_system_prompt = get_group_system_prompt(group)
    rag_config = get_group_rag_config_name(group)

    # Get conversation
    conversation = RAGConversation.objects.get(id=conversation_id, user=request.user)

    # Add assistant message to database
    message = RAGMessage(
        id=message_id,
        conversation=conversation,
        role="assistant",
        content=content,
        group=group,
        group_system_prompt=group_system_prompt,
        relevant_extracts=relevant_extracts,
        rag_config=rag_config,
    )
    message.save()

    return HttpResponse(status=200)


@login_required
@require_POST
def handle_vote(request):
    """
    Handles user vote for a message.
    Retrieve message based on its id and update its rating.
    """
    # Get request data
    data = json.loads(request.body)
    message_id = data.get("message_id")
    rating = data.get("rating")

    # Get associated message and update rating
    message = RAGMessage.objects.get(id=message_id)
    message.rating = rating
    message.save()

    return HttpResponse(status=200)


@login_required
@require_POST
def handle_FAQ(request):
    """
    Handles user FAQ answer for a message.
    Retrieve message based on its id and update its FAQ answer.
    Update FAQ file for the associated group.
    """
    # Get request data
    data = json.loads(request.body)
    user_message_id = data.get("user_message_id")
    assistant_message_id = data.get("assistant_message_id")
    faq_answer = data.get("faq_answer")

    # Get user message
    user_message = RAGMessage.objects.get(id=user_message_id)

    # Get assistant message and update faq answer
    assistant_message = RAGMessage.objects.get(id=assistant_message_id)
    assistant_message.faq_answer = faq_answer
    assistant_message.save()

    # Update group FAQ file
    update_FAQ(user_message, faq_answer)

    return HttpResponse(status=200)


@login_required
@require_POST
def upload(request):
    """
    Handles file upload for RAG.
    """
    # Get the uploaded file
    uploaded_file = request.FILES.get("file")

    if not uploaded_file:
        logger.warning(
            f"[User: {request.user.username}] No file to be uploaded, action ignored"
        )
        return JsonResponse({"error_message": "No file received!"}, status=500)
    else:
        user_group_id = get_user_active_group(request.user)
        logger.debug(
            f"[User: {request.user.username}] Received file: {uploaded_file.name}"
        )

        # Create a temporary directory
        # TODO: investigate in memory temp file, probably a better option
        with tempfile.TemporaryDirectory() as temp_dir:
            # Construct the full path for the uploaded file within the temporary directory
            temp_file_path = Path(temp_dir) / Path(uploaded_file.name)

            # Open the temporary file for writing
            with open(temp_file_path, "wb") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            try:
                # Use the temporary file path for upload
                RAG_API.upload_files(user_group_id, [temp_file_path])
            except RAGAPIError as e:
                logger.error(f"[User: {request.user.username}] {e}")
                return JsonResponse(
                    {
                        "error_message": f"Erreur de téléchargement de {uploaded_file.name}\n{e}"
                    },
                    status=500,
                )

        # Returns the list of updated available documents
        return JsonResponse(
            {"available_docs": RAG_API.list_available_docs(user_group_id)},
            status=200,
        )


@login_required
@require_POST
def delete(request):
    """
    Handles documents deletion for RAG.
    """
    # Get request data
    data = json.loads(request.body)
    selected_docs = data.get("selected_docs", None)

    # Select documents for removal
    logger.info(
        f"[User: {request.user.username}] Docs selected for removal: {selected_docs}"
    )
    if not selected_docs:
        logger.warning(
            f"[User: {request.user.username}] No docs selected for removal received, action ignored"
        )
        return JsonResponse({"message": "Aucun document sélectionné"}, status=500)
    else:
        user_group = get_user_active_group(request.user)
        try:
            RAG_API.remove_documents(user_group.name, selected_docs)
            return JsonResponse(
                {
                    "available_docs": RAG_API.list_available_docs(user_group.name),
                    "message": "Documents supprimés",
                },
                status=200,
            )
        except Exception as e:
            logger.error(f"[User: {request.user.username}] {e}")
            return JsonResponse(
                {
                    "message": f"Erreur serveur : échec lors de la suppression des documents"
                },
                status=500,
            )


@login_required
def file_viewer(request, file_name: str):
    """
    Main function to render a file. The url to access this function should be :
    file_viewer/file_name.suffix
    It will render the file if the user belongs to the right group and if the file format is supported.
    """
    # TODO: manage more file type
    # FIXME: to be rethought in case of distributed deployment (here we suppose RAG backend and Django on the same server...)
    file_path = DATA_DIR / get_user_active_group(request.user).name / file_name
    if file_path.exists():
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            content_type = "application/pdf"
            with open(file_path, "rb") as f:
                response = HttpResponse(f.read(), content_type=content_type)
                response["Content-Disposition"] = f'inline; filename="{file_path.name}"'
                return response
        elif suffix == ".docx":
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html = result.value
                return HttpResponse(html)
        elif suffix == ".xlsx":
            xlsx_file = open(file_path, "rb")
            out_file = io.StringIO()
            xlsx2html(xlsx_file, out_file, locale="en")
            out_file.seek(0)
            return HttpResponse(out_file.read())
    else:
        raise Http404()


@login_required
@require_POST
def add_user_to_group(request):
    """
    Function to add a new user to a group.
    The superuser send a POST request with data `new_username`.
    If `new_username` exists, add it to superuser group.
    If new_username doesn't belong to any group yet, set it as its active group.
    """
    # Get superuser group object
    superuser = request.user
    superuser_group = get_user_active_group(superuser)

    # Get new_username user object if it exists
    data = json.loads(request.body)
    new_username = data.get("new_username", None)
    if User.objects.filter(username=new_username).exists():
        new_user = User.objects.get(username=new_username)
    else:
        logger.error(
            f"[User: {request.user.username}] Username {new_username} not found"
        )
        return JsonResponse(
            {"message": f"{new_username} non trouvé"},
            status=500,
        )

    # Check if new user already has an active group
    new_user_already_in_group = get_user_active_group(new_user) is not None

    # Add new user to superuser group
    try:
        # User to group
        print("OK1")
        new_user.groups.add(superuser_group)

        # If user belong to no group, also set it as active group
        if not new_user_already_in_group:
            new_user_profile = UserProfile.objects.get(user=new_user)
            new_user_profile.active_group = superuser_group
            new_user_profile.save()
            print("OK1")
        return JsonResponse(
            {"message": f"{new_username} ajouté au groupe"},
        )
    except Exception as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {
                "message": "Erreur serveur : échec lors de l'ajout de l'utilisateur",
                "new_user_already_in_group": new_user_already_in_group,
            },
            status=500,
        )


@login_required
@require_POST
def remove_user_from_group(request):
    """
    Function to remove a user from a group.
    The superuser send a POST request with data `username_to_delete`.
    If `username_to_delete` exists, remove it from superuser group.
    Also remove its active group and set it to:
      - another group it belongs to
      - None if it does not belong to any other group
    """
    # Get superuser group object
    superuser = request.user
    superuser_group_id = get_user_active_group(superuser)
    superuser_group = Group.objects.get(name=superuser_group_id)

    # Get username_to_remove user object if it exists
    data = json.loads(request.body)
    username_to_remove = data.get("username_to_delete", None)
    if User.objects.filter(username=username_to_remove).exists():
        user_to_remove = User.objects.get(username=username_to_remove)
        user_profile_to_remove = UserProfile.objects.get(user=user_to_remove)
    else:
        return JsonResponse(
            {"message": f"{username_to_remove} non trouvé"},
            status=500,
        )

    # Send an error if a user tries to remove himself
    if request.user.username == username_to_remove:
        return JsonResponse({"message": "Impossible de vous supprimer"}, status=500)

    # Remove user_to_remove
    try:
        # Remove group for user groups
        user_to_remove.groups.remove(superuser_group)

        # Get first remaining user group
        user_remaining_group = user_to_remove.groups.first()

        # Set active group as first remaining group (None is does not exist)
        user_profile_to_remove.active_group = user_remaining_group
        user_profile_to_remove.save()
        logger.info(
            f"[User: {request.user.username}] Removed {username_to_remove} from group {superuser_group_id}"
        )
        return JsonResponse({"message": f"{username_to_remove} supprimé"})
    except Exception as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {"message": "Erreur serveur : échec lors de la suppression"}, status=500
        )


@login_required
@require_POST
def change_active_group(request):
    """
    Function to change active group using web interface.
    """
    # Get requested group
    new_active_group_id = request.POST.get("new_group")

    # If user is not admin, check if it is an allowed group
    if not request.user.is_superuser:
        allowed_groups = get_user_groups(request.user)
        if new_active_group_id not in allowed_groups:
            logger.error(
                f"[User: {request.user.username}] Tried to change to a non allowed group"
            )
            return JsonResponse(
                {"message": "Groupe non-autorisé"},
                status=500,
            )

    # If group is allowed, change active group
    try:
        # Retrieve the UserProfile associated with the User
        user_profile = UserProfile.objects.get(user=request.user)

        # Retrieve the new group
        new_active_group = Group.objects.get(name=new_active_group_id)

        # Update the active group
        user_profile.active_group = new_active_group
        user_profile.save()
        return redirect(reverse("doc:main_page"))
    except Exception as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {"message": "Erreur serveur : échec lors du changement de groupe"},
            status=500,
        )


# def get_questions_count_since_last_feedback(request):
#     """
#     Counts the number of entries without feedback since the user's last feedback
#     Returns:
#         int: The number of entries without feedback since the last feedback.
#     """
#     if request.method == "GET":
#         # Get request data
#         source_path = request.GET.get("source_path")

#         # Get database to use based on source path
#         ChatModel = get_chat_model(source_path)

#         try:
#             last_feedback = (
#                 ChatModel.objects.filter(
#                     ~Q(short_feedback__exact=""),
#                     user=request.user,
#                     short_feedback__isnull=False,
#                 )
#                 .order_by("-question_timestamp")
#                 .first()
#             )
#             if last_feedback:
#                 last_feedback_date = last_feedback.question_timestamp
#             else:
#                 return JsonResponse({"count": 9999})  # arbitrary big value
#         except ChatModel.DoesNotExist:
#             return JsonResponse({"count": 9999})  # arbitrary big value

#         count = ChatModel.objects.filter(
#             user=request.user, question_timestamp__gt=last_feedback_date
#         ).count()
#         return JsonResponse({"count": count})
#     else:
#         raise Http404()


@login_required
@require_POST
def manage_superuser_permission(request):
    """
    Function to allow a superuser to add or remove superuser permissions
    for users of its group. The request should contain a JSON body with the
    `upgrade` parameter: true to upgrade user and false to downgrade.
    """
    # Get response data
    data = json.loads(request.body)
    user = User.objects.get(username=data.get("username", None))
    upgrade = data.get("upgrade", False)

    # Get permissions
    content_type = ContentType.objects.get_for_model(SuperUserPermissions)
    permissions = Permission.objects.filter(content_type=content_type)

    # Assign permissions to user
    try:
        for permission in permissions:
            if upgrade:
                logger.info(
                    f"[User: {user.username}] Added permission: {permission.name}"
                )
                user.user_permissions.add(permission)
            else:
                user.user_permissions.remove(permission)
                logger.info(
                    f"[User: {user.username}] Removed permission: {permission.name}"
                )
        return JsonResponse({"message": f"Modifications validées"})

    # Return error message if something goes wrong with permissions assignment
    except Exception as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {"message": "Erreur serveur : échec lors de la modification"},
            status=500,
        )


@login_required
@require_GET
def get_conversation_messages(request):
    """
    Returns conversation messages associated with a conversation id in the OpenAI format.
    """
    # Get request data
    conversation_id = uuid.UUID(request.GET.get("id"))

    # Get conversation
    conversation = RAGConversation.objects.get(id=conversation_id, user=request.user)

    # Get conversation history
    messages = conversation_messages(conversation)

    return JsonResponse({"messages": messages}, status=200)


@login_required
@require_POST
def update_group_system_prompt(request):
    """
    Updates the group system prompt for a group.
    """
    # Get request data
    data = json.loads(request.body)
    new_system_prompt = data.get("group_system_prompt", None)

    # Get group id
    user = request.user
    group_id = get_user_active_group(user)
    group = Group.objects.get(name=group_id)

    # Set new group system prompt
    try:
        group_system_prompt = GroupProfile.objects.get_or_create(
            group=group, defaults={"system_prompt": ""}
        )[0]
        group_system_prompt.system_prompt = new_system_prompt
        group_system_prompt.save()
        logger.info(
            f"Group {group_id}: Succesfully changed system prompt to: {group_system_prompt.system_prompt}"
        )
        return JsonResponse({"message": f"System prompt sauvegardé"})
    except Exception as e:
        logger.error(f"[User: {request.user.username}] {e}")
        return JsonResponse(
            {
                "message": f"Erreur serveur : échec lors de la sauvegarde du system prompt"
            },
            status=500,
        )
