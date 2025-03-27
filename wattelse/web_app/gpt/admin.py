from django.contrib import admin
from .models import Conversation, Message


class ConversationAdmin(admin.ModelAdmin):
    list_display = ("user", "title", "created_at", "updated_at")


class MessageAdmin(admin.ModelAdmin):
    list_display = ("content", "role", "created_at", "rating")


admin.site.register(Conversation, ConversationAdmin)
admin.site.register(Message, MessageAdmin)
