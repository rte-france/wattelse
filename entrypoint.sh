#!/bin/bash

# Start rag_orchestrator in the background before running django migrations
echo "Starting rag_orchestrator..."
/app/wattelse/api/rag_orchestrator/start.sh &

# Wait for rag_orchestrator to be ready
echo "Waiting for rag_orchestrator to be ready..."
while ! nc -z localhost 1978; do
  sleep 1
done

# Run Django migrations
echo "Running Django migrations..."
python /app/wattelse/chatbot/frontend/manage.py migrate

# Load Secret Key from cfg file if not set in environment
CONFIG_FILE="/app/django_config/django_superuser.cfg"
if [ -z "$DJANGO_SECRET_KEY" ]; then
    if [ -f "$CONFIG_FILE" ]; then
        echo "Loading DJANGO_SECRET_KEY from $CONFIG_FILE"
        DJANGO_SECRET_KEY=$(grep "DJANGO_SECRET_KEY" "$CONFIG_FILE" | cut -d'=' -f2 | tr -d '"')
        export DJANGO_SECRET_KEY
    else
        echo "No DJANGO_SECRET_KEY found in environment variables or config file."
        exit 1  # Exit if no secret key is found, since it's necessary for Django to run
    fi
else
    echo "DJANGO_SECRET_KEY is already set in environment variables."
fi


# Check if superuser environment variables are set
echo "Checking if superuser environment variables are set"
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo "Superuser credentials found in environment variables."
    echo "from django.contrib.auth.models import User; User.objects.create_superuser('$DJANGO_SUPERUSER_USERNAME', '$DJANGO_SUPERUSER_EMAIL', '$DJANGO_SUPERUSER_PASSWORD')" | python /app/RAG/chatbot_frontend/manage.py shell
else
    # If environment variables are not set, check for django_superuser.cfg file
    if [ -f "$CONFIG_FILE" ]; then
        echo "Superuser credentials not found in environment variables. Checking $CONFIG_FILE"
        
        # Read the file and extract credentials
        DJANGO_SUPERUSER_USERNAME=$(grep "DJANGO_SUPERUSER_USERNAME" "$CONFIG_FILE" | cut -d'=' -f2)
        DJANGO_SUPERUSER_EMAIL=$(grep "DJANGO_SUPERUSER_EMAIL" "$CONFIG_FILE" | cut -d'=' -f2)
        DJANGO_SUPERUSER_PASSWORD=$(grep "DJANGO_SUPERUSER_PASSWORD" "$CONFIG_FILE" | cut -d'=' -f2)

        if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
            echo "Superuser credentials found in $CONFIG_FILE."
            echo "from django.contrib.auth.models import User; User.objects.create_superuser('$DJANGO_SUPERUSER_USERNAME', '$DJANGO_SUPERUSER_EMAIL', '$DJANGO_SUPERUSER_PASSWORD')" | python /app/RAG/chatbot_frontend/manage.py shell
        else
            echo "Superuser credentials not found in $CONFIG_FILE."
        fi
    else
        echo "No superuser environment variables or config file found."
    fi
fi

# Optionally collect static files
echo "Collecting static files..."
python /app/wattelse/chatbot/frontend/manage.py collectstatic --noinput

#stopping LLM service is needed for it to be rerun by supervisord.conf
echo "Stop LLM service"
/app/wattelse/api/fastchat/stop.sh &
sleep 4

# Start supervisord (this will keep the container running and manage other services)
exec /usr/bin/supervisord -c /app/supervisord.conf
