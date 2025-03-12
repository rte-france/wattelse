#!/bin/bash

# Set logs directory and create if not exists
export WATTELSE_LOGS_DIR=$WATTELSE_BASE_DIR/logs/wattelse
mkdir -p $WATTELSE_LOGS_DIR

# Function to wait until an API is running
wait_for_api() {
    local api_name=$1
    local api_dir=$2  # Directory where status.py is located
    local sleep_time=5 # Sleep time between API health checks
    local timeout=60  # Timeout in seconds

    echo "Waiting for $api_name to start..."
    
    # Start time for timeout
    local start_time=$(date +%s)
    
    while true; do
        python "$api_dir/status.py"
        if [[ $? -eq 0 ]]; then
            echo "$api_name is running!"
            break
        fi
        
        # Calculate elapsed time
        local current_time=$(date +%s)
        local elapsed_time=$((current_time - start_time))
        
        if [[ $elapsed_time -ge $timeout ]]; then
            echo "Timeout reached while waiting for $api_name to start. Exiting..."
            exit 1
        fi
        sleep 5
    done
}

# Start Embedding API
echo "Starting Embedding API..."
screen -dmS embedding_api bash -c 'cd `pwd`/wattelse/api/embedding && python start.py 2>&1 | tee -a $WATTELSE_LOGS_DIR/embedding.log; bash'
wait_for_api "Embedding API" "`pwd`/wattelse/api/embedding"

# Start RAGOrchestrator API
echo "Starting RAGOrchestrator API..."
screen -dmS rag_orchestrator_api bash -c 'cd `pwd`/wattelse/api/rag_orchestrator && python start.py 2>&1 | tee -a $WATTELSE_LOGS_DIR/rag.log; bash'
wait_for_api "RAGOrchestrator API" "`pwd`/wattelse/api/rag_orchestrator"

# Start Django application
echo "Starting Django application..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend && python start.py 2>&1 | tee -a $WATTELSE_LOGS_DIR/django.log; bash'

# Start Dashboard
echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend/dashboard && ./dashboard.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/dashboard.log; bash'
