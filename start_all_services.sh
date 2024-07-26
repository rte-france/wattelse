#!/bin/bash

# Set logs directory and create if not exists
export WATTELSE_LOGS_DIR=$WATTELSE_BASE_DIR/logs
mkdir -p $WATTELSE_LOGS_DIR

echo "Starting LLM service"
screen -dmS llm bash -c 'bash `pwd`/wattelse/api/fastchat/start_vllm.sh | tee -a $WATTELSE_LOGS_DIR/llm.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Embedding service..."
screen -dmS embedding bash -c 'bash `pwd`/wattelse/api/embedding/start.sh | tee -a $WATTELSE_LOGS_DIR/embedding.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting RAG service..."
screen -dmS rag bash -c 'bash `pwd`/wattelse/api/rag_orchestrator/start.sh | tee -a $WATTELSE_LOGS_DIR/rag.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Django..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend; ./start.sh | tee -a $WATTELSE_LOGS_DIR/django.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend/dashboard; ./dashboard.sh | tee -a $WATTELSE_LOGS_DIR/dashboard.log; bash'

echo "Starting Wattelse Veille & Analyse"
screen -dmS curebot bash -c 'cd `pwd`/exploration/curebot; ./start_newsletter_generator.sh | tee -a $WATTELSE_LOGS_DIR/curebot.log; bash'