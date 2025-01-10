#!/bin/bash

# Set logs directory and create if not exists
export WATTELSE_LOGS_DIR=$WATTELSE_BASE_DIR/logs/wattelse
mkdir -p $WATTELSE_LOGS_DIR

#echo "Starting LLM service"
#screen -dmS llm bash -c 'bash `pwd`/wattelse/api/vllm/start.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/llm.log; bash'
#sleep 3  # Waits 5 seconds.

echo "Starting Embedding service..."
screen -dmS embedding bash -c 'bash `pwd`/wattelse/api/embedding/start.sh 2>&1| tee -a $WATTELSE_LOGS_DIR/embedding.log; bash'
sleep 15  # Waits 5 seconds.

echo "Starting RAG service..."
screen -dmS rag bash -c 'bash `pwd`/wattelse/api/rag_orchestrator/start.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/rag.log; bash'
sleep 8  # Waits 5 seconds.

echo "Starting Django..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend && ./start.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/django.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend/dashboard && ./dashboard.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/dashboard.log; bash'
