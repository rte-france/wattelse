#!/bin/bash

# Set logs directory and create if not exists
export WATTELSE_LOGS_DIR=$WATTELSE_BASE_DIR/logs/wattelse
mkdir -p $WATTELSE_LOGS_DIR

#echo "Starting LLM service"
#screen -dmS llm bash -c 'bash `pwd`/wattelse/api/vllm/start.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/llm.log; bash'
#sleep 3  # Waits 5 seconds.

echo "Starting Embedding API..."
screen -dmS embedding_api bash -c 'cd `pwd`/wattelse/api/embedding && python start.py 2>&1| tee -a $WATTELSE_LOGS_DIR/embedding.log; bash'
sleep 30  # Full service start is required before to launch the RAG

echo "Starting RAGOrchestrator API..."
screen -dmS rag_orchestrator_api bash -c 'cd `pwd`/wattelse/api/rag_orchestrator && python start.py  2>&1 | tee -a $WATTELSE_LOGS_DIR/rag.log; bash'
sleep 8  # Waits 5 seconds.

echo "Starting Django application..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend && python start.py 2>&1 | tee -a $WATTELSE_LOGS_DIR/django.log; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend/dashboard && ./dashboard.sh 2>&1 | tee -a $WATTELSE_LOGS_DIR/dashboard.log; bash'
