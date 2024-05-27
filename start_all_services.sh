#!/bin/bash
echo "Starting LLM service"
screen -dmS llm bash -c 'bash `pwd`/wattelse/api/fastchat/start_vllm.sh; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Embedding service..."
screen -dmS embedding bash -c 'bash `pwd`/wattelse/api/embedding/start.sh; bash'
sleep 3  # Waits 5 seconds.

echo "Starting RAG service..."
screen -dmS rag bash -c 'bash `pwd`/wattelse/api/rag_orchestrator/start.sh; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Django..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend; ./start.sh; bash'
sleep 3  # Waits 5 seconds.

echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend; ./dashboard.sh; bash'

