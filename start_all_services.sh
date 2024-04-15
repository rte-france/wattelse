#!/bin/bash
echo "Starting LLM service"
screen -dmS llm bash -c 'bash `pwd`/wattelse/api/fastchat/start.sh; bash'

echo "Starting Embedding service..."
screen -dmS embedding bash -c 'bash `pwd`/wattelse/api/embedding/start.sh; bash'

echo "Starting RAG service..."
screen -dmS rag bash -c 'bash `pwd`/wattelse/api/rag_orchestrator/start.sh; bash'

echo "Starting Django..."
screen -dmS django bash -c 'cd `pwd`/wattelse/chatbot/frontend; ./start.sh; bash'

echo "Starting Dashboard..."
screen -dmS dashboard bash -c 'cd `pwd`/wattelse/chatbot/frontend; ./dashboard.sh; bash'

