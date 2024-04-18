#!/bin/bash

echo "Stopping Dashboard..."
screen -X -S dashboard quit

echo "Stopping Django..."
screen -X -S django quit

echo "Stopping RAG service..."
sudo `pwd`/wattelse/api/rag_orchestrator/stop.sh
screen -X -S rag quit

echo "Stopping Embedding service..."
sudo `pwd`/wattelse/api/embedding/stop.sh
screen -X -S rembedding quit

echo "Stopping LLM service"
sudo `pwd`/wattelse/api/fastchat/stop.sh
screen -X -S llm quit
