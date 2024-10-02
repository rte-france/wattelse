#!/bin/bash

echo "Stopping Dashboard..."
screen -X -S dashboard quit

echo "Stopping Django..."
screen -X -S django quit

echo "Stopping RAG service..."
cd `pwd`/wattelse/api/rag_orchestrator; ./stop.sh; cd -
screen -X -S rag quit

echo "Stopping Embedding service..."
cd `pwd`/wattelse/api/embedding; ./stop.sh; cd -
screen -X -S embedding quit

echo "Stopping LLM service"
cd `pwd`/wattelse/api/vllm; ./stop.sh; cd -
screen -X -S llm quit
