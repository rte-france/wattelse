#!/bin/bash

echo "Stopping Dashboard..."
screen -X -S dashboard quit

echo "Stopping Django..."
cd `pwd`/wattelse/chatbot/frontend; python stop.py; cd -
screen -X -S django quit

echo "Stopping RAG Orchestrator API..."
cd `pwd`/wattelse/api/rag_orchestrator; python stop.py; cd -
screen -X -S rag quit

echo "Stopping Embedding API..."
cd `pwd`/wattelse/api/embedding; python stop.py; cd -
screen -X -S embedding quit

#echo "Stopping LLM service"
#cd `pwd`/wattelse/api/vllm; ./stop.sh; cd -
#screen -X -S llm quit
