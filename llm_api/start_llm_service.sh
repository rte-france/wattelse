# Launches the LLM (FastChat) service on port 8888
# This script has to be run on a GPU server!
PORT=8888
python -m fastchat.serve.controller&
CUDA_VISIBLE_DEVICES=2,1 python -m fastchat.serve.model_worker --model-path bofenghuang/vigogne-2-7b-instruct --num-gpus 2 &
python -m fastchat.serve.openai_api_server --host localhost --port 8888 &
