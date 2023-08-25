# Launches the LLM (FastChat) service on port 8888
# This script has to be run on a GPU server!
HOST=localhost
PORT_API=8888
PORT_CONTROLLER=21001
PORT_WORKER=21002
python -m fastchat.serve.controller --host $HOST --port $PORT_CONTROLLER &
CUDA_VISIBLE_DEVICES=2,1 python -m fastchat.serve.model_worker --host $HOST --port $PORT_WORKER --worker-address http://$HOST:$PORT_WORKER --controller-address http://$HOST:$PORT_CONTROLLER --model-path bofenghuang/vigogne-2-7b-instruct --num-gpus 2&
python -m fastchat.serve.openai_api_server --host $HOST --port $PORT_API --controller-address http://$HOST:$PORT_CONTROLLER &
