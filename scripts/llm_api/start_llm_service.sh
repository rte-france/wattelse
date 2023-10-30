# Launches the LLM (FastChat) service on port 8888
# This script has to be run on a GPU server!
# use the option "--load-8bit" to launch it in 8-bit mode

MODEL=bofenghuang/vigogne-2-7b-chat
#MODEL=mistralai/Mistral-7B-Instruct-v0.1
#MODEL=bofenghuang/vigogne-2-13b-instruct

HOST=localhost
PORT_API=8888
PORT_CONTROLLER=21001
PORT_WORKER=21002

# launch controller
python3 -m fastchat.serve.controller --host $HOST --port $PORT_CONTROLLER &

# launch worker
if [ -z "$1" ]; then
  # no args provided
  CUDA_VISIBLE_DEVICES=2,1 python3 -m fastchat.serve.model_worker --host $HOST --port $PORT_WORKER --worker-address http://$HOST:$PORT_WORKER --controller-address http://$HOST:$PORT_CONTROLLER --model-path $MODEL --model-names $MODEL --num-gpus 2&
else
  # we assume any args means '--load-8-bit' (NB. it seems that if we do not change the num-gpus values, the parameter is not taken into account
  echo "Starting service in 8-bit mode"
  CUDA_VISIBLE_DEVICES=2,1 python3 -m fastchat.serve.model_worker --host $HOST --port $PORT_WORKER --worker-address http://$HOST:$PORT_WORKER --controller-address http://$HOST:$PORT_CONTROLLER --model-path $MODEL --model-names $MODEL --num-gpus 1 --load-8bit&
fi

# launch API server
python3 -m fastchat.serve.openai_api_server --host $HOST --port $PORT_API --controller-address http://$HOST:$PORT_CONTROLLER &
