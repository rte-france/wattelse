# Launches ollama service

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/../../wattelse/config/ollama_api.cfg"

# Use grep to extract the port number from the config file
PORT=$(grep -Po '(?<=port = )\d+' "$CONFIG_FILE")

# Launch ollama service
while true; do OLLAMA_HOST=$PORT CUDA_VISIBLE_DEVICES=1,2 ollama serve && break; done