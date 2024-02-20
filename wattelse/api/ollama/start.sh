# Launches ollama service

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/ollama_api.cfg"

# Use grep to extract the port number from the config file
PORT=$(grep -Po '(?<!#.*)(?<=port=).*' "$CONFIG_FILE")
CUDA_VISIBLE_DEVICES=$(grep -Po '(?<!#.*)(?<=cuda_visible_devices=).*' "$CONFIG_FILE")

# Launch ollama service
while true; do OLLAMA_HOST=$PORT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ollama serve && break; done