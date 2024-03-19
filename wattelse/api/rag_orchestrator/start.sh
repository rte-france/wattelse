# Start embedding API

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Specify the relative path to the config file
CONFIG_FILE="$SCRIPT_DIR/rag_orchestrator.cfg"

# Use grep to extract config from the config file
PORT=$(grep -Po '(?<!#)port=\K.*' "$CONFIG_FILE")

uvicorn wattelse.api.rag_orchestrator.rag_orchestrator_api:app --port=$PORT --reload --host 0.0.0.0