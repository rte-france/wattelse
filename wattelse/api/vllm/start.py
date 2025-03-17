import os
import subprocess
from wattelse.api.vllm.config.settings import get_config

# Load the configuration
CONFIG = get_config()

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG.cuda_visible_devices

# Start the vLLM OpenAI API server
command = [
    "python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    CONFIG.model_name,
    "--port",
    CONFIG.port,
    "--host",
    CONFIG.host,
    "--device",
    "auto",
    "--worker-use-ray",
    "--tensor-parallel-size",
    len(CONFIG.cuda_visible_devices.split(",")),
    "--enforce-eager",
    "--dtype",
    "half",
]
subprocess.run(command)
