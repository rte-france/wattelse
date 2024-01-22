# Kills existing LLM service
# Uses sudo to kill processes possibly launched by another user
sudo pkill -SIGINT -c  -f 'ollama'
