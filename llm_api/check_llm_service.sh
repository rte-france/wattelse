# Checks the PIDs of processes serving the LLM model, if not empty the service is running
sudo pgrep  -f 'python.*fastchat'