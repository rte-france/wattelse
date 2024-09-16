FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

WORKDIR /app

# Copy the entire repository to the container
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    lsof \
    build-essential \
    libffi-dev \
    python3-dev \
    screen \
    supervisor \
    netcat \
    sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install .

# Show all installed modules
RUN pip list

# Create the logs directory
RUN mkdir -p /app/logs
RUN mkdir -p /app/RAG_DIR

# Set the environment variable for the base directory
ENV PACKAGE_BASE_DIR=/app/RAG_DIR

# Set default environment variable for generative model
ENV LLM_API_NAME="Fastchat LLM"

# Ensure scripts are executable
RUN chmod +x /app/wattelse/api/embedding/start.sh
RUN chmod +x /app/wattelse/api/fastchat/start.sh
RUN chmod +x /app/wattelse/api/rag_orchestrator/start.sh
RUN chmod +x /app/wattelse/chatbot/frontend/start.sh
RUN chmod +x /app/start_all_services.sh

RUN chmod +x /app/wattelse/api/embedding/stop.sh
RUN chmod +x /app/wattelse/api/fastchat/stop.sh
RUN chmod +x /app/wattelse/api/rag_orchestrator/stop.sh
RUN chmod +x /app/stop_all_services.sh

RUN chmod +x /app/entrypoint.sh

# Expose port for the application
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
