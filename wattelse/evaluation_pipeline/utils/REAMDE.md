# Utilities

This directory contains utility modules that support the RAG evaluation pipeline.

## Files

- `port_manager.py` - Manages network port allocation and cleanup for model servers

## Port Manager

The `PortManager` class provides robust port management capabilities:

- Checking if ports are in use
- Killing processes that occupy needed ports
- Waiting for ports to become available
- Ensuring clean port allocation between model servers
- Monitoring server startup

## Features

- Process identification and management using `psutil`
- Graceful process termination with escalation to force-kill if needed
- Configurable timeouts for port availability
- Server responsiveness checking via HTTP requests
- Detailed logging of port management operations

## Usage

```python
from loguru import logger
from wattelse.evaluation_pipeline.utils.port_manager import PortManager

# Create a port manager instance
port_manager = PortManager(logger)

# Ensure a port is free
port_manager.ensure_port_free(8888)

# Wait for a server to start on a port
port_manager.wait_for_server_startup(8888, timeout=180)

# Kill any process using a specific port
port_manager.kill_process(8888)
```

The `PortManager` is a critical component for running multiple model servers in sequence without port conflicts or orphaned processes.