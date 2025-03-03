import time
import subprocess
import typer
from pathlib import Path
from loguru import logger
import psutil
import signal
import os
import socket
import requests
from wattelse.evaluation_pipeline.eval_config import EvalConfig
from wattelse.evaluation_pipeline.server_config import ServerConfig

app = typer.Typer()

# Define base paths
BASE_DIR = Path("/DSIA/nlp/experiments")
DATA_PREDICTIONS_DIR = BASE_DIR / "data_predictions"


class PortManager:
    """Class to encapsulate all port management operations."""

    def __init__(self, logger):
        self.logger = logger
        self._last_checked_port = None
        self._last_check_time = 0

    def is_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def kill_process(self, port, verbose=True):
        """Kill any process running on the specified port"""
        if not self.is_in_use(port):
            if verbose:
                self.logger.info(f"Port {port} is not in use")
            return False

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.net_connections():
                    if hasattr(conn.laddr, "port") and conn.laddr.port == port:
                        self.logger.info(
                            f"Killing process {proc.pid} using port {port}"
                        )
                        os.kill(proc.pid, signal.SIGTERM)
                        time.sleep(2)
                        if psutil.pid_exists(proc.pid):
                            os.kill(proc.pid, signal.SIGKILL)
                        self.logger.info(f"Killed process {proc.pid} using port {port}")

                        # Verify the port is now free
                        if self.is_in_use(port):
                            self.logger.warning(
                                f"Port {port} is still in use after killing process!"
                            )
                            return False
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def wait_for_availability(self, port, timeout=180, check_interval=5, verbose=True):
        """Wait until port becomes available, with timeout"""
        start_time = time.time()
        while self.is_in_use(port):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.error(
                    f"Timeout waiting for port {port} to become available"
                )
                raise TimeoutError(f"Port {port} still in use after {timeout} seconds")

            if verbose:
                self.logger.info(
                    f"Port {port} still in use, waiting {check_interval} seconds..."
                )
            time.sleep(check_interval)

        if verbose:
            self.logger.info(f"Port {port} is now available")
        return True

    def wait_for_server_startup(self, port, timeout=180, check_interval=5):
        """Wait until VLLM server is responsive on the given port"""
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.error(
                    f"Timeout waiting for VLLM server to start on port {port}"
                )
                raise TimeoutError(f"VLLM server didn't start after {timeout} seconds")

            # Check if port is in use
            if self.is_in_use(port):
                # Try to connect to the OpenAI API endpoint to check if server is responsive
                try:
                    response = requests.get(
                        f"http://localhost:{port}/v1/models", timeout=3
                    )
                    if response.status_code == 200:
                        self.logger.info(
                            f"VLLM server is now responsive on port {port}"
                        )
                        return True
                except requests.RequestException:
                    pass

            self.logger.info(f"Waiting for VLLM server to start on port {port}...")
            time.sleep(check_interval)

    def ensure_port_free(self, port, timeout=180, force=False, verbose=True):
        """Ensure a port is free by killing any process and waiting for availability

        Args:
            port (int): The port to check
            timeout (int): Timeout in seconds
            force (bool): If True, always perform check regardless of recent checks
            verbose (bool): If True, log detailed info messages
        """
        current_time = time.time()

        # Skip if we just checked this port recently (within 2 seconds) and it was free
        if (
            not force
            and self._last_checked_port == port
            and current_time - self._last_check_time < 2
        ):
            return True

        self.kill_process(port, verbose=verbose)
        result = self.wait_for_availability(port, timeout, verbose=verbose)

        # Update the last check info
        self._last_checked_port = port
        self._last_check_time = time.time()

        return result


# Create a global instance of PortManager
port_manager = PortManager(logger)


def cleanup_screen_sessions(model_name: str = None, eval_config: EvalConfig = None):
    """Clean up VLLM screen sessions, checking model type if provided"""
    try:
        result = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "vllm_" in line:
                session_name = (
                    line.split("\t")[1].split(".")[1]
                    if "\t" in line and "." in line.split("\t")[1]
                    else None
                )
                if not session_name:
                    continue

                # If model info provided, only clean if it's a local model
                if model_name and eval_config:
                    model_type = get_model_type(model_name, eval_config)
                    if model_type != "local":
                        continue
                logger.info(f"Cleaning up screen session: {session_name}")
                subprocess.run(["screen", "-S", session_name, "-X", "quit"])
                logger.info(f"Cleaned up screen session: {session_name}")
    except Exception as e:
        logger.error(f"Error cleaning up screen sessions: {e}")


def create_screen_session(
    session_name: str, model_name: str = None, eval_config: EvalConfig = None
) -> None:
    """Create a new screen session if it doesn't exist and model is local"""
    if model_name and eval_config:
        model_type = get_model_type(model_name, eval_config)
        if model_type != "local":
            return

    try:
        check_cmd = ["screen", "-ls"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)

        if session_name not in result.stdout:
            subprocess.run(["screen", "-dmS", session_name])
            logger.info(f"Created new screen session: {session_name}")
            time.sleep(2)
        else:
            logger.info(f"Screen session {session_name} already exists")
            # Kill the existing session to ensure a clean state
            subprocess.run(["screen", "-S", session_name, "-X", "quit"])
            time.sleep(1)
            subprocess.run(["screen", "-dmS", session_name])
            logger.info(f"Recreated screen session: {session_name}")
            time.sleep(2)
    except Exception as e:
        logger.error(f"Error creating screen session: {e}")
        raise


def get_section_name(model_name: str) -> str:
    """Convert model name to config section name format."""
    return f"MODEL_{model_name.replace('/', '_').replace('-', '_').replace('.', '_').upper()}"


def get_model_type(model_name: str, eval_config: EvalConfig) -> str:
    """Determine if the model is local or cloud-based."""
    model_config = eval_config.get_model_config(model_name)
    model_type = model_config.get("deployment_type", "local")
    logger.debug(f"Model type for {model_name}: {model_type}")
    return model_type


def get_env_vars(
    model_name: str, eval_config: EvalConfig, server_config: ServerConfig
) -> dict:
    """Get environment variables based on model configuration."""
    model_config = eval_config.get_model_config(model_name)
    deployment_type = model_config.get("deployment_type", "local")

    if deployment_type == "cloud":
        return {
            "OPENAI_API_KEY": model_config["api_key"],
            "OPENAI_ENDPOINT": model_config["api_base"],
            "OPENAI_DEFAULT_MODEL_NAME": model_config["model_name"],
        }
    else:
        return {
            "OPENAI_ENDPOINT": f"http://localhost:{server_config.port}/v1",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_DEFAULT_MODEL_NAME": model_name,
        }


def start_vllm_server(
    model_name: str, eval_config: EvalConfig, server_config: ServerConfig
) -> str:
    """Start the VLLM server for local models only"""
    model_type = get_model_type(model_name, eval_config)
    logger.info(
        f"Attempting to start VLLM server for {model_name} (type: {model_type})"
    )
    if model_type != "local":
        logger.info(f"Skipping VLLM server for non-local model {model_name}")
        return None

    # Port should already be free from run_evaluation function
    # We don't need to check again, which would create duplicate logs
    port = server_config.port

    session_name = f"vllm_{model_name.replace('/', '_')}"
    create_screen_session(session_name, model_name, eval_config)

    vllm_cmd = f"""python -m vllm.entrypoints.openai.api_server \
        --model {model_name} \
        --port {port} \
        --host {server_config.host} \
        --device auto \
        --worker-use-ray \
        --tensor-parallel-size 2 \
        --enforce-eager \
        --max-model-len 8192 \
        --dtype=half"""

    cuda_devices = ",".join(map(str, server_config.cuda_visible_devices))
    env_vars = f"CUDA_VISIBLE_DEVICES={cuda_devices}"

    screen_cmd = [
        "screen",
        "-S",
        session_name,
        "-X",
        "stuff",
        f"{env_vars} {vllm_cmd}\n",
    ]
    try:
        subprocess.run(screen_cmd, check=True)
        logger.info(f"Started VLLM server for {model_name}")

        # Wait for server to be responsive using PortManager
        port_manager.wait_for_server_startup(port)

        return session_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start VLLM server: {e}")
        raise


def run_evaluation(
    model_name: str,
    eval_config: EvalConfig,
    server_config: ServerConfig,
    qr_df_path: Path,
    config_path: Path,
    output_path: Path,
):
    """Run evaluation for a single model"""
    model_type = get_model_type(model_name, eval_config)
    session_name = None

    try:
        # Get proper environment variables with correct port reference
        env_vars = get_env_vars(model_name, eval_config, server_config)
        cmd = [
            "python",
            str(Path(__file__).parent / "evaluation.py"),
            str(qr_df_path),
            "--config-path",
            str(config_path),
            "--report-output-path",
            str(output_path),
        ]

        env = os.environ.copy()
        env.update(env_vars)

        if model_type == "local":
            # Ensure port is free using PortManager - this is the ONE place we do the port check
            port_manager.ensure_port_free(server_config.port, force=True, verbose=True)

            # Start the VLLM server
            session_name = start_vllm_server(model_name, eval_config, server_config)

            # Verify server is actually running before proceeding
            if not port_manager.is_in_use(server_config.port):
                raise RuntimeError(f"VLLM server for {model_name} failed to start")

        # Run the evaluation
        logger.info(f"Running evaluation for {model_name}")
        subprocess.run(cmd, env=env, check=True)
        logger.info(f"Completed evaluation with model: {model_name}")

    except Exception as e:
        logger.error(f"Error during evaluation with {model_name}: {e}")
        raise
    finally:
        if session_name and model_type == "local":
            stop_vllm_server(session_name, server_config)

            # Make sure port is free for next model using PortManager
            port_manager.ensure_port_free(server_config.port, timeout=60)


def stop_vllm_server(session_name: str, server_config: ServerConfig):
    """Stop the VLLM server and clean up"""
    try:
        # First check if the session actually exists
        result = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
        if session_name not in result.stdout:
            logger.warning(
                f"Screen session {session_name} not found when trying to stop it"
            )
            port_manager.kill_process(server_config.port)
            return

        # Send Ctrl+C to the screen session
        subprocess.run(["screen", "-S", session_name, "-X", "stuff", "\x03"])
        time.sleep(5)

        # Check if still running and send SIGTERM if needed
        if (
            session_name
            in subprocess.run(["screen", "-ls"], capture_output=True, text=True).stdout
        ):
            subprocess.run(["screen", "-S", session_name, "-X", "quit"])
            logger.info(f"Stopped screen session: {session_name}")

        # Make sure port is free using PortManager
        port_manager.kill_process(server_config.port)

        # Additional cleanup time
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error stopping VLLM server: {e}")
        # Try to kill the port anyway
        port_manager.kill_process(server_config.port)


@app.command()
def main(
    qr_df_filename: str,
    eval_config_path: Path = Path("eval_config.cfg"),
    server_config_path: Path = Path("server_config.cfg"),
    output_dir: str = "evaluation_results",
    retry_attempts: int = 2,  # New parameter for retry attempts
):
    # Convert relative paths to absolute paths based on the base directory
    qr_df_path = DATA_PREDICTIONS_DIR / qr_df_filename
    full_output_dir = BASE_DIR / output_dir

    logger.info(f"Input file path: {qr_df_path}")
    logger.info(f"Output directory: {full_output_dir}")

    # Load both configurations
    eval_config = EvalConfig(eval_config_path)
    server_config = ServerConfig(server_config_path)

    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Get models from evaluation config
    model_configs = eval_config.model_configs
    models = list(model_configs.keys())
    logger.info(f"Running evaluation with {len(models)} models")

    # Clean up any lingering screen sessions before starting
    cleanup_screen_sessions()

    # Make sure the port is free before starting any evaluation using PortManager
    port_manager.ensure_port_free(server_config.port, force=True)

    for model in models:
        logger.info(f"Starting evaluation with model: {model}")
        model_name = model.split("/")[-1]

        output_path = full_output_dir / f"evaluation_{model_name}.xlsx"

        # Add retry logic
        for attempt in range(retry_attempts + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for model {model}")
                    # More aggressive cleanup before retry
                    cleanup_screen_sessions()
                    port_manager.ensure_port_free(server_config.port, force=True)
                    time.sleep(15)  # Give more time for resources to free up

                run_evaluation(
                    model,
                    eval_config,
                    server_config,
                    qr_df_path,
                    eval_config_path,
                    output_path,
                )
                break  # If successful, break retry loop

            except Exception as e:
                logger.error(
                    f"Failed to evaluate with {model} (attempt {attempt+1}/{retry_attempts+1}): {e}"
                )

                # Clean up after error
                cleanup_screen_sessions(model, eval_config)
                port_manager.kill_process(server_config.port, verbose=True)

                # If this was the last retry attempt, continue to next model
                if attempt == retry_attempts:
                    logger.warning(
                        f"All retry attempts failed for {model}, moving to next model"
                    )
                    time.sleep(10)
                    continue

    # Final cleanup
    cleanup_screen_sessions()
    port_manager.kill_process(server_config.port, verbose=True)

    logger.success("All evaluations completed!")


if __name__ == "__main__":
    typer.run(main)
