import os
import sys
import time
import subprocess
import typer
from pathlib import Path
from datetime import datetime
from loguru import logger

from wattelse.evaluation_pipeline import BASE_OUTPUT_DIR, RESULTS_BASE_DIR

logger.remove()
logger.add(sys.stderr, level="INFO")  # Only show INFO and above by default

# Parent directory to sys.path to resolve imports correctly
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from wattelse.evaluation_pipeline.config import EvalConfig, ServerConfig
from wattelse.evaluation_pipeline.utils import PortManager

app = typer.Typer()

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
                logger.success(f"Cleaned up screen session: {session_name}")
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
            logger.warning(f"Screen session {session_name} already exists")
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
    # Only log non-local model types to reduce verbosity
    if model_type != "local":
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
        logger.success(f"Started VLLM server for {model_name}")

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

        # Path to evaluation.py based on this file's location
        evaluation_script = Path(__file__).parent / "evaluation.py"

        cmd = [
            "python",
            str(evaluation_script),
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
        logger.success(f"Completed evaluation with model: {model_name}")

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
            port_manager.kill_process(server_config.port, verbose=True)
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
        port_manager.kill_process(server_config.port, verbose=True)

        # Additional cleanup time
        time.sleep(5)
    except Exception as e:
        logger.error(f"Error stopping VLLM server: {e}")
        # Try to kill the port anyway
        port_manager.kill_process(server_config.port, verbose=True)


def handle_output_path(path: Path, overwrite: bool) -> Path:
    """Handle file path logic based on overwrite parameter."""
    if not path.exists() or overwrite:
        if path.exists() and overwrite:
            logger.info(f"Overwriting existing file: {path.name}")
        return path

    # If not overwriting, create a new filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    logger.warning(f"File already exists. Using alternative path: {new_path.name}")
    return new_path


@app.command()
def main(
    qr_df_filename: str,
    eval_config_path: Path = Path("eval_config.toml"),
    server_config_path: Path = Path("server_config.toml"),
    output_dir: str = "evaluation_results",
    retry_attempts: int = 2,
    overwrite: bool = False,
) -> None:
    """Main function to run evaluation on all models defined in the config."""
    # Convert relative paths to absolute paths based on the base directory
    qr_df_path = BASE_OUTPUT_DIR / qr_df_filename
    full_output_dir = RESULTS_BASE_DIR / output_dir

    logger.info(
        f"Evaluation setup - Input: {'/'.join(qr_df_path.parts[-2:])}, Output dir: {'/'.join(full_output_dir.parts[-2:])}"
    )

    # Check if eval_config_path is relative and if so, try to resolve it
    if not eval_config_path.is_absolute():
        # First try current directory
        if not eval_config_path.exists():
            # Try the config directory relative to this file
            potential_path = Path(__file__).parent / "config" / eval_config_path.name
            if potential_path.exists():
                eval_config_path = potential_path

    # Same for server_config_path
    if not server_config_path.is_absolute():
        if not server_config_path.exists():
            potential_path = Path(__file__).parent / "config" / server_config_path.name
            if potential_path.exists():
                server_config_path = potential_path

    logger.info(
        f"Config paths - Eval: {'/'.join(eval_config_path.parts[-2:])}, Server: {'/'.join(server_config_path.parts[-2:])}"
    )

    # Minimal check to ensure config files exist before loading
    if not eval_config_path.exists() or not server_config_path.exists():
        logger.error("Config files are missing. Please provide valid paths.")
        sys.exit(1)

    # Load both configurations
    eval_config = EvalConfig(eval_config_path)
    server_config = ServerConfig(server_config_path)

    # Create the output directory without changing it
    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Get models from evaluation config
    model_configs = eval_config.model_configs
    models = list(model_configs.keys())
    logger.info(f"Running evaluation with {len(models)} models")

    if len(models) == 0:
        logger.error(f"No models found in config file. Please check {eval_config_path}")
        logger.debug(f"Config sections: {eval_config.config_path.read_text()}")
        return

    # Clean up any lingering screen sessions before starting
    cleanup_screen_sessions()

    # Make sure the port is free before starting any evaluation using PortManager
    port_manager.ensure_port_free(server_config.port, force=True, verbose=False)

    for model in models:
        logger.info(f"Starting evaluation with model: {model}")
        model_name = model.split("/")[-1]

        base_output_path = full_output_dir / f"evaluation_{model_name}.xlsx"
        output_path = handle_output_path(base_output_path, overwrite)

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
