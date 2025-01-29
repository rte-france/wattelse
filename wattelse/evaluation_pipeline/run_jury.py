import time
import subprocess
import typer
from pathlib import Path
import configparser
from loguru import logger
import psutil
import signal
import os

app = typer.Typer()

def kill_process_on_port(port):
    """Kill any process running on the specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections():
                if hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    time.sleep(2)
                    if psutil.pid_exists(proc.pid):
                        os.kill(proc.pid, signal.SIGKILL)
                    logger.info(f"Killed process {proc.pid} using port {port}")
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def cleanup_screen_sessions():
    """Clean up any existing VLLM screen sessions"""
    try:
        result = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'vllm_' in line:
                session_name = line.split('\t')[1].split('.')[1]
                subprocess.run(["screen", "-S", session_name, "-X", "quit"])
                logger.info(f"Cleaned up screen session: {session_name}")
    except Exception as e:
        logger.error(f"Error cleaning up screen sessions: {e}")

def create_screen_session(session_name: str) -> None:
    """Create a new screen session if it doesn't exist"""
    try:
        check_cmd = ["screen", "-ls"]
        result = subprocess.run(check_cmd, capture_output=True, text=True)
        
        if session_name not in result.stdout:
            subprocess.run(["screen", "-dmS", session_name])
            logger.info(f"Created new screen session: {session_name}")
            time.sleep(2)
    except Exception as e:
        logger.error(f"Error creating screen session: {e}")
        raise

def get_model_type(model_name: str, config: configparser.ConfigParser) -> str:
    """Determine if the model is local or cloud-based"""
    section_name = f"MODEL_{model_name.replace('/', '_').upper()}"
    if section_name in config:
        return config[section_name].get('deployment_type', 'local')
    return 'local'

def get_env_vars(model_name: str, config: configparser.ConfigParser) -> dict:
    """Get the appropriate environment variables based on model name and type"""
    model_type = get_model_type(model_name, config)
    section_name = f"MODEL_{model_name.replace('/', '_').upper()}"
    
    if model_type == 'azure':
        return {
            "OPENAI_API_TYPE": "azure",
            "OPENAI_API_VERSION": config[section_name].get('api_version', '2024-02-01'),
            "OPENAI_ENDPOINT": config[section_name]['api_base'],
            "OPENAI_API_KEY": config[section_name]['api_key'],
            "OPENAI_DEFAULT_MODEL_NAME": config[section_name]['deployment_name']
        }
    else:
        return {
            "OPENAI_ENDPOINT": "http://localhost:8888/v1",
            "OPENAI_API_KEY": "EMPTY",
            "OPENAI_DEFAULT_MODEL_NAME": model_name
        }

# TODO Set a configuration for --max-model-len {max_model_len} (Optional)
def start_vllm_server(model_name: str, config: configparser.ConfigParser) -> str:
    """Start the VLLM server with specified model"""
    port = int(config['EVAL_CONFIG']['port'])
    
    # First, ensure no process is using our port
    kill_process_on_port(port)
    
    session_name = f"vllm_{model_name.replace('/', '_')}"
    create_screen_session(session_name)
    
    vllm_cmd = f"""python -m vllm.entrypoints.openai.api_server \
        --model {model_name} \
        --port {port} \
        --host {config['EVAL_CONFIG']['host']} \
        --device auto \
        --worker-use-ray \
        --tensor-parallel-size 2 \
        --enforce-eager \
        --max-model-len 3000 \
        --dtype=half"""
    
    env_vars = f"CUDA_VISIBLE_DEVICES={config['EVAL_CONFIG']['cuda_visible_devices']}"
    
    screen_cmd = ["screen", "-S", session_name, "-X", "stuff", f"{env_vars} {vllm_cmd}\n"]
    try:
        subprocess.run(screen_cmd, check=True)
        logger.info(f"Started VLLM server for {model_name}")
        time.sleep(120)  # Give the model time to load (need to increase the time)
        return session_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start VLLM server: {e}")
        raise

def stop_vllm_server(session_name: str, config: configparser.ConfigParser):
    """Stop the VLLM server and clean up"""
    try:
        subprocess.run(["screen", "-S", session_name, "-X", "stuff", "\x03"])
        time.sleep(5)
        
        subprocess.run(["screen", "-S", session_name, "-X", "quit"])
        logger.info(f"Stopped screen session: {session_name}")
        
        port = int(config['EVAL_CONFIG']['port'])
        kill_process_on_port(port)
        
        time.sleep(15)  # Additional cleanup time
    except Exception as e:
        logger.error(f"Error stopping VLLM server: {e}")
        kill_process_on_port(int(config['EVAL_CONFIG']['port']))

def run_evaluation(model_name: str, config: configparser.ConfigParser, 
                  qr_df_path: Path, config_path: Path, output_path: Path):
    """Run evaluation for a single model"""
    model_type = get_model_type(model_name, config)
    session_name = None
    
    try:
        # Only start VLLM server for local models
        if model_type == 'local':
            kill_process_on_port(int(config['EVAL_CONFIG']['port']))
            session_name = start_vllm_server(model_name, config)
        
        # Get environment variables
        env_vars = get_env_vars(model_name, config)
        
        # Prepare and run evaluation command
        cmd = [
            "python",
            str(Path(__file__).parent / "evaluation.py"),
            str(qr_df_path),
            "--config-path", str(config_path),
            "--report-output-path", str(output_path)
        ]
        
        env = os.environ.copy()
        env.update(env_vars)
        
        subprocess.run(cmd, env=env, check=True)
        logger.info(f"Completed evaluation with model: {model_name}")
        
    except Exception as e:
        logger.error(f"Error during evaluation with {model_name}: {e}")
        raise
    finally:
        # Clean up VLLM server only if it was started
        if session_name and model_type == 'local':
            stop_vllm_server(session_name, config)
            kill_process_on_port(int(config['EVAL_CONFIG']['port']))
            time.sleep(10)

@app.command()
def main(
    qr_df_path: Path,
    config_path: Path = Path("eval_config.cfg"),
    output_dir: Path = Path("evaluation_results")
):
    """Run sequential evaluation with multiple models"""
    cleanup_screen_sessions()
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = [model.strip() for model in config['JURY_ROOM']['models'].split(',')]
    logger.info(f"Running evaluation with {len(models)} models")
    
    for model in models:
        logger.info(f"Starting evaluation with model: {model}")
        model_name = model.split('/')[-1]
        
        # Update config for current model
        if 'DEFAULT_MODEL' not in config:
            config.add_section('DEFAULT_MODEL')
        config['DEFAULT_MODEL']['default_model'] = model
        
        output_path = output_dir / f"evaluation_{model_name}.xlsx"
        
        try:
            run_evaluation(model, config, qr_df_path, config_path, output_path)
        except Exception as e:
            logger.error(f"Failed to evaluate with {model}: {e}")
            continue
    
    logger.success("All evaluations completed!")

if __name__ == "__main__":
    typer.run(main)