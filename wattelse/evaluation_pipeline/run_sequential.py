import time
import subprocess
import typer
import pandas as pd
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
            # Use net_connections() instead of connections attribute
            for conn in proc.net_connections():
                if hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    time.sleep(2)  # Give it time to terminate gracefully
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

def get_env_vars(model_name: str) -> dict:
    """Get the appropriate environment variables based on model name"""
    base_vars = {
        "OPENAI_ENDPOINT": "http://localhost:8888/v1",
        "OPENAI_API_KEY": "EMPTY"
    }
    
    if "prometheus" in model_name.lower():
        base_vars["OPENAI_DEFAULT_MODEL_NAME"] = "prometheus-eval/prometheus-7b-v2.0"
    elif "llama-3-8b" in model_name.lower():
        base_vars["OPENAI_DEFAULT_MODEL_NAME"] = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        logger.warning(f"Unknown model type {model_name}, using Meta-Llama as default")
        base_vars["OPENAI_DEFAULT_MODEL_NAME"] = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    return base_vars

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
        --dtype=half"""
    
    env_vars = f"CUDA_VISIBLE_DEVICES={config['EVAL_CONFIG']['cuda_visible_devices']}"
    
    screen_cmd = ["screen", "-S", session_name, "-X", "stuff", f"{env_vars} {vllm_cmd}\n"]
    try:
        subprocess.run(screen_cmd, check=True)
        logger.info(f"Started VLLM server for {model_name}")
        # Give the model time to load
        time.sleep(45)
        return session_name
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start VLLM server: {e}")
        raise

def stop_vllm_server(session_name: str, config: configparser.ConfigParser):
    """Stop the VLLM server and clean up"""
    try:
        # Send Ctrl+C to the screen session
        subprocess.run(["screen", "-S", session_name, "-X", "stuff", "\x03"])
        time.sleep(5)
        
        # Kill the screen session
        subprocess.run(["screen", "-S", session_name, "-X", "quit"])
        logger.info(f"Stopped screen session: {session_name}")
        
        # Clean up port
        port = int(config['EVAL_CONFIG']['port'])
        kill_process_on_port(port)
        
        # Additional cleanup time
        time.sleep(15)
    except Exception as e:
        logger.error(f"Error stopping VLLM server: {e}")
        # Force cleanup
        kill_process_on_port(int(config['EVAL_CONFIG']['port']))

def combine_evaluations(eval_files: list, output_path: Path):
    """Combine evaluations from multiple models"""
    combined_df = None
    
    for file_path in eval_files:
        try:
            df = pd.read_excel(file_path)
            if combined_df is None:
                combined_df = df
            else:
                model_name = file_path.stem.split('_')[-1]
                df_columns = df.columns.difference(['question'])
                renamed_columns = {col: f"{col}_{model_name}" for col in df_columns}
                df = df.rename(columns=renamed_columns)
                combined_df = pd.merge(combined_df, df, on='question', how='outer')
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    if combined_df is not None:
        combined_df.to_excel(output_path, index=False)
        logger.info(f"Combined evaluation saved to {output_path}")
    else:
        logger.error("No data to combine")

@app.command()
def main(
    qr_df_path: Path,
    config_path: Path = Path("eval_config.cfg"),
    output_dir: Path = Path("evaluation_results")
):
    """Run sequential evaluation with multiple models as a jury"""
    # Initial cleanup
    cleanup_screen_sessions()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of models
    models = [model.strip() for model in config['EVALUATION_SEQUENCE']['models'].split(',')]
    logger.info(f"Running evaluation with {len(models)} models as jury")
    
    evaluation_files = []
    for model in models:
        logger.info(f"Starting evaluation with model: {model}")
        model_name = model.split('/')[-1]
        
        # Ensure clean state before starting
        kill_process_on_port(int(config['EVAL_CONFIG']['port']))
        
        # Update config for current model
        if 'DEFAULT_MODEL' not in config:
            config.add_section('DEFAULT_MODEL')
        config['DEFAULT_MODEL']['default_model'] = model
        
        try:
            # Start VLLM server
            session_name = start_vllm_server(model, config)
            
            # Run evaluation with correct environment
            output_path = output_dir / f"evaluation_{model_name}.xlsx"
            evaluation_files.append(output_path)
            
            # Get the appropriate environment variables
            env_vars = get_env_vars(model)
            
            # Prepare the evaluation command
            cmd = [
                "python",
                str(Path(__file__).parent / "evaluation.py"),
                str(qr_df_path),
                "--config-path", str(config_path),
                "--report-output-path", str(output_path)
            ]
            
            # Update environment with our variables
            env = os.environ.copy()
            env.update(env_vars)
            
            # Run the command with the updated environment
            subprocess.run(cmd, env=env, check=True)
            logger.info(f"Completed evaluation with model: {model}")
            
        except Exception as e:
            logger.error(f"Error during evaluation with {model}: {e}")
        finally:
            # Stop VLLM server and clean up
            if 'session_name' in locals():
                stop_vllm_server(session_name, config)
            
            # Extra cleanup step
            kill_process_on_port(int(config['EVAL_CONFIG']['port']))
            time.sleep(10)  # Additional cool-down period
    
    # Combine evaluations
    output_suffix = config.get('EVALUATION_SEQUENCE', 'output_suffix', fallback='combined')
    combined_output = output_dir / f"{output_suffix}_evaluation.xlsx"
    combine_evaluations(evaluation_files, combined_output)
    logger.info("All evaluations completed!")

if __name__ == "__main__":
    typer.run(main)