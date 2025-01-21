from dataclasses import dataclass, field
from pathlib import Path
import configparser
from typing import Dict, Any, List, Set
from wattelse.evaluation_pipeline.regex_patterns import RegexPatterns
from wattelse.evaluation_pipeline.prompt_eval import PROMPTS

@dataclass
class EvalConfig:
    """Configuration class for RAG evaluation settings."""
    config_path: Path
    default_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    server_config: Dict[str, Any] = field(default_factory=dict)
    model_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)
    regex_patterns: RegexPatterns = field(default_factory=RegexPatterns)
    enabled_metrics: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.load_config()
        self.prompts = PROMPTS
        
    def load_config(self):
        """Load configuration from the config file."""
        config = configparser.ConfigParser()
        config.read(self.config_path)
        
        # Load server configuration
        if 'EVAL_CONFIG' in config:
            self.server_config = dict(config['EVAL_CONFIG'])
            if 'cuda_visible_devices' in self.server_config:
                self.server_config['cuda_visible_devices'] = [
                    int(x.strip()) for x in self.server_config['cuda_visible_devices'].split(',')
                ]
                
            # Load enabled metrics
            metrics_str = self.server_config.get('enabled_metrics', 'faithfulness,correctness,retrievability')
            self.enabled_metrics = {metric.strip() for metric in metrics_str.split(',')}
        
        # Load default model
        if 'DEFAULT_MODEL' in config:
            self.default_model = config['DEFAULT_MODEL'].get('default_model', self.default_model)
        
        # Load model-specific configurations
        for section in config.sections():
            if section.startswith('MODEL_'):
                model_name = config[section]['model_name']
                self.model_configs[model_name] = {
                    'prompt_type': config[section].get('prompt_type', 'default'),
                    'regex_type': config[section].get('regex_type', 'default'),
                    'temperature': float(config[section].get('temperature', 0.0))
                }
    
    def get_prompt(self, metric: str, model_name: str) -> str:
        """Get the appropriate prompt for a given metric and model."""
        if metric not in self.enabled_metrics:
            raise ValueError(f"Metric '{metric}' is not enabled in the configuration")
        prompt_type = self.get_prompt_type(model_name)
        return self.prompts[metric][prompt_type]
    
    def get_regex_patterns(self, model_name: str) -> Dict[str, str]:
        """Get regex patterns for a specific model."""
        regex_type = self.get_regex_type(model_name)
        return self.regex_patterns.get_patterns(regex_type)
    
    def get_prompt_type(self, model_name: str) -> str:
        """Get the prompt type for a specific model."""
        model_config = self.model_configs.get(model_name, {})
        return model_config.get('prompt_type', 'default')
    
    def get_regex_type(self, model_name: str) -> str:
        """Get the regex type for a specific model."""
        model_config = self.model_configs.get(model_name, {})
        return model_config.get('regex_type', 'default')
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get the complete configuration for a specific model."""
        return self.model_configs.get(model_name, {})
    
    @property
    def available_metrics(self) -> Set[str]:
        """Get all available metrics from prompts."""
        return set(self.prompts.keys())
    
    @property
    def active_metrics(self) -> Set[str]:
        """Get currently enabled metrics that are also available."""
        return self.enabled_metrics.intersection(self.available_metrics)
    
    @property
    def cuda_devices(self) -> List[int]:
        """Get the list of CUDA devices."""
        return self.server_config.get('cuda_visible_devices', [2,1])
    
    @property
    def host(self) -> str:
        """Get the server host."""
        return self.server_config.get('host', '0.0.0.0')
    
    @property
    def port(self) -> int:
        """Get the main server port."""
        return int(self.server_config.get('port', 8888))
    
    @property
    def port_controller(self) -> int:
        """Get the controller port."""
        return int(self.server_config.get('port_controller', 21001))
    
    @property
    def port_worker(self) -> int:
        """Get the worker port."""
        return int(self.server_config.get('port_worker', 21002))