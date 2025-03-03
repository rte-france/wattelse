from dataclasses import dataclass, field
from pathlib import Path
import configparser
from typing import Dict, Any, Set
from wattelse.evaluation_pipeline.regex_patterns import RegexPatterns
from wattelse.evaluation_pipeline.prompt_eval import PROMPTS


@dataclass
class EvalConfig:
    """Configuration class for RAG evaluation settings."""

    config_path: Path
    default_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
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

        # Load enabled metrics
        if "EVAL_CONFIG" in config:
            metrics_str = config["EVAL_CONFIG"].get(
                "enabled_metrics", "faithfulness,correctness,retrievability"
            )
            self.enabled_metrics = {metric.strip() for metric in metrics_str.split(",")}

        # Load default model
        if "DEFAULT_MODEL" in config:
            self.default_model = config["DEFAULT_MODEL"].get(
                "default_model", self.default_model
            )

        # Load model-specific configurations
        for section in config.sections():
            if section.startswith("MODEL_"):
                model_name = config[section]["model_name"]
                # Store all configuration values for the model
                self.model_configs[model_name] = dict(config[section])

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
        return model_config.get("prompt_type", "default")

    def get_regex_type(self, model_name: str) -> str:
        """Get the regex type for a specific model."""
        model_config = self.model_configs.get(model_name, {})
        return model_config.get("regex_type", "default")

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
