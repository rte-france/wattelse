#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from wattelse.config_utils import load_toml_config


@dataclass
class ServerConfig:
    """Configuration class for server settings."""

    config_path: Path
    host: str = "0.0.0.0"
    port: int = 8888
    port_controller: int = 21001
    port_worker: int = 21002
    cuda_visible_devices: List[int] = field(default_factory=lambda: [2, 1])

    def __post_init__(self):
        self.load_config()

    def load_config(self):
        """Load configuration from the server config file."""
        config = load_toml_config(self.config_path)

        if "SERVER_CONFIG" in config:
            server_config = config["SERVER_CONFIG"]
            self.host = server_config.get("host", self.host)
            self.port = int(server_config.get("port", self.port))
            self.port_controller = int(
                server_config.get("port_controller", self.port_controller)
            )
            self.port_worker = int(server_config.get("port_worker", self.port_worker))

            if "cuda_visible_devices" in server_config:
                self.cuda_visible_devices = server_config["cuda_visible_devices"]
