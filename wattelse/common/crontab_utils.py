#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of Wattelse, a NLP application suite.
import configparser
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from wattelse.common import BERTOPIC_LOG_PATH, BEST_CUDA_DEVICE


def add_job_to_crontab(schedule, command, env_vars=""):
    logger.info(f"Adding to crontab: {schedule} {command}")
    home = os.getenv("HOME")
    # Create crontab, add command - NB: we use the .bashrc to source all environment variables that may be required by the command
    cmd = (
        f'(crontab -l; echo "{schedule} umask 002; source {home}/.bashrc; {env_vars} {command}" ) | crontab -'
    )
    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    logger.info(f"Crontab updated with status {returned_value}")


def schedule_scrapping(
        feed_cfg: Path,
):
    """Schedule data scrapping on the basis of a feed configuration file"""
    data_feed_cfg = configparser.ConfigParser()
    data_feed_cfg.read(feed_cfg)
    schedule = data_feed_cfg.get("data-feed", "update_frequency")
    id = data_feed_cfg.get("data-feed", "id")
    command = f"{sys.prefix}/bin/python -m wattelse.data_provider scrape-feed {feed_cfg.resolve()} > {BERTOPIC_LOG_PATH}/cron_feed_{id}.log 2>&1"
    add_job_to_crontab(schedule, command, "")


def schedule_newsletter(
        newsletter_cfg_path: Path,
        data_feed_cfg_path: Path,
        cuda_devices: str = BEST_CUDA_DEVICE
):
    """Schedule data scrapping on the basis of a feed configuration file"""
    newsletter_cfg = configparser.ConfigParser()
    newsletter_cfg.read(newsletter_cfg_path)
    schedule = newsletter_cfg.get("newsletter", "update_frequency")
    id = newsletter_cfg.get("newsletter", "id")
    command = f"{sys.prefix}/bin/python -m wattelse.bertopic newsletter {newsletter_cfg_path.resolve()} {data_feed_cfg_path.resolve()} > {BERTOPIC_LOG_PATH}/cron_newsletter_{id}.log 2>&1"
    env_vars = f"CUDA_VISIBLE_DEVICES={cuda_devices}"
    add_job_to_crontab(schedule, command, env_vars)
