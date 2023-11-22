import subprocess

from loguru import logger


def add_job_to_crontab(schedule, command, env_vars=""):
    logger.info(f"Adding to crontab: {schedule} {command}")
    # Create crontab, add command
    cmd = (
        f'(crontab -l; echo "{schedule} umask 002; {env_vars} {command}" ) | crontab -'
    )
    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    logger.info(f"Crontab updated with status {returned_value}")
