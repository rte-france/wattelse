import subprocess
from wattelse.web_app.config.settings import CONFIG

# Start Django application
command = [
    "python",
    "manage.py",
    "runserver",
    f"{CONFIG.host}:{CONFIG.port}",
    "--insecure",
]
subprocess.run(command)
