import subprocess
from wattelse.django.config.settings import CONFIG

# Start Django application
command = [
    "python",
    "manage.py",
    "runserver",
    f"{CONFIG.host}:{CONFIG.port}",
    "--insecure",
]
subprocess.run(command)
