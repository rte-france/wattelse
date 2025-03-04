import os
import time
import signal
import socket
import psutil
import requests


class PortManager:
    """Class to encapsulate all port management operations."""

    def __init__(self, logger):
        """Initialize the PortManager with a logger.

        Args:
            logger: A logger object that has info(), error(), and warning() methods
        """
        self.logger = logger
        self._last_checked_port = None
        self._last_check_time = 0

    def is_in_use(self, port):
        """Check if a port is already in use.

        Args:
            port (int): The port number to check

        Returns:
            bool: True if the port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def kill_process(self, port, verbose=True):
        """Kill any process running on the specified port.

        Args:
            port (int): The port number to free
            verbose (bool): If True, log detailed messages

        Returns:
            bool: True if a process was killed, False otherwise
        """
        if not self.is_in_use(port):
            if verbose:
                self.logger.info(f"Port {port} is not in use")
            return False

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.net_connections():
                    if hasattr(conn.laddr, "port") and conn.laddr.port == port:
                        self.logger.info(
                            f"Killing process {proc.pid} using port {port}"
                        )
                        os.kill(proc.pid, signal.SIGTERM)
                        time.sleep(2)
                        if psutil.pid_exists(proc.pid):
                            os.kill(proc.pid, signal.SIGKILL)
                        self.logger.info(f"Killed process {proc.pid} using port {port}")

                        # Verify the port is now free
                        if self.is_in_use(port):
                            self.logger.warning(
                                f"Port {port} is still in use after killing process!"
                            )
                            return False
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def wait_for_availability(self, port, timeout=180, check_interval=5, verbose=True):
        """Wait until port becomes available, with timeout.

        Args:
            port (int): The port number to wait for
            timeout (int): Maximum seconds to wait
            check_interval (int): Seconds between checks
            verbose (bool): If True, log detailed messages

        Returns:
            bool: True if the port becomes available

        Raises:
            TimeoutError: If the port doesn't become available within timeout
        """
        start_time = time.time()
        while self.is_in_use(port):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.error(
                    f"Timeout waiting for port {port} to become available"
                )
                raise TimeoutError(f"Port {port} still in use after {timeout} seconds")

            if verbose:
                self.logger.info(
                    f"Port {port} still in use, waiting {check_interval} seconds..."
                )
            time.sleep(check_interval)

        if verbose:
            self.logger.info(f"Port {port} is now available")
        return True

    def wait_for_server_startup(self, port, timeout=180, check_interval=5):
        """Wait until server is responsive on the given port.

        Args:
            port (int): The port number to check
            timeout (int): Maximum seconds to wait
            check_interval (int): Seconds between checks

        Returns:
            bool: True if the server becomes responsive

        Raises:
            TimeoutError: If the server doesn't start within timeout
        """
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.error(f"Timeout waiting for server to start on port {port}")
                raise TimeoutError(f"Server didn't start after {timeout} seconds")

            # Check if port is in use
            if self.is_in_use(port):
                # Try to connect to the API endpoint to check if server is responsive
                try:
                    response = requests.get(
                        f"http://localhost:{port}/v1/models", timeout=3
                    )
                    if response.status_code == 200:
                        self.logger.info(f"Server is now responsive on port {port}")
                        return True
                except requests.RequestException:
                    pass

            self.logger.info(f"Waiting for server to start on port {port}...")
            time.sleep(check_interval)

    def ensure_port_free(self, port, timeout=180, force=False, verbose=True):
        """Ensure a port is free by killing any process and waiting for availability.

        Args:
            port (int): The port to check
            timeout (int): Timeout in seconds
            force (bool): If True, always perform check regardless of recent checks
            verbose (bool): If True, log detailed info messages

        Returns:
            bool: True if the port is now available

        Raises:
            TimeoutError: If the port doesn't become available within timeout
        """
        current_time = time.time()

        # Skip if we just checked this port recently (within 2 seconds) and it was free
        if (
            not force
            and self._last_checked_port == port
            and current_time - self._last_check_time < 2
        ):
            return True

        self.kill_process(port, verbose=verbose)
        result = self.wait_for_availability(port, timeout, verbose=verbose)

        # Update the last check info
        self._last_checked_port = port
        self._last_check_time = time.time()

        return result
