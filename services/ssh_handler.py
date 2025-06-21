import paramiko
import logging

from paramiko.ssh_exception import AuthenticationException

from common.exceptions import SSHCommandException


class SshHandler:
    def __init__(
        self, host: str, user: str, port: int = 22, timeout: int = 200
    ):
        """Initialize SSH client without connection details."""
        self._host = host
        self._user = user
        self._port = port
        self._timeout = timeout
        self._client = paramiko.SSHClient()
        self._is_connected = False

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def client(self):
        return self._client

    @is_connected.setter
    def is_connected(self, is_connected):
        self._is_connected = is_connected

    def connect(self):
        """Establish SSH connection dynamically."""
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self._client.connect(
                hostname=self._host,
                username=self._user,
                port=self._port,
                timeout=self._timeout,
                banner_timeout=self._timeout,
                allow_agent=False,
                look_for_keys=False,
                key_filename="/home/mbadea/.ssh/id_rsa"
            )
            self._is_connected = True
            logging.info(f"SSH connection established to {self._host}.")
        except AuthenticationException as e:
            logging.error(f"SSH connection failed to {self._host}.", exc_info=True)
            raise e

    def disconnect(self):
        """Close SSH connection."""
        if self._is_connected:
            self._client.close()
            self._is_connected = False
            logging.info("SSH connection closed.")

    def run_command(self, command: str, strict: bool = False):
        if not self._is_connected:
            raise ConnectionError("SSH connection is not established.")
        logging.info(f"Executing command: {command}")

        stdin, stdout, stderr = self._client.exec_command(command)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        error = stderr.read().decode("utf-8", errors="replace").strip()

        logging.debug(f"STDOUT: {output}")
        logging.debug(f"STDERR: {error}")

        if strict and error:
            raise SSHCommandException(command, error)

        return output, error

    def download_file(self, remote_path: str, local_path: str):
        if not self._is_connected:
            raise ConnectionError("SSH connection is not established.")

        try:
            sftp = self._client.open_sftp()
            try:
                sftp.stat(remote_path)  # Check if file exists
            except FileNotFoundError:
                raise FileNotFoundError(f"Remote file not found: {remote_path}")

            sftp.get(remote_path, local_path)
            sftp.close()
            logging.info(f"Downloaded file from {remote_path} to {local_path}")
        except Exception as e:
            logging.error(f"Failed to download file: {e}", exc_info=True)
            raise e

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit."""
        self.disconnect()
