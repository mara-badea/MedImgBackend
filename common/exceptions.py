class SSHCommandException(Exception):
    def __init__(self, command: str = None, error: str = None):
        self._command = command
        self._error = error
        if command and error:
            self._message = f"SSH command failed: '{command}' -> {error}"
        else:
            self._message = f"SSH command failed."
        super().__init__(self._message)

    @property
    def message(self):
        return self._message