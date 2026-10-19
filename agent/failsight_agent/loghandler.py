"""logging handler that forwards application log records to the monitor."""
import logging
import time


class FailSightLogHandler(logging.Handler):
    def __init__(self, client, level=logging.INFO):
        super().__init__(level)
        self.client = client

    def emit(self, record):
        if record.name.startswith(("failsight_agent", "urllib3", "httpx", "httpcore")):
            return                                        # never forward the agent's own traffic (no feedback loop)
        try:
            msg = record.getMessage()
            if record.exc_info:
                msg += " | " + repr(record.exc_info[1])
            self.client.emit("logs", dict(timestamp=record.created or time.time(), level=record.levelname, logger=record.name, message=msg[:300]))
        except Exception:
            pass
