"""Helpers for provider lifecycle operations."""

import subprocess
from typing import Optional

from client.config import ProviderConfig
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import Request, MessageKind, ServerInfo


def info(config: ProviderConfig) -> ServerInfo:
    """Return provider info from a live daemon."""
    transport = SSHTransport(config)
    try:
        transport.connect()
        response = transport.send_request(Request(kind=MessageKind.SERVER_INFO.value))
        if not isinstance(response, ServerInfo):
            raise TransportError(f"unexpected response: {type(response)}")
        return response
    finally:
        transport.close()


def restart(config: ProviderConfig) -> None:
    """Stop remote daemon processes so the next client starts a fresh one."""
    if config.kind == "local":
        return
    subprocess.run(["ssh", config.target, "pkill -f 'python.*server.daemon' || true"], check=False)


def remove_remote_files(config: ProviderConfig) -> None:
    """Delete provider files from its remote path."""
    if config.kind == "local":
        subprocess.check_call(["rm", "-rf", config.remote_path])
        return
    subprocess.check_call(["ssh", config.target, f"rm -rf {config.remote_path}"])


def tail_log(_config: ProviderConfig) -> Optional[str]:
    """Placeholder for provider-specific logs."""
    return None
