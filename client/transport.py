"""Transport layer for communicating with remote server via SSH."""
import subprocess
import threading
import shlex
from typing import BinaryIO, Optional, cast
import time

from shared.protocol.wire import read_message, write_message, FrameError
from shared.protocol.messages import Request, Response, MessageKind, JobResult, ErrorResponse, ServerInfo
from client.config import ProviderConfig

class TransportError(Exception):
    pass

class SSHTransport:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready_event: Optional[threading.Event] = None
        
    def connect(self):
        """Start the remote server process via SSH."""
        # Make connect thread-safe so multiple threads don't spawn duplicate
        # SSH processes for the same transport instance.
        with self._lock:
            if self.process is not None:
                return

        # Pre-flight checks for Singularity/Container existence
        if self.config.kind == "ssh":
            self._run_remote_checks()

        remote_path_quoted = shlex.quote(self.config.remote_path)
        
        if self.config.container_image:
            # Singularity execution
            container_quoted = shlex.quote(self.config.container_image)
            
            # RESOURCE MANAGEMENT:
            # Use configured num_workers (defaults to 4) for thread control.
            num_workers = getattr(self.config, "num_workers", 4)
            
            # OMP/MKL: Controls underlying matrix multiplication threads
            # PYTORCH: Controls torch internal threads
            env_vars_list = [
                f"OMP_NUM_THREADS={num_workers}",
                f"MKL_NUM_THREADS={num_workers}",
                f"PYTORCH_NUM_THREADS={num_workers}",
                "PYTHONWARNINGS=ignore"
            ]

            # Forward HuggingFace Token if present locally
            import os
            from pathlib import Path
            hf_token = os.environ.get("HF_TOKEN")
            
            # Check for .hf_token file in project root
            if not hf_token:
                token_path = Path(__file__).parent.parent / ".hf_token"
                if token_path.exists():
                    try:
                        hf_token = token_path.read_text().strip()
                    except Exception:
                        pass

            if hf_token:
                env_vars_list.append(f"HF_TOKEN={hf_token}")

            env_vars = " ".join(env_vars_list)
            
            # We construct the inner command carefully
            # We use --env to pass variables into the container
            if hf_token:
                 # For singularity, we need to handle env vars specifically if using --env
                 # But simplistic env var prepending also works if we wrap the command properly.
                 # Actually --env takes key=val.
                 container_env = f"OMP_NUM_THREADS={num_workers},MKL_NUM_THREADS={num_workers},PYTORCH_NUM_THREADS={num_workers},PYTHONWARNINGS=ignore"
                 if hf_token:
                     container_env += f",HF_TOKEN={hf_token}"
                 
                 cmd_exec = f"singularity exec --nv --env \"{container_env}\" {container_quoted} {self.config.python_cmd} -m server.daemon"
            else:
                 cmd_exec = f"singularity exec --nv --env {env_vars} {container_quoted} {self.config.python_cmd} -m server.daemon"

            server_cmd = f"cd {remote_path_quoted} && {cmd_exec}"
        else:
            # Standard python execution
            num_workers = getattr(self.config, "num_workers", 4)
            env_vars_list = [
                f"OMP_NUM_THREADS={num_workers}",
                f"MKL_NUM_THREADS={num_workers}",
                f"PYTORCH_NUM_THREADS={num_workers}",
                "PYTHONWARNINGS=ignore"
            ]
            
            # Forward HuggingFace Token if present locally
            import os
            from pathlib import Path
            hf_token = os.environ.get("HF_TOKEN")
            
            # Check for .hf_token file in project root
            if not hf_token:
                token_path = Path(__file__).parent.parent / ".hf_token"
                if token_path.exists():
                    try:
                        hf_token = token_path.read_text().strip()
                    except Exception:
                        pass

            if hf_token:
                # print(f"[DEBUG] Forwarding HF_TOKEN: {hf_token[:4]}...{hf_token[-4:]}")
                env_vars_list.append(f"HF_TOKEN={hf_token}")
                
            env_vars = " ".join(env_vars_list)
            
            server_cmd = f"cd {remote_path_quoted} && PYTHONPATH={remote_path_quoted} {env_vars} {self.config.python_cmd} -m server.daemon"

        supervised_cmd = self._supervised_command(server_cmd)
        
        if self.config.kind == "local":
            try:
                self.process = subprocess.Popen(
                    ["bash", "-lc", supervised_cmd],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0
                )
            except Exception as e:
                raise TransportError(f"Failed to launch local process: {e}")
        else:
            ssh_cmd = [
                "ssh",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "ServerAliveInterval=15",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "BatchMode=yes",
                self.config.target,
                f"bash -lc {shlex.quote(supervised_cmd)}"
            ]
            
            try:
                self.process = subprocess.Popen(
                    ssh_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, # Capture stderr for logging
                    bufsize=0 # Unbuffered
                )
            except Exception as e:
                raise TransportError(f"Failed to launch SSH process: {e}")

        # Start stderr reader thread
        # Create an event the stderr reader can set when the server prints its ready line.
        import threading as _th
        self._ready_event = _th.Event()
        self._stderr_thread = _th.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        # Wait briefly for the server to announce readiness on stderr. Many remote
        # systems may print login banners on stdout which would corrupt our framed
        # JSON wire protocol. Waiting for the server's stderr-ready line and
        # draining any stray stdout bytes avoids deadlocks where the client
        # misinterprets login text as a frame header.
        try:
            ready = self._ready_event.wait(timeout=20)
        except Exception:
            ready = False
        if not ready:
            # Continue anyway; we'll still attempt the handshake but warn the user.
            print("[Transport] warning: server did not announce ready on stderr (continuing handshake)")

        # Drain any stray stdout bytes that may have been emitted before the
        # framed protocol started. Use select to avoid blocking.
        try:
            import select
            stdout_fd = self.process.stdout.fileno() if self.process and self.process.stdout else None
            if stdout_fd is not None:
                # Read available bytes without blocking
                while True:
                    rlist, _, _ = select.select([stdout_fd], [], [], 0)
                    if not rlist:
                        break
                    chunk = self.process.stdout.read(4096)
                    if not chunk:
                        break
                    # Discard the chunk
        except Exception:
            pass
        
        # Initial ping to verify connection
        try:
            # Use a short handshake timeout so connect() fails fast on unreachable hosts
            pong = self.send_request(Request(kind=MessageKind.PING.value), timeout_s=15)
            if pong.kind != MessageKind.PONG.value:
                raise TransportError(f"Handshake failed, got {pong.kind}")
        except Exception as e:
            # Ensure we close the process under the lock so other threads don't
            # observe a partially-initialized transport and try to reconnect.
            try:
                with self._lock:
                    self.close()
            except Exception:
                pass
            raise TransportError(f"Connection failed: {e}")

    def _run_remote_checks(self):
        """Run pre-flight checks on the remote host."""
        # Check 1: Singularity availability (if needed)
        if self.config.container_image:
            # We use 'command -v' which is standard POSIX
            check_singularity = [
                "ssh",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "BatchMode=yes",
                self.config.target,
                "command -v singularity",
            ]
            try:
                res = subprocess.run(check_singularity, capture_output=True, timeout=30)
                if res.returncode != 0:
                    raise TransportError(f"Singularity is not installed or not in PATH on {self.config.target}. Please install Singularity/Apptainer.")
            except subprocess.TimeoutExpired:
                raise TransportError(f"Timed out checking for Singularity on {self.config.target}. Is the host reachable?")
            except Exception as e:
                raise TransportError(f"Failed to check remote environment: {e}")

            # Check 2: Container image existence
            container_path = self.config.container_image
            # Resolve path relative to remote_path if not absolute
            if not container_path.startswith("/"):
                full_path = f"{self.config.remote_path}/{container_path}"
            else:
                full_path = container_path
            
            # Quote the path to prevent shell injection on the remote side
            quoted_path = shlex.quote(full_path)
            
            check_file = [
                "ssh",
                "-o",
                "ConnectTimeout=30",
                "-o",
                "BatchMode=yes",
                self.config.target,
                f"test -f {quoted_path}",
            ]
            try:
                res = subprocess.run(check_file, capture_output=True, timeout=30)
                if res.returncode != 0:
                    raise TransportError(f"Container image not found on {self.config.target} at {full_path}. Did you run 'deploy'? \nRun: python3 -m client.deploy {self.config.name}")
            except subprocess.TimeoutExpired:
                raise TransportError(f"Timed out checking for container file on {self.config.target}.")

    def _supervised_command(self, server_cmd: str) -> str:
        """Run server keeping it attached to SSH session."""
        return f"exec bash -c {shlex.quote(server_cmd)}"

    def _read_stderr(self):
        """Log remote stderr to local console for debugging."""
        import sys
        if not self.process or not self.process.stderr:
            return

        try:
            seen_ready = False
            for line in iter(self.process.stderr.readline, b""):
                try:
                    text = line.decode('utf-8', errors='replace').strip()
                except Exception:
                    text = repr(line)

                # Filter out noisy remote banners that are not useful and often
                # cause confusing output (e.g. vast.ai login banner).
                noisy_markers = ["welcome to vast.ai", "have fun!"]
                lowered = text.lower()
                if any(marker in lowered for marker in noisy_markers):
                    # Skip printing this known noisy banner
                    continue

                # Print only meaningful server lines (keep server logs)
                print(f"[REMOTE STDERR] {text}", file=sys.stderr)

                # If the server prints the readiness marker, set the event so
                # connect() can proceed with the handshake. Only set once.
                try:
                    if ("server ready" in lowered or "server ready" in text) and not seen_ready:
                        seen_ready = True
                        if self._ready_event:
                            self._ready_event.set()
                except Exception:
                    pass
        except (ValueError, OSError):
            pass

    def send_request(self, req: Request, timeout_s: float = 300.0) -> Response:
        """Send a request and block for the response."""
        with self._lock:
            if self.process is None:
                raise TransportError("Not connected. Call connect() first.")

            proc = self.process
            if proc is None:
                raise TransportError("Failed to start process")

            if not proc.stdin or not proc.stdout:
                raise TransportError("Remote process pipes are not available.")

            stdin = cast(BinaryIO, proc.stdin)
            stdout = cast(BinaryIO, proc.stdout)

            # Wire transaction
            write_message(stdin, req.to_dict())
            stdin.flush()

            response_box: dict[str, object] = {}

            def _read_once() -> None:
                try:
                    response_box["value"] = read_message(stdout)
                except Exception as exc:
                    response_box["error"] = exc

            reader = threading.Thread(target=_read_once, daemon=True)
            reader.start()
            reader.join(timeout_s)

            if reader.is_alive():
                raise TransportError(f"Request timed out after {timeout_s:.0f}s")
            if "error" in response_box:
                raise cast(Exception, response_box["error"])

            resp_dict = cast(Optional[dict], response_box.get("value"))

            if resp_dict is None:
                raise FrameError("Connection closed by server (EOF).")

            # Success - Parse response
            kind = resp_dict.get("kind")
            if kind == MessageKind.RESULT.value:
                return JobResult.from_dict(resp_dict)
            elif kind == MessageKind.ERROR.value:
                return ErrorResponse.from_dict(resp_dict)
            elif kind == MessageKind.INFO.value:
                return ServerInfo.from_dict(resp_dict)
            else:
                return Response.from_dict(resp_dict)

    def close(self):
        if self.process:
            try:
                if self.process.stdin:
                    try:
                        write_message(cast(BinaryIO, self.process.stdin), {"kind": MessageKind.SHUTDOWN.value, "job_id": "shutdown"})
                        self.process.stdin.flush()
                    except Exception:
                        pass
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=2)
            self.process = None
            self._stderr_thread = None

    def __del__(self):
        self.close()
