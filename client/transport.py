"""Transport layer for communicating with remote server via SSH."""
import subprocess
import threading
import json
import shlex
from typing import Optional, Dict, Any, Generator
import time

from shared.protocol.wire import read_message, write_message
from shared.protocol.messages import Request, Response, MessageKind, JobResult, ErrorResponse, ServerInfo
from client.config import ServerConfig

class TransportError(Exception):
    pass

class SSHTransport:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        
    def connect(self):
        """Start the remote server process via SSH."""
        if self.process is not None:
            return

        # Pre-flight checks for Singularity/Container existence
        if self.config.host != "local":
            self._run_remote_checks()

        # Construct safe command strings
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

            remote_cmd = f"cd {remote_path_quoted} && {cmd_exec}"
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
            
            remote_cmd = f"cd {remote_path_quoted} && {env_vars} {self.config.python_cmd} -m server.daemon"
        
        if self.config.host == "local":
            try:
                self.process = subprocess.Popen(
                    remote_cmd,
                    shell=True,
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
                self.config.host,
                remote_cmd
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
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        
        # Initial ping to verify connection
        try:
            pong = self.send_request(Request(kind=MessageKind.PING))
            if pong.kind != MessageKind.PONG:
                raise TransportError(f"Handshake failed, got {pong.kind}")
        except Exception as e:
            self.close()
            raise TransportError(f"Connection failed: {e}")

    def _run_remote_checks(self):
        """Run pre-flight checks on the remote host."""
        # Check 1: Singularity availability (if needed)
        if self.config.container_image:
            # We use 'command -v' which is standard POSIX
            check_singularity = ["ssh", self.config.host, "command -v singularity"]
            try:
                res = subprocess.run(check_singularity, capture_output=True, timeout=10)
                if res.returncode != 0:
                    raise TransportError(f"Singularity is not installed or not in PATH on {self.config.host}. Please install Singularity/Apptainer.")
            except subprocess.TimeoutExpired:
                raise TransportError(f"Timed out checking for Singularity on {self.config.host}. Is the host reachable?")
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
            
            check_file = ["ssh", self.config.host, f"test -f {quoted_path}"]
            try:
                res = subprocess.run(check_file, capture_output=True, timeout=10)
                if res.returncode != 0:
                    raise TransportError(f"Container image not found on {self.config.host} at {full_path}. Did you run 'deploy'? \nRun: python3 -m client.deploy {self.config.name}")
            except subprocess.TimeoutExpired:
                raise TransportError(f"Timed out checking for container file on {self.config.host}.")

    def _read_stderr(self):
        """Log remote stderr to local console for debugging."""
        if not self.process or not self.process.stderr:
            return
            
        for line in iter(self.process.stderr.readline, b""):
            print(f"[REMOTE STDERR] {line.decode('utf-8', errors='replace').strip()}")

    def send_request(self, req: Request) -> Response:
        """Send a request and block for the response."""
        if self.process is None:
            self.connect()

        with self._lock:
            # Type guard for process
            proc = self.process
            if proc is None or proc.poll() is not None:
                raise TransportError("Remote process died unexpectedly.")

            if not proc.stdin or not proc.stdout:
                 raise TransportError("Remote process pipes are not available.")

            try:
                # Write Request
                write_message(proc.stdin, req.to_dict())
                proc.stdin.flush()
                
                # Read Response
                resp_dict = read_message(proc.stdout)
                
                if resp_dict is None:
                    raise TransportError("Connection closed by server (EOF).")
                    
                # Hydrate response object based on kind
                kind = resp_dict.get("kind")
                if kind == MessageKind.RESULT:
                    return JobResult.from_dict(resp_dict)
                elif kind == MessageKind.ERROR:
                    return ErrorResponse.from_dict(resp_dict)
                elif kind == MessageKind.INFO:
                    return ServerInfo.from_dict(resp_dict)
                else:
                    return Response.from_dict(resp_dict)
                    
            except BrokenPipeError:
                self.close()
                raise TransportError("Broken pipe (remote server disconnected).")
            except Exception as e:
                raise TransportError(f"Protocol error: {e}")

    def close(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                self.process.kill()
            self.process = None

    def __del__(self):
        self.close()
