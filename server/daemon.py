import sys
import traceback
import logging
import signal
from typing import Optional

from shared.protocol.wire import read_message, write_message
from shared.protocol.messages import (
    MessageKind, 
    Request, 
    InferenceRequest, 
    EncodeRequest,
    IntrospectRequest,
    Response,
    JobResult,
    ErrorResponse,
    ServerInfo,
    Request
)
from server.inference.runner import InferenceRunner

# Configure logging to stderr so it doesn't corrupt stdout (which is for binary protocol)
logging.basicConfig(
    level=logging.INFO,
    format='[SERVER] %(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class ServerDaemon:
    def __init__(self):
        self.runner = InferenceRunner()
        self.running = True
        
        # Handle cleanup
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def handle_ping(self, req: Request) -> Response:
        return Response(kind=MessageKind.PONG, job_id=req.job_id)

    def handle_info(self, req: Request) -> ServerInfo:
        import torch
        import platform
        
        info = ServerInfo(
            job_id=req.job_id,
            hostname=platform.node(),
            cuda_version=torch.version.cuda if torch.cuda.is_available() else "N/A",
            torch_version=torch.__version__,
            capabilities=["inference", "physics_hooks"],
            loaded_models=[self.runner.current_model_id] if self.runner.current_model_id else []
        )
        
        if torch.cuda.is_available():
            info.gpu_name = torch.cuda.get_device_name(0)
            # Simple VRAM check
            t = torch.cuda.get_device_properties(0).total_memory
            info.gpu_memory_mb = int(t / 1024 / 1024)
            
        return info

    def process_request(self, req_dict: dict) -> Response:
        """Dispatch request to appropriate handler."""
        try:
            kind = req_dict.get("kind")
            
            if kind == MessageKind.PING:
                return self.handle_ping(Request.from_dict(req_dict))
                
            elif kind == MessageKind.SERVER_INFO:
                return self.handle_info(Request.from_dict(req_dict))
                
            elif kind == MessageKind.INFERENCE:
                req = InferenceRequest.from_dict(req_dict)
                logger.info(f"Running inference: {req.prompt[:50]}...")
                return self.runner.run(req)
                
            elif kind == MessageKind.SHUTDOWN:
                self.running = False
                return Response(kind=MessageKind.INFO, job_id=req_dict.get("job_id", ""), elapsed_s=0)

            else:
                return ErrorResponse(
                    job_id=req_dict.get("job_id", ""),
                    error=f"Unknown message kind: {kind}"
                )

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            traceback.print_exc(file=sys.stderr)
            return ErrorResponse(
                job_id=req_dict.get("job_id", ""),
                error=str(e),
                traceback=traceback.format_exc()
            )

    def run(self):
        logger.info("SLOP Inference Server started. Waiting for input on stdin...")
        
        # Unbuffered binary I/O
        input_stream = sys.stdin.buffer
        output_stream = sys.stdout.buffer

        while self.running:
            try:
                # 1. Read Message
                msg_data = read_message(input_stream)
                if msg_data is None:
                    # EOF
                    logger.info("Stdin closed, exiting.")
                    break
                
                # 2. Process
                response = self.process_request(msg_data)
                
                # 3. Write Response
                write_message(output_stream, response.to_dict())
                output_stream.flush()
                
            except Exception as e:
                logger.critical(f"Fatal error in main loop: {e}")
                traceback.print_exc(file=sys.stderr)
                break
                
        logger.info("Server shutting down.")

if __name__ == "__main__":
    server = ServerDaemon()
    server.run()
