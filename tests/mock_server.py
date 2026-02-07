import sys
import os
import time
import base64
import zlib
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

from shared.protocol.wire import read_message, write_message
from shared.protocol.messages import (
    MessageKind, Request, Response, JobResult, ServerInfo, ErrorResponse
)
from shared.protocol.serialization import pack_array

def handle_inference(req_dict):
    """Mock inference response."""
    # Create fake latents: (steps, batch, channels, h, w)
    steps = req_dict.get("num_steps", 10)
    latents = np.random.randn(steps, 1, 4, 64, 64).astype(np.float32)
    
    # Pack arrays
    arrays = {
        "latents": pack_array(latents, compress=True)
    }
    
    # Create fake image (black square)
    # 1x1 pixel black png
    # minimal png header
    fake_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    
    # Construct result
    return JobResult(
        job_id=req_dict.get("job_id", "mock-job"),
        request_kind=MessageKind.INFERENCE.value,
        payload={"image": base64.b64encode(fake_png).decode('ascii')},
        arrays=arrays
    )

def main():
    # log to stderr so it doesn't corrupt stdout protocol
    print("[MOCK] Server started", file=sys.stderr)
    
    while True:
        try:
            msg = read_message(sys.stdin.buffer)
            if msg is None:
                break
                
            kind = msg.get("kind")
            print(f"[MOCK] Received: {kind}", file=sys.stderr)
            
            if kind == MessageKind.PING:
                resp = Response(kind=MessageKind.PONG, job_id=msg.get("job_id"))
                
            elif kind == MessageKind.SERVER_INFO:
                resp = ServerInfo(
                    hostname="mock-server",
                    gpu_name="MockGPU",
                    gpu_memory_mb=99999,
                    capabilities=["inference", "mocking"]
                )
                
            elif kind == MessageKind.INFERENCE:
                resp = handle_inference(msg)
                
            else:
                resp = ErrorResponse(error=f"Unknown message kind: {kind}")
            
            write_message(sys.stdout.buffer, resp.to_dict())
            sys.stdout.buffer.flush()
            
        except Exception as e:
            print(f"[MOCK] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            break

if __name__ == "__main__":
    main()
