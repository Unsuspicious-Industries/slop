"""Test connection to a registered server."""
import argparse
import sys
from pathlib import Path

# Fix path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.config import registry
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import Request, MessageKind

def connect_and_test(name: str):
    config = registry.get(name)
    if not config:
        print(f"Error: Server '{name}' not found in registry.")
        print("Available servers:", ", ".join([s.name for s in registry.list()]))
        sys.exit(1)
        
    print(f"Connecting to '{name}' ({config.host})...")
    
    transport = SSHTransport(config)
    
    try:
        transport.connect()
        print("✅ Connection established!")
        
        # Request Info
        print("Requesting server info...")
        info = transport.send_request(Request(kind=MessageKind.SERVER_INFO))
        
        print("\n=== Server Info ===")
        print(f"Hostname: {info.hostname}")
        print(f"GPU: {info.gpu_name} ({info.gpu_memory_mb} MB)")
        print(f"PyTorch: {info.torch_version}")
        print(f"CUDA: {info.cuda_version}")
        print(f"Capabilities: {', '.join(info.capabilities)}")
        print("===================\n")
        
    except TransportError as e:
        print(f"❌ Connection failed: {e}")
    finally:
        transport.close()

def main():
    parser = argparse.ArgumentParser(description="Connect to a registered SLOP server.")
    parser.add_argument("name", help="Name of the server to connect to")
    args = parser.parse_args()
    
    connect_and_test(args.name)

if __name__ == "__main__":
    main()
