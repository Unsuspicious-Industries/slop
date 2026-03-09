"""Test connection to a registered provider."""
import argparse
import sys
from pathlib import Path

# Fix path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.config import registry
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import Request, MessageKind, ServerInfo

def connect_and_test(name: str):
    config = registry.get(name)
    if not config:
        print(f"Error: Provider '{name}' not found in registry.")
        print("Available providers:", ", ".join([s.name for s in registry.list()]))
        sys.exit(1)
        
    print(f"Connecting to '{name}' ({config.target})...")
    
    transport = SSHTransport(config)
    
    try:
        transport.connect()
        print("✅ Connection established!")
        
        # Request Info
        print("Requesting server info...")
        info = transport.send_request(Request(kind=MessageKind.SERVER_INFO.value))
        if not isinstance(info, ServerInfo):
            raise TransportError(f"unexpected response: {type(info)}")
        
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
    parser = argparse.ArgumentParser(description="Connect to a registered inference provider.")
    parser.add_argument("name", help="Name of the provider to connect to")
    args = parser.parse_args()
    
    connect_and_test(args.name)

if __name__ == "__main__":
    main()
