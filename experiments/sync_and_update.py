
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from client.deploy import deploy
from client.vastai import install_dependencies

def main():
    print("Syncing code and updating dependencies for 'vast-auto-test'...")
    
    # 1. Deploy (Sync)
    # Construct args object as expected by deploy()
    args = argparse.Namespace(
        target="vast-auto-test",
        path="/root/slop",
        name="vast-auto-test",
        python_cmd="python3",
        container=None,
        build=False,
        workers=8,
        unrestricted=True
    )
    
    try:
        print("Step 1: Syncing code...")
        deploy(args)
    except Exception as e:
        print(f"Sync failed: {e}")
        return

    # 2. Update Dependencies
    try:
        print("\nStep 2: Updating dependencies...")
        install_dependencies("vast-auto-test")
    except Exception as e:
        print(f"Dependency update failed: {e}")
        return
        
    print("\n✓ Update complete!")

if __name__ == "__main__":
    main()
