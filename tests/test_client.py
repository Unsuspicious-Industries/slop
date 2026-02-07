
import unittest
import sys
import os
import shutil
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.config import registry, ServerConfig
from client.interface import SlopClient

class TestClientIntegration(unittest.TestCase):
    def setUp(self):
        # Create a dummy config pointing to our mock server
        # We simulate the "remote" command by just running the mock script locally
        self.mock_script = project_root / "tests" / "mock_server.py"
        
        # We'll use a specific python command that runs the mock server
        # The transport layer does: "cd {remote_path} && {python_cmd} -m server.daemon"
        # Since we are hacking "host=local", we can manipulate python_cmd to run our mock.
        
        # However, Transport expects to run a module. 
        # Let's register a config where 'python_cmd' is the python executable, 
        # but we need to trick it into running mock_server.py instead of server.daemon.
        
        # Actually, simpler approach:
        # Create a fake 'server' package in a temp dir that imports mock_server logic?
        # Or just rely on the fact that Transport allows us to override things?
        
        # Let's assume we modify the config to point to a test-specific entry point if needed.
        # But wait, Transport hardcodes `-m server.daemon`. 
        
        # BETTER IDEA: 
        # We can temporarily swap the transport's command construction or subclass it?
        # No, that modifies code.
        
        # Let's create a temporary `server/daemon.py` in a temp directory that redirects to `tests/mock_server.py` logic?
        # No, that's messy.
        
        # The cleanest integration test without changing code:
        # 1. Create a temporary directory.
        # 2. Copy `tests/mock_server.py` to `{temp_dir}/server/daemon.py`.
        # 3. Create `__init__.py` in `{temp_dir}/server`.
        # 4. Point the client config `remote_path` to `{temp_dir}`.
        # 5. Point `host` to "local".
        
        self.test_dir = project_root / "tests" / "temp_env"
        self.server_dir = self.test_dir / "server"
        self.server_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy mock server to be server.daemon
        shutil.copy(self.mock_script, self.server_dir / "daemon.py")
        (self.server_dir / "__init__.py").touch()
        
        # Create shared/protocol symlink so the daemon can import it
        # (Since mock_server imports shared.protocol...)
        self.shared_dir = self.test_dir / "shared"
        if not self.shared_dir.exists():
            # os.symlink(project_root / "shared", self.shared_dir)
            # Symlinks might get messy with relative paths in python imports, lets copy
            shutil.copytree(project_root / "shared", self.shared_dir)

        # Config
        self.config = ServerConfig(
            name="test-mock",
            host="local",
            remote_path=str(self.test_dir),
            python_cmd=sys.executable,
            container_image=None # No container for local test
        )

    def tearDown(self):
        # Cleanup
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_client_flow(self):
        print("\n--- Testing Client <-> Mock Server ---")
        
        with SlopClient(self.config) as client:
            # 1. Check Info
            print("1. Getting Server Info...")
            info = client.get_server_info()
            print(f"   Got info from {info.hostname}")
            self.assertEqual(info.gpu_name, "MockGPU")
            
            # 2. Run Inference
            print("2. Running Inference...")
            prompt = "Test prompt"
            result = client.generate(prompt, num_steps=5)
            
            # 3. Verify Results
            print("3. Verifying Results...")
            self.assertIsNotNone(result.image)
            self.assertGreater(len(result.image), 0)
            
            self.assertIsNotNone(result.latents)
            # Mock server generates (steps, 1, 4, 64, 64)
            print(f"   Latents shape: {result.latents.shape}")
            self.assertEqual(result.latents.shape, (5, 1, 4, 64, 64))
            
            # 4. Test Analytics Helpers
            print("4. Testing Analytics...")
            # Create a fake pole vector (matches flattened latent dimension)
            # flattened dim = 1 * 4 * 64 * 64 = 16384
            flat_dim = 1 * 4 * 64 * 64
            dummy_pole = np.random.randn(1, flat_dim)
            
            drift = client.analyze_drift(result, dummy_pole)
            print(f"   Drift score: {drift}")
            self.assertIsInstance(drift, float)

if __name__ == "__main__":
    unittest.main()
