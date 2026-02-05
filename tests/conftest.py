"""Pytest configuration and fixtures.

Shared fixtures for all tests.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def temp_output_dir():
    """Create temporary output directory."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories for testing."""
    trajectories = []
    for i in range(5):
        t = np.linspace(0, 2*np.pi, 30)
        traj = np.column_stack([
            np.cos(t) + 0.1 * i,
            np.sin(t) + 0.1 * i
        ])
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def sample_flow_field():
    """Create sample 2D flow field."""
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    
    # Simple rotating field
    V = np.stack([-Y, X], axis=-1)
    return V, X, Y


@pytest.fixture
def device():
    """Get appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (model loading, etc.)"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU tests"
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and options."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(skip_gpu)
