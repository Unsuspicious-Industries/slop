import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent.parent


def run(script: str) -> None:
    with tempfile.TemporaryDirectory(prefix="slop_exp_") as temp:
        temp_dir = Path(temp) / "remote"
        home_dir = Path(temp) / "home"
        server_dir = temp_dir / "server"
        server_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "tests" / "mock_server.py", server_dir / "daemon.py")
        (server_dir / "__init__.py").touch()
        shutil.copytree(ROOT / "shared", temp_dir / "shared")

        providers_dir = home_dir / ".slop"
        providers_dir.mkdir(parents=True, exist_ok=True)
        providers_file = providers_dir / "providers.json"
        providers_file.write_text(
            "[\n"
            f"  {{\"name\": \"local-test\", \"kind\": \"local\", \"target\": \"local\", \"remote_path\": \"{str(temp_dir)}\", \"python_cmd\": \"{sys.executable}\", \"container_image\": null, \"num_workers\": 1}}\n"
            "]\n",
            encoding="utf-8",
        )

        subprocess.check_call(
            [sys.executable, script, "--provider", "local-test"],
            cwd=ROOT,
            env={**__import__("os").environ, "HOME": str(home_dir)},
        )


def test_sample_experiment_runs() -> None:
    run("experiments/sample.py")


def test_probe_delta_experiment_runs() -> None:
    run("experiments/probe_delta.py")


def test_render_latents_experiment_runs() -> None:
    run("experiments/render_latents.py")
