import json
from pathlib import Path

import client.config as config_module
from client.config import ProviderConfig, Registry


def test_registry_discards_unreachable_provider(tmp_path, monkeypatch):
    providers_file = tmp_path / "providers.json"
    providers_file.write_text(json.dumps([
        {"name": "ok", "kind": "local", "target": "local", "remote_path": str(tmp_path), "python_cmd": "python"},
        {"name": "bad", "kind": "local", "target": "local", "remote_path": str(tmp_path / 'missing'), "python_cmd": "python"},
    ]))
    monkeypatch.setattr(config_module, "PROVIDERS_FILE", providers_file)
    monkeypatch.setattr(config_module, "LEGACY_FILE", tmp_path / "servers.json")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    registry = Registry()

    assert registry.get("ok") is not None
    assert registry.get("bad") is None


def test_registry_saves_provider_list(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROVIDERS_FILE", tmp_path / "providers.json")
    monkeypatch.setattr(config_module, "LEGACY_FILE", tmp_path / "servers.json")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    registry = Registry()
    registry.add(ProviderConfig(name="local-test", kind="local", target="local", remote_path=str(tmp_path), python_cmd="python"))

    saved = json.loads((tmp_path / "providers.json").read_text())
    assert saved[0]["name"] == "local-test"
