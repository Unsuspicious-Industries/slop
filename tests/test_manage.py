from client.config import ProviderConfig


def test_provider_config_fields():
    cfg = ProviderConfig(name="x", kind="ssh", target="host", remote_path="/tmp")
    assert cfg.name == "x"
    assert cfg.kind == "ssh"
    assert cfg.target == "host"
