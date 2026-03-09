from shared.protocol.messages import InferenceRequest


def test_probe_delta_fields_roundtrip():
    req = InferenceRequest(prompt="b", base_prompt="a", delta_probe=True, batch_size=4)
    restored = InferenceRequest.from_dict(req.to_dict())
    assert restored.base_prompt == "a"
    assert restored.delta_probe is True
    assert restored.batch_size == 4
