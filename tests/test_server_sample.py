from shared.protocol.messages import InferenceRequest


def test_sample_request_has_batch_size():
    req = InferenceRequest(prompt="x", batch_size=8, num_steps=10)
    assert req.batch_size == 8
    assert req.num_steps == 10
