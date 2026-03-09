"""Smoke test for img2img protocol, client, and server changes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# 1. Test protocol changes
from shared.protocol.messages import InferenceRequest
req = InferenceRequest(
    prompt='test',
    init_image='dGVzdA==',
    strength=0.6,
)
d = req.to_dict()
assert d['init_image'] == 'dGVzdA=='
assert d['strength'] == 0.6
assert d['kind'] == 'inference'
print('✓ InferenceRequest serializes init_image + strength')

req2 = InferenceRequest.from_dict(d)
assert req2.init_image == 'dGVzdA=='
assert req2.strength == 0.6
print('✓ InferenceRequest roundtrips init_image + strength')

req3 = InferenceRequest(prompt='no img2img')
assert req3.init_image is None
assert req3.strength == 0.75
print('✓ Default: init_image=None, strength=0.75')

# 2. Test InferenceResult unpacks init_latent
from client.interface import InferenceResult
from shared.protocol.serialization import pack_array
from shared.protocol.messages import JobResult

fake_init_latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
jr = JobResult(
    job_id='test',
    request_kind='inference',
    model_id='test',
    payload={'image': ''},
    arrays={
        'init_latent': pack_array(fake_init_latent, compress=True, half=True),
    },
)
result = InferenceResult.from_job_result(jr)
assert result.init_latent is not None
assert result.init_latent.shape == (1, 4, 64, 64)
print('✓ InferenceResult unpacks init_latent correctly')

# 3. Test runner dispatch logic
from server.inference.runner import InferenceRunner
runner = InferenceRunner()
assert hasattr(runner, '_run_img2img')
print('✓ InferenceRunner has _run_img2img method')

# 4. Test client method exists
from client.interface import SlopClient
assert hasattr(SlopClient, 'generate_img2img_with_embeds')
print('✓ SlopClient has generate_img2img_with_embeds method')

# 5. Test dispatch: init_image=None -> no img2img
req_txt = InferenceRequest(prompt='txt2img test')
assert req_txt.init_image is None
print('✓ txt2img request has init_image=None (no dispatch)')

# 6. Verify generate_img2img_with_embeds signature
import inspect
sig = inspect.signature(SlopClient.generate_img2img_with_embeds)
params = list(sig.parameters.keys())
assert 'init_image' in params
assert 'strength' in params
assert 'prompt_embeds' in params
print('✓ generate_img2img_with_embeds has correct signature')

print('\nAll img2img smoke tests passed ✓')
