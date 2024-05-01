ASR+Diarization handler that works natively with Inference Endpoints.

Example payload:
```python
import base64
import requests

API_URL = "<your endpoint URL>"
filepath = "/path/to/audio"

with open(filepath, 'rb') as f:
    audio_encoded = base64.b64encode(f.read()).decode("utf-8")

data = {
    "inputs": audio_encoded,
    "parameters": {
        "batch_size": 24
    }
}

resp = requests.post(API_URL, json=data, headers={"Authorization": "Bearer <your token>"})
print(resp.json())
```