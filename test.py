import json
import torch
import requests


url = 'http://127.0.0.1:7850/te'
prompt = ['astronaut in a diner']

response = requests.post(
    url=url,
    headers={ "Content-Type": "application/json" },
    json=prompt,
    timeout=300,
)
shape = json.loads(response.headers["shape"])
bytes = bytearray(response.content)
tensor = torch.frombuffer(bytes, dtype=torch.bfloat16).reshape(shape)
print('response:', tensor.shape)
