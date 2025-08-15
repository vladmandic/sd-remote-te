from typing import Union, List, Optional
import json
import logging
import torch
import transformers
import uvicorn
import fastapi
import bitsandbytes as bnb
from safetensors.torch import _tobytes


class T5():
    def __init__(self, repo_id: Optional[str] = None, cache_dir: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        self.device = torch.device('cuda')
        self.dtype = dtype or torch.bfloat16
        self.repo_id = repo_id or 'Disty0/t5-xxl'
        self.cache_dir = cache_dir or '/mnt/models/huggingface'
        self.quant = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.dtype,
            bnb_4bit_quant_type= 'nf4',
        )
        log.info(f'Loading: repo_id={self.repo_id} device={self.device} dtype={self.dtype} cache_dir={self.cache_dir}')
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            self.repo_id,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
        )
        log.info(f'Loaded: tokenizer={self.tokenizer.__class__.__name__} vocab={self.tokenizer.vocab_size} length={self.tokenizer.model_max_length}')
        self.text_encoder = transformers.T5EncoderModel.from_pretrained(
            self.repo_id,
            cache_dir=self.cache_dir,
            torch_dtype=self.dtype,
            quantization_config=self.quant,
        ).to(self.device)
        log.info(f'Loaded: text_encoder={self.text_encoder.__class__.__name__}')


    def __call__(self, prompt: Union[str, List[str]] = None, max_sequence_length: int = 512):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device)
            )[0]
        return prompt_embeds


logging.basicConfig(level = logging.INFO, format = '%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)
log.info(f'Versions: torch={torch.__version__} transformers={transformers.__version__} uvicorn={uvicorn.__version__} fastapi={fastapi.__version__} bitsandbytes={bnb.__version__}')
t5 = T5()
log.info('Start')
app = fastapi.FastAPI()


@app.post("/te")
def encode(prompt: Union[str, List[str]] = None) -> fastapi.Response:
    log.info(f'Request: prompt={prompt}')
    embeds = t5(prompt)
    log.info(f'Response: embeds={embeds.shape}')
    embeds = embeds.to(device=torch.device('cpu'), dtype=torch.float16)
    data = _tobytes(embeds, "embeds")
    headers = { 'shape': json.dumps(list(embeds.shape)) }
    return fastapi.Response(content=data, headers=headers, media_type="tensor/binary")


uvicorn.run(app, host="0.0.0.0", port=7850)
