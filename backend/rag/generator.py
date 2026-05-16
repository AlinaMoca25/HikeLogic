"""Backends: 'local' (transformers, 4-bit, needs GPU) or 'hf_api'. Select via GENERATION_BACKEND."""
from threading import Lock

from .config import (
    GENERATION_BACKEND,
    GENERATION_LOAD_4BIT,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    GENERATION_TEMPERATURE,
    HF_TOKEN,
)


class Generator:
    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls) -> "Generator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        if GENERATION_BACKEND == "hf_api":
            self._impl = _HFApiBackend()
        else:
            self._impl = _LocalBackend()

    def generate(self, system: str, user: str) -> str:
        return self._impl.generate(system, user)


class _LocalBackend:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        kwargs = {"device_map": "auto"}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        if GENERATION_LOAD_4BIT:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, **kwargs)
        self.model.eval()

    def generate(self, system: str, user: str) -> str:
        torch = self._torch
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = GENERATION_TEMPERATURE > 0
        gen_kwargs = dict(
            max_new_tokens=GENERATION_MAX_TOKENS,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = GENERATION_TEMPERATURE
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()


class _HFApiBackend:
    def __init__(self):
        from huggingface_hub import InferenceClient

        if not HF_TOKEN:
            raise RuntimeError("HF_TOKEN required for hf_api backend.")
        self.client = InferenceClient(token=HF_TOKEN, provider=GENERATION_PROVIDER)

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat_completion(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE if GENERATION_TEMPERATURE > 0 else 0.01,
        )
        return response.choices[0].message.content.strip()
