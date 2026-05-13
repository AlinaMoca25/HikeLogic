from threading import Lock

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

from .config import (
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
        MERGED_REPO = "alinamoca25/hikelogic-qwen2.5-1.5b-merged"

        tokenizer = AutoTokenizer.from_pretrained(MERGED_REPO)
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_REPO,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def generate(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        out = self.pipe(messages, max_new_tokens=512, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
