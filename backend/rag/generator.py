from threading import Lock

from huggingface_hub import InferenceClient

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
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is required for generation. "
                "Get a token at https://huggingface.co/settings/tokens "
                "and add HF_TOKEN=hf_... to backend/.env."
            )
        self.client = InferenceClient(token=HF_TOKEN, provider=GENERATION_PROVIDER)

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat_completion(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
