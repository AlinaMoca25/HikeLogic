"""Sample two answers per prompt at different temperatures, write JSONL for human ranking → DPO."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are HikeLogic, an assistant for Romanian hiking trails.
Answer the user's question using ONLY the context provided below.
Every factual claim must be supported by one or more source ids in square brackets, such as [1] or [2].
If the context does not contain the answer, say so plainly and do not invent details.
When the context contains safety information (closures, avalanche risk, exposure, difficulty), surface it explicitly.
Prefer concise, factual answers. Cite source names when referring to specific trails or places.
Do not recommend a route as safe unless the provided context explicitly supports that."""


def _load_model(model_id: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model.eval()
    return tok, model


def _sample(tok, model, user_msg: str, temperature: float, max_new: int) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="edededi/hikelogic-qwen2.5-7b")
    parser.add_argument("--prompts", type=Path, required=True,
                        help="Plain-text file: one full user-message (incl. Context: and Question:) per non-blank section, separated by '---' lines.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temp-a", type=float, default=0.3)
    parser.add_argument("--temp-b", type=float, default=0.9)
    args = parser.parse_args()

    sections = [s.strip() for s in args.prompts.read_text(encoding="utf-8").split("\n---\n") if s.strip()]
    tok, model = _load_model(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for section in sections:
            a = _sample(tok, model, section, args.temp_a, args.max_new_tokens)
            b = _sample(tok, model, section, args.temp_b, args.max_new_tokens)
            f.write(json.dumps({
                "prompt": section,
                "answer_a": a,
                "answer_b": b,
                "chosen": "",
                "rejected": "",
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
