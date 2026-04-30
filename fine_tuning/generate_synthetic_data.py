import os
import json
import re
import urllib.request
import urllib.error

from dotenv import load_dotenv
import frontmatter
from tqdm import tqdm
from time import sleep

from pathlib import Path

# --- CONFIGURATION ---
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
GENERATION_MODE = os.getenv("GENERATION_MODE", "template").lower()

# 2. Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "chunking_setup" / "hiking_docs"
OUTPUT_FILE = BASE_DIR / "fine_tuning" / "train_osm_mistral.jsonl"
QUESTIONS_PER_DOC = int(os.getenv("QUESTIONS_PER_DOC", "1"))
MAX_DOCS = int(os.getenv("MAX_DOCS", os.getenv("GENAI_MAX_DOCS", "0")))
CONTENT_MAX_CHARS = int(os.getenv("CONTENT_MAX_CHARS", "800"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "220"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90"))
SLEEP_BETWEEN_DOCS = float(os.getenv("SLEEP_BETWEEN_DOCS", "0"))


def parse_json_response(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to recover JSON embedded in extra text.
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if not match:
            raise
        candidate = match.group(1)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)  # remove trailing commas
        parsed = json.loads(candidate)

    # Accept common model output shapes:
    # 1) [{"user":"...","assistant":"..."}]
    # 2) {"pairs":[{"user":"...","assistant":"..."}]}
    # 3) {"user":"...","assistant":"..."}
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "pairs" in parsed and isinstance(parsed["pairs"], list):
            return parsed["pairs"]
        if "conversations" in parsed and isinstance(parsed["conversations"], list):
            return parsed["conversations"]
        if "user" in parsed and "assistant" in parsed:
            return [parsed]
    raise ValueError("Unexpected JSON shape from model response.")


def clean_text(value):
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()

def call_ollama(prompt, max_output_tokens):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_output_tokens,
            "temperature": 0.2,
        },
    }
    req = urllib.request.Request(
        url=f"{OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Start it with `ollama serve` and make sure the model is pulled."
        ) from exc


def generate_pairs(prompt):
    raw_response = call_ollama(prompt, MAX_OUTPUT_TOKENS)
    try:
        return parse_json_response(raw_response), raw_response
    except Exception:
        # Retry once with a higher token budget to reduce truncated JSON.
        retry_prompt = (
            prompt
            + "\n\nIMPORTANT: Return only valid JSON with closed quotes/brackets."
        )
        raw_response = call_ollama(retry_prompt, max(MAX_OUTPUT_TOKENS * 2, 300))
        return parse_json_response(raw_response), raw_response


def generate_template_pairs(metadata, content):
    name = clean_text(metadata.get("name", "acest traseu"))
    difficulty = clean_text(metadata.get("difficulty", "necunoscut"))
    marking = clean_text(metadata.get("marking", "necunoscut"))
    elevation_gain = clean_text(metadata.get("elevation_gain", "necunoscut"))
    region = clean_text(metadata.get("region", "Romania"))

    snippet = clean_text(content[:220])
    if not snippet:
        snippet = "Nu exista descriere suplimentara in document."

    pairs = []
    for _ in range(max(1, QUESTIONS_PER_DOC)):
        user = (
            f"Vreau sa merg pe traseul {name}. "
            f"Cat de dificil este, ce marcaj are si ce ar trebui sa stiu inainte sa plec?"
        )
        assistant = (
            f"Traseul {name} este in zona {region}. "
            f"Dificultatea raportata este '{difficulty}', marcajul este '{marking}', "
            f"iar diferenta de nivel indicata este '{elevation_gain}'. "
            f"Detalii utile din document: {snippet}. "
            "Verifica prognoza, ia apa suficienta si anunta ruta inainte de plecare."
        )
        pairs.append({"user": user, "assistant": assistant})
    return pairs


def normalize_pairs(qa_pairs):
    normalized = []
    for pair in qa_pairs:
        if not isinstance(pair, dict):
            continue
        user_text = clean_text(pair.get("user"))
        assistant_text = clean_text(pair.get("assistant"))
        if user_text and assistant_text:
            normalized.append({"user": user_text, "assistant": assistant_text})
    return normalized

def create_prompt(metadata, content):
    """Creates a prompt for the teacher LLM to generate Q&A pairs."""
    return f"""
        You are an expert Romanian mountain guide (Salvamont expert).
        Based on the technical hiking data below, generate {QUESTIONS_PER_DOC} user-assistant conversation pairs.

        TECHNICAL DATA:
        {json.dumps(metadata, indent=2)}
        Description: {content[:CONTENT_MAX_CHARS]}

        REQUIREMENTS:
        - The 'User' should ask like a hiker (vague, curious, or concerned about safety).
        - The 'Assistant' must use the technical data (markings, difficulty, time) to answer accurately.
        - Output MUST be valid JSON and only JSON.
        - Format the output as a JSON list of objects exactly like:
          [{{"user": "...", "assistant": "..."}}, {{"user": "...", "assistant": "..."}}]
        - IMPORTANT: Include specific Romanian trail markings (e.g., 'Cruce Albastră', 'Triunghi Roșu') if present.
        """

def format_for_mistral(user_text, assistant_text):
    """Formats the pair into the Mistral [INST] template."""
    return f"<s>[INST] {user_text} [/INST] {assistant_text}</s>"

def main():
    print(f"Using local Ollama model: {OLLAMA_MODEL}")
    print(f"Ollama endpoint: {OLLAMA_URL}")
    print(f"Generation mode: {GENERATION_MODE}")
    print(
        "Speed settings: "
        f"QUESTIONS_PER_DOC={QUESTIONS_PER_DOC}, "
        f"CONTENT_MAX_CHARS={CONTENT_MAX_CHARS}, "
        f"MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS}"
    )
    if not INPUT_DIR.exists():
        print(f"Error: {INPUT_DIR} not found.")
        return

    all_files = sorted(INPUT_DIR.glob("*.md"))
    if MAX_DOCS > 0:
        all_files = all_files[:MAX_DOCS]
    print(f"Found {len(all_files)} documents. Starting augmentation...")
    if not all_files:
        print("No source markdown files found. Run chunking_setup/create_hiking_docs.py first.")
        return

    generated_pairs = 0
    llm_success_docs = 0
    fallback_docs = 0
    with OUTPUT_FILE.open("w", encoding="utf-8") as f_out:
        for file_path in tqdm(all_files):
            raw_response = ""
            try:
                # 1. Parse Markdown + YAML
                post = frontmatter.load(file_path)
                metadata = post.metadata
                content = post.content

                # 2. Generate pairs with selected mode
                qa_pairs = []
                if GENERATION_MODE in {"llm", "hybrid"}:
                    try:
                        qa_pairs, raw_response = generate_pairs(create_prompt(metadata, content))
                        qa_pairs = normalize_pairs(qa_pairs)
                        if qa_pairs:
                            llm_success_docs += 1
                    except Exception:
                        qa_pairs = []

                if not qa_pairs:
                    qa_pairs = generate_template_pairs(metadata, content)
                    qa_pairs = normalize_pairs(qa_pairs)
                    fallback_docs += 1

                # 3. Parse and Save
                for pair in qa_pairs:
                    formatted_line = {
                        "text": format_for_mistral(pair["user"], pair["assistant"])
                    }
                    f_out.write(json.dumps(formatted_line, ensure_ascii=False) + "\n")
                    generated_pairs += 1

                if SLEEP_BETWEEN_DOCS > 0:
                    sleep(SLEEP_BETWEEN_DOCS)

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                # Print a short preview to help diagnose model format issues.
                if raw_response:
                    preview = raw_response.replace("\n", " ")[:300]
                    print(f"Response preview: {preview}")
                continue

    print(f"Success! Data saved to {OUTPUT_FILE} with {generated_pairs} examples.")
    print(
        f"Docs with LLM output: {llm_success_docs} | "
        f"Docs with template fallback: {fallback_docs}"
    )

if __name__ == "__main__":
    main()