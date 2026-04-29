import os
import json

from dotenv import load_dotenv
import frontmatter
import google.generativeai as genai
from tqdm import tqdm
from time import sleep

from pathlib import Path 
from dotenv import load_dotenv

# --- CONFIGURATION ---
env_path = Path(__file__).resolve().parent.parent / ".env" 
load_dotenv(dotenv_path=env_path)

genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel('gemini-2-flash') 

# 2. Paths
BASE_DIR = Path(__file__).resolve().parent.parent 
INPUT_DIR = BASE_DIR / "chunking_setup" / "hiking_docs"
OUTPUT_FILE = "train_osm_mistral.jsonl"
QUESTIONS_PER_DOC = 2

def create_prompt(metadata, content):
    """Creates a prompt for the teacher LLM to generate Q&A pairs."""
    return f"""
        You are an expert Romanian mountain guide (Salvamont expert).
        Based on the technical hiking data below, generate {QUESTIONS_PER_DOC} user-assistant conversation pairs.

        TECHNICAL DATA:
        {json.dumps(metadata, indent=2)}
        Description: {content}

        REQUIREMENTS:
        - The 'User' should ask like a hiker (vague, curious, or concerned about safety).
        - The 'Assistant' must use the technical data (markings, difficulty, time) to answer accurately.
        - Format the output as a JSON list of objects: [{{"user": "...", "assistant": "..."}}]
        - IMPORTANT: Include specific Romanian trail markings (e.g., 'Cruce Albastră', 'Triunghi Roșu') if present.
        """

def format_for_mistral(user_text, assistant_text):
    """Formats the pair into the Mistral [INST] template."""
    return f"<s>[INST] {user_text} [/INST] {assistant_text}</s>"

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found.")
        return

    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.md')]
    test_files = all_files[:5]  # For quick testing, process only the first 5 files
    print(f"Found {len(all_files)} documents. Starting augmentation...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for filename in tqdm(test_files):
            try:
                # 1. Parse Markdown + YAML
                post = frontmatter.load(os.path.join(INPUT_DIR, filename))
                metadata = post.metadata
                content = post.content

                # 2. Call Teacher LLM
                response = model.generate_content(
                    create_prompt(metadata, content),
                    generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
                )
                
                # 3. Parse and Save
                qa_pairs = json.loads(response.text)
                for pair in qa_pairs:
                    formatted_line = {
                        "text": format_for_mistral(pair['user'], pair['assistant'])
                    }
                    f_out.write(json.dumps(formatted_line) + "\n")
                
                # Rate limiting for free tier (adjust as needed)
                sleep(2) 

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print(f"Success! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()