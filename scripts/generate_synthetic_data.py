import json
import random
import requests
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Add project root to path to import assets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from assets.skills.globalVector import GLOBAL_SKILL_VECTOR
    from assets.skills.skillAliases import skills as SKILL_ALIASES
except ImportError as e:
    print(f"Error importing assets: {e}")
    print("Please ensure you are running this script from the project root or scripts directory and that the assets directory is correctly structured.")
    sys.exit(1)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3" # or "mistral", ensuring 8GB RAM compatibility

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'synthetic_dataset.jsonl')
NUM_SAMPLES = 5 # Initial batch for testing

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ... (imports remain similar, need glob)
import glob

# ... (Configuration)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'prompts')

def load_prompt_templates() -> List[str]:
    """Loads all .txt prompt templates from the assets directory."""
    templates = []
    pattern = os.path.join(PROMPTS_DIR, "*.txt")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                templates.append(f.read())
        except Exception as e:
            print(f"Warning: Failed to read prompt file {filepath}: {e}")
    if not templates:
        print("Warning: No prompt templates found. Using fallback.")
    return templates

def generate_prompt(selected_skills: Dict[str, float], template: str = None) -> str:
    """
    Constructs the prompt using a specific template and selected skills.
    """
    
    # 1. Prepare Skills Info (including aliases for context)
    # We filter the global alias dict to only include our selected skills
    skills_info = {}
    for skill_name, label in selected_skills.items():
        # Find aliases in global SKILL_ALIASES
        # SKILL_ALIASES is a dict where keys are canonical names
        details = SKILL_ALIASES.get(skill_name, {})
        aliases = details.get("aliases", [])
        category = details.get("category", "generic")
        
        status = "EXPLICIT (Mention Name/Alias)" if label == 1.0 else "IMPLICIT (Describe context/action, NO Name/Alias)"
        if label == 0.0: status = "NONE"

        skills_info[skill_name] = {
            "target_label": label,
            "instruction": status,
            "aliases": aliases,
            "category": category
        }

    skills_block = json.dumps(skills_info, indent=2)

    # 2. Use Template or Fallback
    if template:
        # We need to adapt the template which asks to "Randomly choose".
        # We will replace the placeholder [PASTE_SKILLS_DICT_HERE] with our specific subset
        # AND we will append a strong instruction to use *exactly* these skills.
        
        base_prompt = template.replace("[PASTE_SKILLS_DICT_HERE]", skills_block)
        
        # Override the "Randomly choose" instruction by appending a specific directive
        prompt = f"""{base_prompt}

--------------------------------------------------
CURRENT TASK OVERRIDE:
Ignore the instruction to "Randomly choose" skills. 
I have already selected the specific skills for this sample.
GENERATE EXACTLY ONE RESUME CHUNK matching the "skills_info" provided above.
Use the definitions provided in "skills_info" for EXPLICIT vs IMPLICIT.
Ensure the "text" field in the output JSON matches the seniority profile described above.
--------------------------------------------------
"""
    else:
        # Fallback simple prompt
        prompt = f"""Generate a resume text.
Skills to include:
{skills_block}
Output JSON with "text" field.
"""

    return prompt

def get_ollama_response(prompt: str, model: str = MODEL_NAME) -> str:
    """
    Sends the prompt to Ollama and returns the generated text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Force JSON output if model supports it
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return ""

def generate_sample(templates: List[str]) -> Dict[str, Any]:
    """
    Generates a single synthetic data sample.
    """
    # 1. Select Random Skills (3-6)
    num_skills = random.randint(3, 6)
    selected_names = random.sample(GLOBAL_SKILL_VECTOR, num_skills)
    
    # 2. Assign Labels
    skills_with_labels = {}
    for name in selected_names:
        label = random.choice([0.5, 1.0])
        skills_with_labels[name] = label
    
    # Ensure mix
    labels = list(skills_with_labels.values())
    if all(l == 1.0 for l in labels) and len(labels) > 1:
        skills_with_labels[selected_names[0]] = 0.5
    elif all(l == 0.5 for l in labels) and len(labels) > 1:
        skills_with_labels[selected_names[0]] = 1.0
        
    # 3. Pick a random template
    template = random.choice(templates) if templates else None
    
    # 4. Generate Prompt
    prompt = generate_prompt(skills_with_labels, template)
    
    # 5. Call LLM
    response_json_str = get_ollama_response(prompt)
    if not response_json_str:
        return None
        
    try:
        # Try to find JSON start/end if there's extra text
        start_idx = response_json_str.find('{')
        end_idx = response_json_str.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = response_json_str[start_idx:end_idx+1]
            response_data = json.loads(json_str)
        else:
            response_data = json.loads(response_json_str)
            
        generated_text = response_data.get("resume_chunk_text") or response_data.get("text", "")
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama")
        # print(f"Raw response: {response_json_str}") # Debug if needed
        return None

    if not generated_text:
        return None

    data_point = {
        "job_description": generated_text,
        "skills": skills_with_labels
    }
    
    return data_point

def main():
    print(f"Starting synthetic data generation with model: {MODEL_NAME}")
    print(f"Target file: {OUTPUT_FILE}")
    
    templates = load_prompt_templates()
    print(f"Loaded {len(templates)} prompt templates.")
    
    generated_count = 0
    # Overwrite file for fresh start as requested (5 samples)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pbar = tqdm(total=NUM_SAMPLES)
        while generated_count < NUM_SAMPLES:
            sample = generate_sample(templates)
            if sample:
                json.dump(sample, f)
                f.write('\n')
                generated_count += 1
                pbar.update(1)
            else:
                print("Skipping failed sample...")
        pbar.close()
        
    print(f"Successfully generated {generated_count} samples.")

if __name__ == "__main__":
    main()
