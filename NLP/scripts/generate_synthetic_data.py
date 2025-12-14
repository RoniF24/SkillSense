import json
import random
import os
import sys
from typing import List, Dict, Any, Optional
import glob
from collections import Counter
import re

import requests
from tqdm import tqdm

# Ensure project root is on sys.path (so running via: python scripts\generate_synthetic_data.py works)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR
from skills.skillAliases import skills as SKILL_ALIASES

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"

PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")   # <<< אצלך Prompts
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")       # <<< אצלך data
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_dataset.jsonl")

NUM_SAMPLES = 5

# generation params
TEMPERATURE_GEN = 0.2
MAX_TOKENS_GEN = 180

# fix pass params (more deterministic)
TEMPERATURE_FIX = 0.0
MAX_TOKENS_FIX = 220

# retries
MAX_GEN_RETRIES = 3          # initial generations
MAX_FIX_RETRIES = 3          # LLM fix passes if still failing
# ----------------------------------------

# Label distribution: 70% explicit, 30% implicit (reduces failures a lot)
P_EXPLICIT = 0.4

ROLE_FAMILIES = ["Software", "Data", "DevOps", "Security", "Product", "Management"]
SENIORITIES = ["Intern", "Junior", "Mid", "Senior", "Lead", "Manager"]
RESP_DEPTH = ["IC", "TechLead", "PeopleManager"]
DOMAINS = ["FinTech", "E-commerce", "Healthcare", "Cyber", "SaaS", "Gaming"]

# משקולות כדי שלא הכל יצא "Senior" אוטומטית
SENIORITY_WEIGHTS = [0.12, 0.22, 0.26, 0.20, 0.12, 0.08]

def sample_context() -> Dict[str, str]:
    seniority = random.choices(SENIORITIES, weights=SENIORITY_WEIGHTS, k=1)[0]

    if seniority in ["Intern", "Junior"]:
        responsibility_depth = "IC"
    elif seniority == "Mid":
        responsibility_depth = random.choice(["IC", "TechLead"])
    elif seniority in ["Senior", "Lead"]:
        responsibility_depth = random.choice(["IC", "TechLead"])
    else:  # Manager
        responsibility_depth = "PeopleManager"

    return {
        "role_family": random.choice(ROLE_FAMILIES),
        "seniority": seniority,
        "responsibility_depth": responsibility_depth,
        "domain": random.choice(DOMAINS),
    }

# --------- IMPLICIT ALLOWED (Layer 2) ---------
IMPLICIT_ALLOWED = {
    # Cloud / services (implicit via behaviors)
    "AWS EC2", "AWS Lambda", "AWS RDS", "AWS S3",
    "Azure Functions",
    "BigQuery", "Redshift", "Snowflake",

    # Delivery / CI-CD / deployments
    "CI/CD", "GitOps", "GitHub Actions", "GitLab CI", "Jenkins",
    "Terraform", "CloudFormation",
    "Blue-Green Deployment", "Canary Releases", "ArgoCD",

    # Infra / reliability / networking
    "Load Balancing", "Cloud Networking", "Security Hardening",
    "Performance Engineering",

    # Observability
    "Grafana", "Prometheus", "ELK Stack",

    # Data engineering / orchestration / formats
    "ETL", "ELT", "Airflow",
    "Parquet", "Avro", "Star Schema",

    # Distributed / streaming
    "Distributed Systems", "Kafka",

    # ML concepts (easy to imply strongly)
    "MLOps", "MLflow",
    "Machine Learning", "NLP", "LLMs", "Transformers",
}

IMPLICIT_REPLACEMENTS = {
    "AWS Lambda": "event-driven serverless functions triggered by system events and queued messages",
    "GitOps": "declarative infrastructure changes managed via version-controlled configuration and automated sync",
    "CI/CD": "pipelines triggered on pull requests with automated test stages, deployment gates, and rollback steps",
    "MLflow": "experiment tracking, model versioning, and promotion between stages to ensure reproducibility",
    "MLOps": "model packaging, automated validation checks, and controlled promotion to staging/production",
    "Star Schema": "fact and dimension tables designed for analytical queries and reporting performance",
    "Cloud Networking": "traffic routing and subnet-level configuration to control request paths and connectivity",
    "Load Balancing": "request distribution across instances with health checks and failover routing",
    "Kafka": "event streams with producers/consumers, topic partitioning, and throughput-focused ingestion",
    "Redshift": "a columnar data warehouse used for analytics with optimized aggregations and reporting queries",
    "BigQuery": "serverless analytics queries over large datasets with partitioning and cost-aware query patterns",
    "Snowflake": "cloud data warehouse workflows with separated compute/storage and optimized analytics queries",
    "Airflow": "scheduled DAG-based workflows with task dependencies, retries, and backfills",
    "Parquet": "columnar storage files used to reduce size and speed up analytics reads",
    "Avro": "schema-based serialization for consistent data exchange between services",
    "Terraform": "infrastructure-as-code with planned changes, state management, and repeatable environments",
    "CloudFormation": "infrastructure templates defining resources and updates in a repeatable way",
    "NLP": "text processing tasks like tokenization, normalization, and lightweight intent/classification experiments",
}

IMPLICIT_HINT_PATTERNS = {
    "CI/CD": [
        r"\bci\s*/\s*cd\b",
        r"\bcontinuous integration\b",
        r"\bcontinuous deployment\b",
        r"\bdeployment gates?\b",
        r"\brelease gates?\b",
        r"\brollback\b",
        r"\bgated releases?\b",
        r"\bbuild[- ]and[- ]deploy\b",
        r"\brelease pipeline\b",
        r"\bdeployment pipeline\b",
        r"\bautomated test stages?\b",
        r"\b(build|test|deploy|release)\s+pipeline(s)?\b",
        r"\bautomated build\b",
        r"\bbuild (and|&)\s*test\b",
        r"\bstaged rollout\b",
        r"\brelease automation\b",
    ],
    "GitOps": [
        r"\bdeclarative\b",
        r"\bversion[- ]controlled\b",
        r"\bautomated sync\b",
        r"\bdesired state\b",
    ],
    "Cloud Networking": [
        r"\bsubnet\b",
        r"\btraffic routing\b",
        r"\brouting tables?\b",
        r"\bnetwork acl\b",
        r"\bvpc\b",
    ],
    "Load Balancing": [
        r"\bload balanc(er|ing)?\b",
        r"\bround[- ]robin\b",
        r"\btraffic distribution across instances\b",
    ],
    "MLflow": [
        r"\bexperiment tracking\b",
        r"\bmodel registry\b",
        r"\bpromot(e|ion) between stages\b",
    ],
    "MLOps": [
        r"\bmodel versioning\b",
        r"\bmodel packaging\b",
        r"\bpromotion to (staging|production)\b",
    ],
    "Star Schema": [
        r"\bfact tables?\b",
        r"\bdimension tables?\b",
        r"\bdimensional modeling\b",
    ],
    "Redshift": [
        r"\bcolumnar\b",
        r"\bdata warehouse\b",
    ],
    "Kafka": [
        r"\btopic(s)?\b",
        r"\bpartition(s)?\b",
        r"\bconsumer(s)?\b",
        r"\bproducer(s)?\b",
        r"\bevent streams?\b",
    ],
}

IMPLICIT_REQUIRED_PATTERNS = {
    "Star Schema": [
        r"\bfact\b", r"\bdimension\b", r"\bfact tables?\b", r"\bdimension tables?\b", r"\bdimensional modeling\b"
    ],
    "Redshift": [
        r"\bcolumnar\b", r"\bdata warehouse\b", r"\baggregations?\b", r"\breporting queries?\b"
    ],
    "Distributed Systems": [
        r"\bservice-to-service\b", r"\bdistributed\b", r"\bconsistency\b", r"\bcoordination\b", r"\bmessage queue\b"
    ],
    "GitOps": [
        r"\bdesired[- ]state\b", r"\breconciliation\b", r"\bautomated sync\b", r"\bdrift\b"
    ],
    "CI/CD": [
        r"\bdeployment gates?\b", r"\bgated releases?\b", r"\brollback\b", r"\bbuild[- ]and[- ]deploy\b"
    ],
    "NLP": [
        r"\btokeni[sz]ation\b", r"\bnormalization\b", r"\btext classification\b", r"\bprecision\b", r"\brecall\b"
    ],
    "ETL": [
        r"\bextract\b", r"\btransform\b", r"\bload\b", r"\bingestion\b"
    ],
}

# ----------------------------------------------

# Stats
STATS = Counter()
TOTAL_CALLS = 0


# ---------------- helpers ----------------
class ValidationError(Exception):
    def __init__(self, kind: str, skill: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.skill = skill


def load_prompt_templates() -> List[str]:
    fp = os.path.join(PROMPTS_DIR, "unified_prompt.txt")
    if not os.path.exists(fp):
        raise RuntimeError(f"unified_prompt.txt not found in: {PROMPTS_DIR}")
    with open(fp, "r", encoding="utf-8") as f:
        return [f.read()]


def build_skills_info(selected_skills: Dict[str, float]) -> Dict[str, Any]:
    skills_info = {}
    for skill_name, label in selected_skills.items():
        details = SKILL_ALIASES.get(skill_name, {})
        aliases = details.get("aliases", [])
        category = details.get("category", "generic")

        instruction = (
            "EXPLICIT: mention the canonical skill name OR one alias verbatim"
            if label == 1.0
            else "IMPLICIT: DO NOT mention canonical name nor any alias verbatim; imply via responsibilities"
        )

        skills_info[skill_name] = {
            "target_label": label,
            "instruction": instruction,
            "aliases": aliases,
            "category": category,
        }
    return skills_info


def generate_prompt(selected_skills: Dict[str, float], template: str) -> str:
    skills_info = build_skills_info(selected_skills)
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)

    base_prompt = template.replace("[PASTE_SKILLS_DICT_HERE]", skills_block)

    ctx = sample_context()
    for k, v in ctx.items():
        base_prompt = base_prompt.replace("{" + k + "}", v)

    return f"""
IMPORTANT OVERRIDE (MUST FOLLOW):
- Ignore ANY instruction above that asks for JSON, code, scripts, or asking the user for input.
- You already have all inputs you need.
- Output MUST be plain text ONLY: ONE resume-style paragraph (4–6 sentences).
- Do NOT output code. Do NOT output markdown. Do NOT output JSON.
- Do NOT include meta text like: "Here is", "I'll", "I will", "Please provide", "Corrected paragraph".

{base_prompt}

--------------------------
OUTPUT RULES (STRICT):
- Return ONLY ONE resume-style paragraph (4–6 sentences), plain text.
- No JSON. No code blocks. No markdown. No bullet lists.
- Do NOT add skills beyond the provided skills_info.
- Follow EXPLICIT vs IMPLICIT rules exactly.
--------------------------
""".strip()


def call_ollama(prompt: str, temperature: float, max_tokens: int) -> str:
    global TOTAL_CALLS
    TOTAL_CALLS += 1

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _count_sentences(text: str) -> int:
    parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(parts)


# NEW: cleanup after deletions (prevents "and," / broken fragments)
def cleanup_after_deletions(text: str) -> str:
    t = text
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\band\s*,", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bof\s+and\b", "of", t, flags=re.IGNORECASE)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s+\!", "!", t)
    t = re.sub(r"\s+\?", "?", t)
    return t.strip()


def validate_text(text: str, selected_skills: Dict[str, float]) -> None:
    if not text or len(text.strip()) < 30:
        raise ValidationError("text_invalid", "", "Text is empty/too short")

    low = text.lower()

    banned_substrings = [
        "```", "import ", "def ", "class ", "print(", "json", "python script",
        "please provide", "based on the input json", "here is", "i'll", "i will",
        "corrected paragraph", "output json", "```python",
    ]
    if any(b in low for b in banned_substrings):
        raise ValidationError("text_invalid", "", "Meta/code/JSON detected in output")

    if any(ch in text for ch in ["{", "}", "[", "]"]):
        raise ValidationError("text_invalid", "", "Braces/brackets detected (likely JSON/code)")

    n_sent = _count_sentences(text)
    if not (4 <= n_sent <= 6):
        raise ValidationError("text_invalid", "", f"Bad sentence count: {n_sent} (expected 4–6)")

    # Selected skills explicit/implicit rules
    for skill_name, label in selected_skills.items():
        canon = skill_name.lower()
        aliases = [a.lower() for a in SKILL_ALIASES.get(skill_name, {}).get("aliases", [])]

        if label == 1.0:
            if not (canon in low or any(a in low for a in aliases)):
                raise ValidationError("explicit_missing", skill_name, f"Explicit skill missing: {skill_name}")

        elif label == 0.5:
            if canon in low or any(a in low for a in aliases):
                raise ValidationError("implicit_leaked", skill_name, f"Implicit skill leaked: {skill_name}")

    selected_set = set(selected_skills.keys())

    # Allow CI/CD-like phrasing when certain tools are selected (otherwise too strict)
    ALLOW_CICD_HINT_IF_SELECTED = {"GitHub Actions", "Azure DevOps", "Jenkins", "GitLab CI"}

    # Reject explicit mentions of NON-selected skills (canonical or aliases)
    for other_skill, info in SKILL_ALIASES.items():
        if other_skill in selected_set:
            continue

        candidates = [other_skill] + info.get("aliases", [])
        for cand in candidates:
            if not cand:
                continue

            token = cand.strip().lower()
            if not token:
                continue

            if re.fullmatch(r"[a-z]+", token) and len(token) <= 2:
                continue

            esc = re.escape(token)

            if re.fullmatch(r"[a-z0-9]+", token) and len(token) >= 3:
                pat = rf"\b{esc}\b"
            else:
                pat = rf"(?<!\w){esc}(?!\w)"

            if re.search(pat, low):
                raise ValidationError(
                    "nonselected_explicit",
                    other_skill,
                    f"Mentions non-selected skill: {other_skill}"
                )

    # NEW: Restrict nonselected_implied checks to a small high-signal set
    STRICT_NONSELECTED_IMPLIED = {"CI/CD", "GitOps", "MLflow", "Machine Learning"}

    for hinted_skill, pats in IMPLICIT_HINT_PATTERNS.items():
        if hinted_skill not in STRICT_NONSELECTED_IMPLIED:
            continue
        if hinted_skill in selected_set:
            continue

        if hinted_skill == "CI/CD" and (selected_set & ALLOW_CICD_HINT_IF_SELECTED):
            continue

        for pat in pats:
            if re.search(pat, low):
                raise ValidationError(
                    "nonselected_implied",
                    hinted_skill,
                    f"Implies non-selected skill: {hinted_skill}"
                )


def pick_random_skills(k_min: int = 3, k_max: int = 6) -> Dict[str, float]:
    k = random.randint(k_min, k_max)
    selected_names = random.sample(GLOBAL_SKILL_VECTOR, k)

    skills_with_labels = {}
    for name in selected_names:
        category = SKILL_ALIASES.get(name, {}).get("category", "")

        # Layer 1: Programming languages cannot be implicit
        if category == "programming_language":
            skills_with_labels[name] = 1.0
            continue

        # Layer 2: Only allow implicit for approved skills
        if name not in IMPLICIT_ALLOWED:
            skills_with_labels[name] = 1.0
        else:
            skills_with_labels[name] = 1.0 if random.random() < P_EXPLICIT else 0.5

    # if CI/CD tool is selected, add CI/CD as implicit
    CICD_TOOLS = {"GitHub Actions", "Azure DevOps", "Jenkins", "GitLab CI"}
    if (set(skills_with_labels.keys()) & CICD_TOOLS) and ("CI/CD" not in skills_with_labels):
        skills_with_labels["CI/CD"] = 0.5

        # keep size similar: remove one other non-language skill (not CI/CD)
        for n in list(skills_with_labels.keys()):
            if n != "CI/CD" and SKILL_ALIASES.get(n, {}).get("category", "") != "programming_language":
                del skills_with_labels[n]
                break

    # ensure mix (but ONLY within allowed implicit skills)
    vals = list(skills_with_labels.values())
    if all(v == 1.0 for v in vals) and len(vals) > 1:
        for n in skills_with_labels:
            cat = SKILL_ALIASES.get(n, {}).get("category", "")
            if cat != "programming_language" and n in IMPLICIT_ALLOWED:
                skills_with_labels[n] = 0.5
                break

    return skills_with_labels


def force_add_missing_explicit(text: str, missing_skill: str) -> str:
    t = text.strip()
    if not t:
        return f"I used {missing_skill}."

    m = re.search(r"^(.*?)([.!?])\s*$", t)
    if not m:
        return (t + f", using {missing_skill} in day-to-day development tasks.").strip()

    body, punct = m.group(1), m.group(2)

    if missing_skill.lower() in t.lower():
        return t

    injected = f"{body}, using {missing_skill} in day-to-day development tasks{punct}"
    injected = re.sub(r"\s{2,}", " ", injected).strip()
    return injected


def _regex_escape_list(items: List[str]) -> List[str]:
    return [re.escape(x) for x in items if x]


def sanitize_implicit_leak(text: str, leaked_skill: str) -> str:
    details = SKILL_ALIASES.get(leaked_skill, {})
    aliases = details.get("aliases", [])
    cat = details.get("category", "generic")

    category_repl = {
        "testing": "automated UI test suites and regression checks",
        "query_language": "relational queries and joins for reporting needs",
        "database": "managed relational databases with schema and indexing work",
        "cloud_platform": "cloud-based infrastructure for compute, storage, and networking",
        "cloud_service": "managed cloud components for compute, storage, and messaging",
        "devops_tool": "deployment automation workflows and environment promotion steps",
        "backend_framework": "backend web services and API development",
        "frontend_framework": "frontend UI components and state management",
        "programming_language": "production code in a general-purpose programming language",
        "ml_framework": "model training and evaluation workflows",
        "ml_library": "data preparation and model experimentation",
        "data_engineering": "batch/stream processing and ingestion workflows",
        "monitoring_logging": "dashboards, alerts, and log-based troubleshooting",
        "ci_cd_tool": "build-and-deploy workflows with automated tests and gated releases",
    }

    repl = IMPLICIT_REPLACEMENTS.get(leaked_skill) or category_repl.get(cat, "production-grade engineering work")

    candidates = [leaked_skill] + aliases
    patterns = _regex_escape_list(candidates)
    if not patterns:
        return text

    for p in sorted(patterns, key=len, reverse=True):
        text = re.sub(rf"(?i)\b{p}\b", repl, text)

    return text


def sanitize_invalid_associations(text: str) -> str:
    languages = [
        r"\.NET", "Python", "Java", r"C\+\+", r"C#", "Ruby", "Go", "Kotlin", "Swift",
        "Scala", "R", "JavaScript", "TypeScript", "C", "Bash"
    ]
    lang_pat = "|".join(languages)

    patterns = [
        rf"(?i)(managed relational database service[^.]*?)\s+using\s+({lang_pat})\b",
        rf"(?i)(automated backups[^.]*?)\s+using\s+({lang_pat})\b",
        rf"(?i)(failover[^.]*?)\s+using\s+({lang_pat})\b",
        rf"(?i)(read replicas[^.]*?)\s+using\s+({lang_pat})\b",
    ]
    for p in patterns:
        text = re.sub(p, r"\1", text)
    return text


def sanitize_placeholders(text: str) -> str:
    repls = {
        r'(?i)\ba relevant tool\b': "",
        r'(?i)\brelevant tool\b': "",
        r'(?i)\ba production tool\b': "",
        r'(?i)\ba cloud service\b': "",
        r'(?i)\ba CI/CD tool\b': "",
    }
    for pat, rep in repls.items():
        text = re.sub(pat, rep, text)

    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def sanitize_nonselected_hint(text: str, hinted_skill: str) -> str:
    out = text

    if hinted_skill == "CI/CD":
        out = re.sub(r"(?i)\bci\s*/\s*cd\b", "", out)
        out = re.sub(r"(?i)\bdeployment pipeline\b", "", out)
        out = re.sub(r"(?i)\bautomated test stages?\b", "", out)
        out = re.sub(r"(?i)\bdeployment gates?\b", "", out)
        out = re.sub(r"(?i)\brollback\b", "", out)

    pats = IMPLICIT_HINT_PATTERNS.get(hinted_skill, [])
    for pat in pats:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)

    out = cleanup_after_deletions(out)
    return out


def sanitize_nonselected_explicit(text: str, other_skill: str) -> str:
    out = text

    info = SKILL_ALIASES.get(other_skill, {})
    candidates = [other_skill] + info.get("aliases", [])

    for cand in sorted(candidates, key=lambda x: len(x or ""), reverse=True):
        if not cand:
            continue
        token = cand.strip()
        if not token:
            continue

        lowtok = token.lower()
        if re.fullmatch(r"[a-z]+", lowtok) and len(lowtok) <= 2:
            continue

        esc = re.escape(token)

        if re.fullmatch(r"[a-z0-9]+", lowtok) and len(lowtok) >= 3:
            pat = rf"(?i)\b{esc}\b"
        else:
            pat = rf"(?i)(?<!\w){esc}(?!\w)"

        out = re.sub(pat, "", out)

    if other_skill == "CI/CD":
        out = sanitize_nonselected_hint(out, "CI/CD")

    out = cleanup_after_deletions(out)
    return out


def enforce_no_as_a_opening(text: str, allow_prob: float = 0.1) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if stripped.lower().startswith("as a"):
        if random.random() > allow_prob:
            text = re.sub(r"(?i)^\s*as a[^,.]*[,.]\s*", "", text).strip()
    return text


def build_fix_prompt(original_text: str, selected_skills: Dict[str, float]) -> str:
    skills_info = build_skills_info(selected_skills)
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)

    return f"""
IMPORTANT OVERRIDE:
- Return ONLY the corrected paragraph (plain text).
- No JSON. No code. No markdown. No headings.
- Do NOT include meta text like: "Here is", "I'll", "I will", "Please provide", "Corrected paragraph".

LABEL RULES:
- If label is 1.0 (EXPLICIT): MUST mention canonical skill name OR one alias verbatim.
- If label is 0.5 (IMPLICIT): MUST NOT mention canonical name NOR any alias verbatim; imply via responsibilities.

Do NOT add skills beyond the list.
Keep it 4–6 sentences, one paragraph.

skills_info:
{skills_block}

Original paragraph:
{original_text}
""".strip()


def _post_process(text: str) -> str:
    text = sanitize_invalid_associations(text)
    text = sanitize_placeholders(text)
    text = enforce_no_as_a_opening(text, allow_prob=0.1)
    text = cleanup_after_deletions(text)
    return text


def generate_sample(templates: List[str]) -> Optional[Dict[str, Any]]:
    selected_skills = pick_random_skills()
    template = random.choice(templates)
    prompt = generate_prompt(selected_skills, template)

    last_err: Optional[Exception] = None
    text = ""

    # 1) initial generation attempts
    for _ in range(MAX_GEN_RETRIES):
        try:
            text = call_ollama(prompt, TEMPERATURE_GEN, MAX_TOKENS_GEN)
            text = _post_process(text)

            validate_text(text, selected_skills)
            STATS["pass_initial"] += 1
            return {"job_description": text, "skills": selected_skills}

        except ValidationError as ve:
            last_err = ve
        except Exception as e:
            last_err = e

    # 2) auto-repair loop
    repaired_text = text

    for _ in range(3):
        try:
            repaired_text = _post_process(repaired_text)

            validate_text(repaired_text, selected_skills)
            STATS["pass_repair"] += 1
            return {"job_description": repaired_text, "skills": selected_skills}

        except ValidationError as ve:
            if ve.kind == "explicit_missing" and ve.skill:
                STATS["repair_add_explicit"] += 1
                repaired_text = force_add_missing_explicit(repaired_text, ve.skill)
                repaired_text = cleanup_after_deletions(repaired_text)
                continue

            if ve.kind == "implicit_leaked" and ve.skill:
                STATS["repair_sanitize_implicit"] += 1
                repaired_text = sanitize_implicit_leak(repaired_text, ve.skill)
                repaired_text = cleanup_after_deletions(repaired_text)
                continue

            if ve.kind == "nonselected_explicit" and ve.skill:
                STATS["repair_remove_nonselected_explicit"] += 1
                repaired_text = sanitize_nonselected_explicit(repaired_text, ve.skill)
                continue

            if ve.kind == "nonselected_implied" and ve.skill:
                STATS["repair_remove_nonselected_hint"] += 1
                repaired_text = sanitize_nonselected_hint(repaired_text, ve.skill)
                continue

            last_err = ve
            break

        except Exception as e:
            last_err = e
            break

    # 3) LLM fix passes + repairs (NEW)
    fix_prompt = build_fix_prompt(repaired_text, selected_skills)
    fixed_text = repaired_text

    for _ in range(MAX_FIX_RETRIES):
        try:
            fixed_text = call_ollama(fix_prompt, TEMPERATURE_FIX, MAX_TOKENS_FIX)
            fixed_text = _post_process(fixed_text)

            validate_text(fixed_text, selected_skills)
            STATS["pass_fix_llm"] += 1
            return {"job_description": fixed_text, "skills": selected_skills}

        except ValidationError as ve:
            last_err = ve

            # NEW: try local repairs during fix loop too
            if ve.kind == "explicit_missing" and ve.skill:
                STATS["fix_add_explicit"] += 1
                fixed_text = force_add_missing_explicit(fixed_text, ve.skill)
                fixed_text = cleanup_after_deletions(fixed_text)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue

            if ve.kind == "implicit_leaked" and ve.skill:
                STATS["fix_sanitize_implicit"] += 1
                fixed_text = sanitize_implicit_leak(fixed_text, ve.skill)
                fixed_text = cleanup_after_deletions(fixed_text)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue

            if ve.kind == "nonselected_explicit" and ve.skill:
                STATS["fix_remove_nonselected_explicit"] += 1
                fixed_text = sanitize_nonselected_explicit(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue

            if ve.kind == "nonselected_implied" and ve.skill:
                STATS["fix_remove_nonselected_hint"] += 1
                fixed_text = sanitize_nonselected_hint(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue

        except Exception as e:
            last_err = e

    # final skip
    msg = str(last_err) if last_err else "unknown"
    STATS["skip"] += 1
    if isinstance(last_err, ValidationError):
        STATS[f"skip_{last_err.kind}"] += 1
    print(f"[SKIP] Failed. Reason: {msg}")
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    templates = load_prompt_templates()

    print(f"Model: {MODEL_NAME}")
    print(f"Prompts dir: {PROMPTS_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Loaded {len(templates)} templates")

    generated = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        while generated < NUM_SAMPLES:
            sample = generate_sample(templates)
            if sample:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
                generated += 1
                pbar.update(1)
        pbar.close()

    print(f"Done. Wrote {generated} samples to {OUTPUT_FILE}")

    print("\n--- STATS ---")
    print(f"Total model calls: {TOTAL_CALLS}")
    print(f"Success rate: {generated}/{TOTAL_CALLS} = {generated / max(1, TOTAL_CALLS):.2%}")
    for k, v in STATS.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    random.seed(42)
    main()
