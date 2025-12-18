# skills/implicitCueBank.py
# Hybrid evidence system for IMPLICIT (0.5):
# - Prefer skill-specific evidence phrases (e.g., NLP, MLOps) when available.
# - Otherwise use category evidence phrases.
# - Evidence phrases must NOT include canonical skill names or aliases (no leak),
#   and should avoid naming other skills from your global vector.

from __future__ import annotations

import random
import re
from typing import Dict, List, Any, Optional

# -----------------------------
# IMPLICIT eligibility policy
# -----------------------------

# Categories where category-level evidence is OK (not vendor/tool-specific).
SAFE_CATEGORY_IMPLICIT = {
    "security",
    "testing",
    "monitoring_logging",
    "systems_design",
    "architecture",
    "methodology",
    "project_management",
    "version_control",
    "ml_concept",
}

# Categories that are tool/vendor-heavy and tend to cause hallucinations if we allow
# category evidence (e.g., choosing MySQL vs PostgreSQL, PyTorch vs TF, AWS vs GCP).
TOOL_LIKE_CATEGORIES = {
    "database",
    "data_warehouse",
    "cloud_platform",
    "ml_framework",
    "ml_library",
    "security_tool",
    "ci_cd_tool",
    "devops_tool",
    "data_engineering",
}

# Minimum number of skill-specific evidence phrases to consider a tool/vendor skill implicit-eligible.
MIN_SKILL_EVIDENCE_PHRASES = 6


# -----------------------------
# Variety helpers (for "implicit_cues" text generation)
# -----------------------------
METRICS = [
    "reduced p95 latency", "reduced p99 latency", "reduced error rate", "improved uptime",
    "improved MTTR", "improved throughput", "reduced deployment time", "reduced regression rate",
    "reduced cost", "reduced query time", "reduced memory footprint", "improved CPU utilization",
    "improved reliability", "improved auditability", "improved traceability", "improved data quality",
    "improved release confidence", "improved scalability", "improved security posture",
]

SCALE_CONTEXT = [
    "during peak traffic", "under bursty workloads", "at high concurrency", "across multiple regions",
    "with multi-tenant constraints", "under strict compliance requirements", "in a production environment",
    "with zero-downtime expectations", "under tight SLAs", "for latency-sensitive workloads",
]

CONSTRAINTS = [
    "with strict SLAs", "with limited maintenance windows", "with backwards compatibility requirements",
    "while minimizing downtime", "while avoiding breaking changes", "while keeping costs predictable",
    "while meeting audit requirements", "while handling partial failures", "while preserving data consistency",
]

IMPACT_FRAMES = [
    "to {metric} {scale}",
    "resulting in {metric} {scale}",
    "which helped {metric} {scale}",
    "to ensure {metric} {scale}",
]

EVIDENCE_FRAMES = [
    "Implemented {action} around {object} {impact}.",
    "Designed {object} and applied {action} {impact}.",
    "Hardened {object} by {action} {impact}.",
    "Automated {object} via {action} {impact}.",
    "Operationalized {object} with {action} {impact}.",
    "Improved {object} using {action} {impact}.",
    "Introduced {action} for {object} {impact}.",
    "Standardized {object} by applying {action} {impact}.",
    "Refined {object} through {action} {impact}.",
    "Built {object} and enforced {action} {impact}.",
    "Validated {object} by {action} {impact}.",
    "Scaled {object} using {action} {impact}.",
]

STYLE_HINTS = [
    "Vary sentence openings; avoid repeating the same structure.",
    "Use a mix of short and long sentences; keep it natural and professional.",
    "Write in a concise engineering tone; avoid buzzwords and clichés.",
    "Include one sentence about reliability constraints or tradeoffs.",
    "Use concrete actions and outcomes; avoid vague claims like 'worked on' or 'helped with'.",
    "Prefer measurable language (latency, throughput, error rate, uptime) when relevant.",
    "Avoid repeating the same verb across consecutive sentences.",
    "Mention one failure mode and how you mitigated it (timeouts, retries, rollback, rate limiting).",
    "Keep the paragraph coherent: same project/team context across all sentences.",
]

# -----------------------------
# Category cue bank (used only to generate "implicit_cues" variety text)
# -----------------------------
CATEGORY_CUES: Dict[str, Dict[str, List[str]]] = {
    "security": {
        "actions": [
            "claim-based access checks", "least-privilege access enforcement",
            "input validation and output encoding", "rate limiting and abuse prevention",
            "audit logging for sensitive actions", "secure default configuration",
            "token expiry controls", "scope-based authorization",
        ],
        "objects": [
            "protected endpoints", "authorization rules", "access tokens",
            "audit trails", "edge policies", "compliance constraints",
        ],
        "outcomes": [
            "reduced security risk", "improved compliance", "improved auditing",
            "reduced abuse incidents",
        ],
    },
    "security_tool": {
        "actions": [
            "centralized secret storage", "runtime secret injection", "credential rotation",
            "access policy enforcement", "audit logging of secret access",
            "eliminated plaintext secrets", "scoped permissions for secret retrieval",
        ],
        "objects": [
            "secret store", "credential rotation", "access policies",
            "environment-scoped secrets", "audit logs",
        ],
        "outcomes": [
            "reduced leaks", "improved compliance", "reduced incident risk",
        ],
    },
    "testing": {
        "actions": [
            "automated validation of endpoints", "regression coverage expansion",
            "mocking external dependencies", "status/payload assertions",
            "negative test cases", "test flakiness reduction",
        ],
        "objects": [
            "test suites", "endpoint validation", "integration workflows",
            "mocked services", "test environments",
        ],
        "outcomes": [
            "reduced regressions", "improved confidence", "caught issues earlier",
        ],
    },
    "monitoring_logging": {
        "actions": [
            "added metrics and alerting", "built dashboards", "implemented structured logging",
            "correlated logs with traces", "added SLOs and error budgets",
            "improved signal-to-noise in alerts", "created runbooks",
        ],
        "objects": [
            "service metrics", "alert rules", "dashboards", "structured logs",
            "distributed traces", "runbooks", "latency percentiles",
        ],
        "outcomes": [
            "reduced MTTR", "improved observability", "caught issues earlier",
            "faster debugging",
        ],
    },
    "systems_design": {
        "actions": [
            "timeouts and retry policies", "idempotency enforcement",
            "backoff and jitter strategies", "circuit breaker patterns",
            "graceful degradation", "rate limiting and load shedding",
            "dead-letter handling", "contract testing across services",
        ],
        "objects": [
            "service-to-service communication", "multi-service transaction workflows",
            "asynchronous processing pipelines", "partial failure scenarios",
            "message-driven workflows", "idempotent write operations",
        ],
        "outcomes": [
            "reduced blast radius", "improved resilience", "fewer cascading failures",
            "lower tail latency", "better incident triage",
        ],
    },
    "architecture": {
        "actions": [
            "consistent request/response semantics", "error handling conventions",
            "endpoint versioning rules", "schema validation on ingress",
            "standardized pagination and filtering", "idempotent update semantics",
            "resource-oriented URL design", "status code normalization",
        ],
        "objects": [
            "resource-oriented endpoints", "versioned routes", "request validation rules",
            "response schemas", "pagination cursors", "error envelopes",
        ],
        "outcomes": [
            "fewer integration issues", "more stable client behavior",
            "safer evolution of endpoints",
        ],
    },
    "infrastructure": {
        "actions": [
            "health checks and failover routing", "capacity planning", "autoscaling policies",
            "traffic shaping", "caching strategy refinement", "connection pool tuning",
            "graceful shutdown handling",
        ],
        "objects": [
            "traffic distribution", "service capacity", "routing rules",
            "cache layers", "connection pools", "health probes",
        ],
        "outcomes": [
            "reduced latency", "improved uptime", "fewer overload incidents",
        ],
    },
    "cloud_platform": {
        "actions": [
            "provisioned managed compute and storage", "configured identity permissions",
            "set up alarms and dashboards", "enabled automated scaling",
            "hardened network access rules", "implemented backup and restore procedures",
            "configured encryption at rest and in transit", "set up cost controls and budgets",
        ],
        "objects": [
            "managed compute", "storage resources", "identity permissions", "alarm rules",
            "scaling policies", "network boundaries", "backup schedules", "encryption settings",
        ],
        "outcomes": [
            "improved uptime", "handled traffic spikes", "reduced operational toil",
        ],
    },
    "ci_cd_tool": {
        "actions": [
            "gated builds with automated checks", "staged rollouts", "artifact versioning",
            "release tagging conventions", "rollback-ready releases",
            "promotion between environments", "deployment approvals",
        ],
        "objects": [
            "build pipelines", "release workflows", "deployment stages",
            "artifact publishing", "environment promotions",
        ],
        "outcomes": [
            "reduced regressions", "improved release confidence", "faster delivery",
        ],
    },
    "devops_tool": {
        "actions": [
            "standardized runtime environments", "automated provisioning",
            "deployment automation", "runtime health checks",
            "resource limit tuning", "service discovery configuration",
        ],
        "objects": [
            "deployment workflows", "runtime configuration", "infrastructure state",
            "health probes", "resource quotas",
        ],
        "outcomes": [
            "reduced operational toil", "improved repeatability",
        ],
    },
    "data_engineering": {
        "actions": [
            "handled backfills", "implemented retries and idempotency",
            "added data quality checks", "incremental loads", "deduplication logic",
            "schema validation and contract checks",
        ],
        "objects": [
            "pipelines", "batch jobs", "workflow dependencies",
            "validation checks", "incremental loads",
        ],
        "outcomes": [
            "improved reliability", "reduced failures", "better auditability",
        ],
    },
    "data_warehouse": {
        "actions": [
            "optimized analytical queries", "implemented partitioning and clustering",
            "managed transformations and snapshots", "ensured reproducibility of reporting",
            "controlled compute usage for analytical workloads",
        ],
        "objects": [
            "analytics tables", "reporting layers", "historical snapshots",
            "partitioned datasets", "query workloads",
        ],
        "outcomes": [
            "improved reporting performance", "reduced costs",
        ],
    },
    "database": {
        "actions": [
            "index tuning", "query optimization",
            "migration management", "transaction isolation considerations",
            "connection pool tuning", "backup and restore procedures",
        ],
        "objects": [
            "relational tables", "indexes", "query plans", "migrations",
            "transactions", "backup schedules", "connection pools",
        ],
        "outcomes": [
            "reduced query time", "reduced timeouts",
        ],
    },
    "methodology": {
        "actions": [
            "planned work in sprints", "refined backlog items",
            "estimated tasks", "ran retrospectives",
        ],
        "objects": [
            "user stories", "sprint boards", "team ceremonies",
        ],
        "outcomes": [
            "improved predictability", "faster feedback cycles",
        ],
    },
    "project_management": {
        "actions": [
            "tracked milestones and risks", "prioritized work based on impact",
            "defined acceptance criteria", "managed stakeholder updates",
        ],
        "objects": [
            "milestones", "risk logs", "acceptance criteria",
        ],
        "outcomes": [
            "reduced delivery risk",
        ],
    },
    "version_control": {
        "actions": [
            "used feature branches and pull requests", "reviewed changes for quality and safety",
            "managed rollbacks via version history", "resolved merge conflicts systematically",
            "tagged releases and hotfixes",
        ],
        "objects": [
            "pull requests", "branching strategy", "release tags",
            "commit history",
        ],
        "outcomes": [
            "reduced integration risk", "safer releases",
        ],
    },
    "ml_concept": {
        "actions": [
            "baseline evaluation on a held-out set",
            "tracked precision/recall tradeoffs",
            "checked feature leakage and data drift symptoms",
            "calibrated thresholds to balance false positives/negatives",
        ],
        "objects": [
            "model evaluation", "baseline experiments", "feature pipelines",
            "classification workflows",
        ],
        "outcomes": [
            "improved model quality", "reduced overfitting",
        ],
    },
    "ml_framework": {
        "actions": [
            "ran hyperparameter tuning", "built reproducible training/evaluation loops",
            "tracked experiments across runs", "packaged preprocessing with training steps",
            "compared baselines across folds",
        ],
        "objects": [
            "model training workflows", "evaluation scripts",
            "training runs", "preprocessing+model pipelines",
        ],
        "outcomes": [
            "more consistent results", "better reproducibility",
        ],
    },
    "ml_library": {
        "actions": [
            "vectorized numerical operations", "used broadcasting rules",
            "performed matrix operations", "optimized array computations",
        ],
        "objects": [
            "array-based computations", "numerical routines",
            "linear algebra workflows",
        ],
        "outcomes": [
            "reduced runtime", "reduced memory usage",
        ],
    },
}

# -----------------------------
# REQUIRED evidence phrases
# - Used for label 0.5 validation.
# - Must NOT include canonical/aliases.
# - Skill-specific evidence overrides category evidence when present.
# -----------------------------

# Skill-specific evidence phrases (best for ambiguous skills like NLP / MLOps).
SKILL_EVIDENCE_PHRASES: Dict[str, List[str]] = {
    # NLP (DO NOT use "nlp" or "natural language processing")
   "NLP": [
        "tokenization",
        "text normalization",
        "stopword removal",
        "stemming",
        "lemmatization",
        "named entity recognition",
        "part-of-speech tagging",
        "sequence labeling",
        "n-gram features",
        "text classification",
        "sentence segmentation",
        "spelling normalization",
        "regex-based preprocessing",
        "keyword extraction",
    ],

    # MLOps (DO NOT use "mlops" / "ml ops")
    "MLOps": [
        "model versioning",
        "model registry",
        "staged promotion",
        "drift monitoring",
        "retraining triggers",
        "reproducible training artifacts",
        "automated validation gates",
        "shadow deployment",
        "canary rollout",
        "offline and online metric parity",
        "feature store",
        "data drift detection",
        "model monitoring",
        "rollback to a previous model version",
    ],

    # Optional extras (safe if you want later; kept minimal)
    # "LLMs": [...],
    # "Transformers": [...],
    # dbt (avoid 'dbt' / 'data build tool')
    "dbt": [
        "incremental models",
        "model refactoring",
        "source freshness tests",
        "data lineage documentation",
        "staging models",
        "schema tests",
        "seed files",
        "environment-specific targets",
    ],

    # Postman (avoid 'postman')
    "Postman": [
        "API collections",
        "collection runner",
        "pre-request scripts",
        "environment variables for requests",
        "response assertions",
        "mock servers for APIs",
    ],

    # Jenkins (avoid 'jenkins')
    "Jenkins": [
        "pipeline stages",
        "build agents",
        "artifact publishing",
        "job orchestration",
        "build logs triage",
        "release tagging conventions",
    ],

    # Snowflake (avoid 'snowflake')
    "Snowflake": [
        "virtual warehouses",
        "separate compute and storage",
        "zero-copy cloning",
        "time travel for tables",
        "auto-suspend warehouses",
    ],

    # BigQuery (avoid 'bigquery' / 'bq')
    "BigQuery": [
        "partitioned tables",
        "clustering keys",
        "slot reservations",
        "federated queries",
    ],

    # scikit-learn (avoid 'scikit-learn' / 'sklearn')
    "scikit-learn": [
        "grid search",
        "pipeline fit/predict",
        "classification metrics",
        "cross-validation folds",
    ],
    "LLMs": [
        "prompt templating",
        "context-window constraints",
        "few-shot examples",
        "retrieval-augmented prompts",
    ],
    "Transformers": [
    "attention mechanism",
    "encoder-decoder setup",
    "sequence-to-sequence training",
    "token-level embeddings",
    ],
    "PyTorch": [
        "autograd",
        "training loop",
        "gradient clipping",
        "checkpointing",
    ],
    "NumPy": [
        "vectorized operations",
        "broadcasting rules",
        "matrix operations",
        "linear algebra routines",
    ],

}

# Category evidence phrases (fallback when no skill-specific list exists)
CATEGORY_EVIDENCE_PHRASES: Dict[str, List[str]] = {
    "security": [
        "least-privilege access",
        "rate limiting",
        "allowlist rules",
        "denylist rules",
        "audit logs",
        "token expiration",
        "claim-based access",
        "scope-based access",
        "TLS termination",
        "mTLS between services",
        "credential rotation",
    ],
    "security_tool": [
        "runtime secret injection",
        "credential rotation",
        "eliminated plaintext secrets",
        "access policies for secret retrieval",
        "audit logs for secret access",
    ],
    "testing": [
        "status code assertions",
        "payload assertions",
        "negative test cases",
        "regression coverage",
        "mocked dependencies",
        "contract tests",
        "test flakiness",
        "test suites",
    ],
    "monitoring_logging": [
        "p95 latency",
        "p99 latency",
        "error rate alerts",
        "structured logging",
        "dashboards",
        "alert rules",
        "distributed tracing",
        "correlation IDs",
        "runbooks",
        "MTTR",
    ],
    "systems_design": [
        "timeouts and retries",
        "exponential backoff",
        "idempotent operations",
        "partial failures",
        "circuit breaker",
        "graceful degradation",
        "dead-letter queue",
        "load shedding",
    ],
    "architecture": [
        "resource-oriented endpoints",
        "pagination",
        "versioned endpoints",
        "request validation",
        "consistent status codes",
        "idempotent updates",
        "error envelope",
    ],
    "infrastructure": [
        "health checks",
        "failover routing",
        "autoscaling policies",
        "traffic shaping",
        "connection pool tuning",
        "graceful shutdown",
    ],
    "cloud_platform": [
        "managed compute",
        "object storage",
        "identity permissions",
        "encryption at rest",
        "encryption in transit",
        "budget alerts",
        "backup and restore",
        "multi-region setup",
    ],
    "ci_cd_tool": [
        "quality gates",
        "staged rollouts",
        "artifact versioning",
        "release tagging",
        "rollback mechanisms",
        "deployment approvals",
    ],
    "devops_tool": [
        "runtime configuration",
        "deployment automation",
        "resource limits",
        "log shipping",
        "health probes",
    ],
    "data_engineering": [
        "incremental loads",
        "backfills",
        "schema validation",
        "data quality checks",
        "deduplication",
        "lineage tracking",
    ],
    "data_warehouse": [
        "partitioning",
        "clustering",
        "historical snapshots",
        "columnar storage",
        "separated compute and storage",
    ],
    "database": [
        "index tuning",
        "query optimization",
        "migration management",
        "transaction isolation",
        "connection pool tuning",
        "backup and restore",
    ],
    "methodology": [
        "sprint planning",
        "backlog refinement",
        "daily standup",
        "retrospectives",
        "user stories",
    ],
    "project_management": [
        "acceptance criteria",
        "risk tracking",
        "milestone planning",
        "stakeholder updates",
    ],
    "version_control": [
        "pull requests",
        "feature branches",
        "release tags",
        "code reviews",
        "merge conflict resolution",
    ],
    "ml_concept": [
        "held-out evaluation",
        "baseline experiments",
        "precision and recall tradeoffs",
        "leakage checks",
        "threshold calibration",
        "model evaluation",
    ],
    "ml_framework": [
        "hyperparameter tuning",
        "model training loop",
        "experiment tracking",
        "reproducible runs",
        "preprocessing pipeline",
        "cross-validation",
    ],
    "ml_library": [
        "vectorized operations",
        "broadcasting",
        "matrix operations",
        "linear algebra",
        "array computations",
    ],
}

# -----------------------------
# Public helpers
# -----------------------------
def style_hint(rng: random.Random) -> str:
    return rng.choice(STYLE_HINTS)

def get_category(skill_aliases: Dict[str, Any], skill_name: str) -> Optional[str]:
    return (skill_aliases.get(skill_name) or {}).get("category")

def category_has_evidence(category: Optional[str]) -> bool:
    if not category:
        return False
    if category not in SAFE_CATEGORY_IMPLICIT:
        return False
    return bool(CATEGORY_EVIDENCE_PHRASES.get(category))


def skill_has_evidence(skill_name: str) -> bool:
    phrases = SKILL_EVIDENCE_PHRASES.get(skill_name, [])
    return len(phrases) >= MIN_SKILL_EVIDENCE_PHRASES

def skill_or_category_has_evidence(skill_name: str, category: Optional[str]) -> bool:
    # For tool-like categories, allow implicit ONLY with strong skill-specific evidence.
    if category in TOOL_LIKE_CATEGORIES:
        return skill_has_evidence(skill_name)

    # For safe categories, allow either skill-specific evidence or category evidence.
    if skill_has_evidence(skill_name):
        return True
    return category_has_evidence(category)


def pick_required_evidence_phrases(
    rng: random.Random,
    skill_name: str,
    category: Optional[str],
    k_min: int = 1,
    k_max: int = 2,
) -> List[str]:
    # Prefer skill-specific evidence
    phrases = SKILL_EVIDENCE_PHRASES.get(skill_name, [])

    # For tool-like categories: do NOT fall back to category evidence.
    # If no strong skill-specific evidence -> return [] (caller should treat as not eligible).
    if category in TOOL_LIKE_CATEGORIES:
        if len(phrases) < MIN_SKILL_EVIDENCE_PHRASES:
            return []
    else:
        # Safe categories can fall back to category evidence
        if (not phrases) and category:
            phrases = CATEGORY_EVIDENCE_PHRASES.get(category, [])

    if not phrases:
        return []
    k = rng.randint(k_min, k_max)
    return rng.sample(phrases, min(k, len(phrases)))

def text_contains_any_required_phrase(text: str, required_phrases: List[str]) -> bool:
    if not required_phrases:
        return True
    low = text.lower()
    for p in required_phrases:
        if p.lower() in low:
            return True
    return False

def pick_metric(rng: random.Random) -> str:
    return rng.choice(METRICS)

def pick_scale(rng: random.Random) -> str:
    return rng.choice(SCALE_CONTEXT)

def pick_constraint(rng: random.Random) -> str:
    return rng.choice(CONSTRAINTS)

def build_impact(rng: random.Random) -> str:
    metric = pick_metric(rng)
    scale = pick_scale(rng)
    frame = rng.choice(IMPACT_FRAMES)
    return frame.format(metric=metric, scale=scale)

def build_dynamic_implicit_sentence(
    rng: random.Random,
    category: str,
    skill_name: str,
) -> str:
    cues = CATEGORY_CUES.get(category)
    impact = build_impact(rng)

    if not cues:
        fallbacks = [
            f"Implemented concrete responsibilities and outcomes {impact}.",
            f"Designed operational improvements and validation steps {impact}.",
            f"Hardened critical workflows with reliability and safety controls {impact}.",
        ]
        return rng.choice(fallbacks)

    action = rng.choice(cues["actions"])
    obj = rng.choice(cues["objects"])

    maybe_constraint = ""
    if rng.random() < 0.30:
        maybe_constraint = f" {pick_constraint(rng)}"

    frame = rng.choice(EVIDENCE_FRAMES)
    s = frame.format(action=action, object=obj, impact=impact)
    return (s.replace("  ", " ") + maybe_constraint).strip()