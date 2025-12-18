from pathlib import Path
import sys
import json
import hashlib
from collections import Counter
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import unicodedata
from itertools import combinations
import re


# CONFIG AND PATHS
def setup_paths():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    JSONL_PATH = ROOT_DIR / "data" / "synthetic_dataset.jsonl"
    OUTPUT_DIR = ROOT_DIR / "EDA"
    OUTPUT_DIR.mkdir(exist_ok=True)
    return ROOT_DIR, JSONL_PATH, OUTPUT_DIR


# LOAD GLOBAL SKILL VECTOR
def load_global_skills():
    from skills.globalVector import GLOBAL_SKILL_VECTOR
    global_set = set(GLOBAL_SKILL_VECTOR)
    return GLOBAL_SKILL_VECTOR, global_set


# ACCUMULATORS (counters & feature lists)
def init_accumulators():
    return {
        # Sanity counters
        "loaded": 0,
        "missing_required": 0,
        "bad_skills_type": 0,
        "unknown_skills": 0,
        "bad_label_values": 0,
        "duplicate_count": 0,

        # Duplicates tracking
        "seen_hashes": set(),

        # Length stats
        "text_len_chars": [],
        "text_len_words": [],
        "sentence_counts": [],

        # Per-sample label stats
        "explicit_per_sample": [],
        "implicit_per_sample": [],

        # Per-skill counts
        "explicit_counts": Counter(),
        "implicit_counts": Counter(),

        # Co-occurrence
        "pair_counts": Counter(),

        # Text Cleanliness Checks (Whitespace & Weird Characters)
        "double_space_texts": 0,
        "non_ascii_texts": 0,
        "control_char_texts": 0,
    }


# HELPERS
def count_sentences(text: str) -> int:
    # rough sentence split by . ! ?
    parts = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    return len(parts)


def has_control_chars(text: str) -> bool:
    #CONTROL/BREACK CHARACTERS
    for ch in text:
        if unicodedata.category(ch) == "Cc" and ch not in ("\n", "\t", "\r"):
            return True
    return False


# SINGLE PASS OVER DATASET (Sanity + Length + Duplicates + Cleanliness)
def process_dataset(jsonl_path: Path, global_set: set, acc: dict):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            acc["loaded"] += 1

            # Sanity checks: required fields + skills format
            if "job_description" not in obj or "skills" not in obj:
                acc["missing_required"] += 1
                continue

            text = obj["job_description"]
            skills = obj["skills"]

            if not isinstance(skills, dict):
                acc["bad_skills_type"] += 1
                continue

            # per-sample counters
            sample_explicit = 0
            sample_implicit = 0

            #count skill pairs per sample (co-occurrence)
            skill_list = sorted(skills.keys())
            for a, b in combinations(skill_list, 2):
                acc["pair_counts"][(a, b)] += 1

            # Sanity checks: skill names & label values + label counts
            for skill, label in skills.items():
                if skill not in global_set:
                    acc["unknown_skills"] += 1

                if label not in (0.5, 1.0):
                    acc["bad_label_values"] += 1

                if label == 1.0:
                    sample_explicit += 1
                    acc["explicit_counts"][skill] += 1
                elif label == 0.5:
                    sample_implicit += 1
                    acc["implicit_counts"][skill] += 1

            acc["explicit_per_sample"].append(sample_explicit)
            acc["implicit_per_sample"].append(sample_implicit)

            # Duplicate detection (exact text hash)
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in acc["seen_hashes"]:
                acc["duplicate_count"] += 1
            else:
                acc["seen_hashes"].add(h)

            # Length
            acc["text_len_chars"].append(len(text))
            acc["text_len_words"].append(len(text.split()))
            acc["sentence_counts"].append(count_sentences(text))

            #Text Cleanliness Checks (Whitespace & Weird Characters)
            #DOUBLE SPACES
            if "  " in text:
                acc["double_space_texts"] += 1

            #NON-ASCII CHARACTERS
            if any(ord(ch) > 127 for ch in text):
                acc["non_ascii_texts"] += 1

            #CONTROL/BREACK CHARACTERS
            if has_control_chars(text):
                acc["control_char_texts"] += 1


# REPORTS (PRINTS)
def print_sanity_and_length(acc: dict):
    print("Loaded records:", acc["loaded"])
    print("Missing required keys:", acc["missing_required"])
    print("Bad skills type:", acc["bad_skills_type"])
    print("Unknown skills:", acc["unknown_skills"])
    print("Bad label values:", acc["bad_label_values"])
    print("Exact duplicate job_descriptions:", acc["duplicate_count"])
    print("Unique job_descriptions:", len(acc["seen_hashes"]))

    print("=== TEXT LENGTH STATS ===")
    print("Samples:", len(acc["text_len_words"]))

    print("\nChars:")
    print("  mean:", round(stats.mean(acc["text_len_chars"]), 2))
    print("  median:", stats.median(acc["text_len_chars"]))
    print("  min/max:", min(acc["text_len_chars"]), "/", max(acc["text_len_chars"]))

    print("\nWords:")
    print("  mean:", round(stats.mean(acc["text_len_words"]), 2))
    print("  median:", stats.median(acc["text_len_words"]))
    print("  min/max:", min(acc["text_len_words"]), "/", max(acc["text_len_words"]))


def print_text_cleanliness(acc: dict):
    print("\n Text Cleanliness")
    print("Texts with double spaces:", acc["double_space_texts"])
    print("Texts with non-ASCII chars:", acc["non_ascii_texts"])
    print("Texts with control chars:", acc["control_char_texts"])


def print_label_sanity(acc: dict):
    print("\nSkill label counts (sanity)")
    print("Total explicit labels:", sum(acc["explicit_counts"].values()))
    print("Total implicit labels:", sum(acc["implicit_counts"].values()))

    print("\nper-sample label stats ")
    print("Avg explicit per sample:", round(stats.mean(acc["explicit_per_sample"]), 2))
    print("Avg implicit per sample:", round(stats.mean(acc["implicit_per_sample"]), 2))
    print("Samples with 0 implicit:", sum(1 for x in acc["implicit_per_sample"] if x == 0))


def print_top_bottom_skills(GLOBAL_SKILL_VECTOR, acc: dict, TOP_K=20):
    explicit_counts = acc["explicit_counts"]
    implicit_counts = acc["implicit_counts"]

    print("\n Top skills by label")
    print(f"\nTop {TOP_K} EXPLICIT (1.0):")
    for skill, cnt in explicit_counts.most_common(TOP_K):
        print(f"  {skill}: {cnt}")

    print(f"\nTop {TOP_K} IMPLICIT (0.5):")
    for skill, cnt in implicit_counts.most_common(TOP_K):
        print(f"  {skill}: {cnt}")

    #Bottom skills (total occurrences, including zeros)
    print(f"\nBottom {TOP_K} TOTAL (lowest occurrences):")
    total_counts = Counter()
    for s in GLOBAL_SKILL_VECTOR:
        total_counts[s] = explicit_counts.get(s, 0) + implicit_counts.get(s, 0)

    for skill, cnt in sorted(total_counts.items(), key=lambda x: x[1])[:TOP_K]:
        print(f"  {skill}: {cnt}")

    return total_counts


def print_imbalance_summary(GLOBAL_SKILL_VECTOR, total_counts: Counter):
    print("\nImbalance summary")

    # 1) zero-occurrence skills
    zero_skills = [s for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) == 0]
    print("Skills with 0 occurrences:", len(zero_skills))

    # 2) rare skills thresholds
    lt10 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 10)
    lt20 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 20)
    print("Skills with <10 occurrences:", lt10)
    print("Skills with <20 occurrences:", lt20)

    # 3) dominance: share covered by top skills
    total_labels = sum(total_counts.values()) or 1
    top10_sum = sum(cnt for _, cnt in total_counts.most_common(10))
    top20_sum = sum(cnt for _, cnt in total_counts.most_common(20))

    print("Total labeled skill occurrences:", total_labels)
    print("Top 10 skills cover:", round(100 * top10_sum / total_labels, 2), "%")
    print("Top 20 skills cover:", round(100 * top20_sum / total_labels, 2), "%")


def print_top_pairs(acc: dict, TOP_PAIRS=10):
    print(f"\nTop {TOP_PAIRS} co-occurring skill pairs")
    for (a, b), cnt in acc["pair_counts"].most_common(TOP_PAIRS):
        print(f"  ({a}, {b}): {cnt}")


def print_sentence_length(acc: dict):
    print("\nLength (Sentence Count)")
    print("Sentences - mean:", round(stats.mean(acc["sentence_counts"]), 2))
    print("Sentences - median:", stats.median(acc["sentence_counts"]))
    print("Sentences - min/max:", min(acc["sentence_counts"]), "/", max(acc["sentence_counts"]))

    in_4_6 = sum(1 for x in acc["sentence_counts"] if 4 <= x <= 6)
    too_short = sum(1 for x in acc["sentence_counts"] if x < 4)
    too_long = sum(1 for x in acc["sentence_counts"] if x > 6)

    n = len(acc["sentence_counts"]) or 1
    print("Samples with 4–6 sentences:", in_4_6, f"({in_4_6/n*100:.1f}%)")
    print("Too short (<4):", too_short, f"({too_short/n*100:.1f}%)")
    print("Too long (>6):", too_long, f"({too_long/n*100:.1f}%)")


def print_types_rates(acc: dict):
    #Types — error rates (%)
    total = acc["loaded"] if acc["loaded"] else 1

    print("\nTypes (Rates)")
    print("Missing required keys (%):", round(100 * acc["missing_required"] / total, 2))
    print("Bad skills type (%):", round(100 * acc["bad_skills_type"] / total, 2))
    print("Duplicates (%):", round(100 * acc["duplicate_count"] / total, 2))
    print("Non-ASCII (%):", round(100 * acc["non_ascii_texts"] / total, 2))
    print("Double spaces (%):", round(100 * acc["double_space_texts"] / total, 2))
    print("Control chars (%):", round(100 * acc["control_char_texts"] / total, 2))
    print("Unknown skill entries (count):", acc["unknown_skills"])
    print("Bad label entries (count):", acc["bad_label_values"])


def print_summary_percentages(acc: dict):
    # SUMMARY WITH PERCENTAGES
    total_samples = acc["loaded"] if acc["loaded"] else 1
    total_labels = (sum(acc["explicit_counts"].values()) + sum(acc["implicit_counts"].values())) or 1

    print("\n=== Summary (Percentages) ===")
    print("Missing required keys (% samples):", round(100 * acc["missing_required"] / total_samples, 2))
    print("Bad skills type (% samples):", round(100 * acc["bad_skills_type"] / total_samples, 2))
    print("Duplicates (% samples):", round(100 * acc["duplicate_count"] / total_samples, 2))

    print("Non-ASCII (% samples):", round(100 * acc["non_ascii_texts"] / total_samples, 2))
    print("Double spaces (% samples):", round(100 * acc["double_space_texts"] / total_samples, 2))
    print("Control chars (% samples):", round(100 * acc["control_char_texts"] / total_samples, 2))

    print("Explicit labels (% of labeled skills):", round(100 * sum(acc["explicit_counts"].values()) / total_labels, 2))
    print("Implicit labels (% of labeled skills):", round(100 * sum(acc["implicit_counts"].values()) / total_labels, 2))

    zero_implicit = sum(1 for x in acc["implicit_per_sample"] if x == 0)
    print("Samples with 0 implicit (% samples):", round(100 * zero_implicit / total_samples, 2))


# OUTPUTS (PLOTS)
def plot_histograms(acc: dict, out_dir: Path):
    #HISTOGRAMA: WORDS
    plt.figure()
    plt.hist(acc["text_len_words"], bins=20)
    plt.title("Job Description Length (Words)")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_words.png", dpi=200)
    plt.show()

    #HISTOGRAMA: CHARS
    plt.figure()
    plt.hist(acc["text_len_chars"], bins=20)
    plt.title("Job Description Length (Chars)")
    plt.xlabel("Characters")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "hist_chars.png", dpi=200)
    plt.show()

    print("Saved plots to:", out_dir)


# DASHBOARD HELPERS
def compute_dashboard_numbers(GLOBAL_SKILL_VECTOR, acc: dict, total_counts: Counter):
    total_samples = acc["loaded"] if acc["loaded"] else 1
    total_labels = sum(acc["explicit_counts"].values()) + sum(acc["implicit_counts"].values())
    total_labels = total_labels if total_labels else 1

    explicit_pct = 100 * sum(acc["explicit_counts"].values()) / total_labels
    implicit_pct = 100 * sum(acc["implicit_counts"].values()) / total_labels
    zero_implicit_pct = 100 * (sum(1 for x in acc["implicit_per_sample"] if x == 0) / total_samples)

    non_ascii_pct = 100 * acc["non_ascii_texts"] / total_samples
    dup_pct = 100 * acc["duplicate_count"] / total_samples

    # Top skills (keep small to avoid clutter)
    TOP_K = 5
    top_exp = acc["explicit_counts"].most_common(TOP_K)
    top_imp = acc["implicit_counts"].most_common(TOP_K)

    # Imbalance summary
    zero_skills = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) == 0)
    lt10 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 10)
    lt20 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 20)
    top10_cover = 100 * (sum(cnt for _, cnt in total_counts.most_common(10)) / (sum(total_counts.values()) or 1))

    # Length summary (sentence constraint)
    sent_mean = stats.mean(acc["sentence_counts"])
    sent_median = stats.median(acc["sentence_counts"])
    sent_min, sent_max = min(acc["sentence_counts"]), max(acc["sentence_counts"])

    words_mean = stats.mean(acc["text_len_words"])
    words_median = stats.median(acc["text_len_words"])
    words_min, words_max = min(acc["text_len_words"]), max(acc["text_len_words"])

    chars_mean = stats.mean(acc["text_len_chars"])
    chars_median = stats.median(acc["text_len_chars"])
    chars_min, chars_max = min(acc["text_len_chars"]), max(acc["text_len_chars"])

    return {
        "total_samples": total_samples,
        "total_labels_occ": sum(total_counts.values()),
        "explicit_pct": explicit_pct,
        "implicit_pct": implicit_pct,
        "zero_implicit_pct": zero_implicit_pct,
        "non_ascii_pct": non_ascii_pct,
        "dup_pct": dup_pct,
        "top_exp": top_exp,
        "top_imp": top_imp,
        "zero_skills": zero_skills,
        "lt10": lt10,
        "lt20": lt20,
        "top10_cover": top10_cover,
        "sent_mean": sent_mean,
        "sent_median": sent_median,
        "sent_min": sent_min,
        "sent_max": sent_max,
        "words_mean": words_mean,
        "words_median": words_median,
        "words_min": words_min,
        "words_max": words_max,
        "chars_mean": chars_mean,
        "chars_median": chars_median,
        "chars_min": chars_min,
        "chars_max": chars_max,
    }


def build_dashboard(GLOBAL_SKILL_VECTOR, acc: dict, total_counts: Counter, out_dir: Path):
    #Numbers for dashboard
    d = compute_dashboard_numbers(GLOBAL_SKILL_VECTOR, acc, total_counts)

    # --- Build 2x2 dashboard ---
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

    # (1) KPI boxes (top-left)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(0.0, 1.0, "EDA Summary", fontsize=16, fontweight="bold", va="top")
    ax0.text(0.0, 0.78, f"Samples: {d['total_samples']}", fontsize=14)
    ax0.text(0.0, 0.62, f"Global skills: {len(GLOBAL_SKILL_VECTOR)}", fontsize=14)
    ax0.text(0.0, 0.46, f"Total labeled skill occurrences: {d['total_labels_occ']}", fontsize=14)
    ax0.text(0.0, 0.25, f"Explicit vs Implicit: {d['explicit_pct']:.1f}% / {d['implicit_pct']:.1f}%", fontsize=12)
    ax0.text(0.0, 0.12, f"Samples with 0 implicit: {d['zero_implicit_pct']:.1f}%", fontsize=12)

    # (2) Donut: explicit vs implicit (top-right)
    ax1 = fig.add_subplot(gs[0, 1])
    vals = [sum(acc["explicit_counts"].values()), sum(acc["implicit_counts"].values())]
    labels = ["Explicit (1.0)", "Implicit (0.5)"]
    wedges, _ = ax1.pie(vals, startangle=90, wedgeprops=dict(width=0.45))
    ax1.set_title("Label Share", fontsize=12)
    ax1.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    # (3) Top skills bar charts (bottom-left)
    ax2 = fig.add_subplot(gs[1, 0])
    names = [s for s, _ in d["top_exp"]] + [s for s, _ in d["top_imp"]]
    counts = [c for _, c in d["top_exp"]] + [c for _, c in d["top_imp"]]
    colors = ["tab:blue"] * len(d["top_exp"]) + ["tab:orange"] * len(d["top_imp"])

    ax2.barh(range(len(names)), counts, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_title("Top 5 Explicit (blue) + Top 5 Implicit (orange)", fontsize=12)
    ax2.set_xlabel("Count")

    # (4) Quality + Length table-like text (bottom-right)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title("Quality (Types) + Length", fontsize=12, fontweight="bold", loc="left")

    total_samples = d["total_samples"]
    quality_lines = [
        f"Missing keys: {acc['missing_required']} ({100*acc['missing_required']/total_samples:.1f}%)",
        f"Bad skills type: {acc['bad_skills_type']} ({100*acc['bad_skills_type']/total_samples:.1f}%)",
        f"Duplicates: {acc['duplicate_count']} ({d['dup_pct']:.1f}%)",
        f"Non-ASCII: {acc['non_ascii_texts']} ({d['non_ascii_pct']:.1f}%)",
        f"Double spaces: {acc['double_space_texts']} ({100*acc['double_space_texts']/total_samples:.1f}%)",
        f"Control chars: {acc['control_char_texts']} ({100*acc['control_char_texts']/total_samples:.1f}%)",
        f"Unknown skills: {acc['unknown_skills']}",
        f"Bad label values: {acc['bad_label_values']}",
        "",
        f"Sentence count: mean {d['sent_mean']:.2f}, median {d['sent_median']:.1f}, min/max {d['sent_min']}/{d['sent_max']}",
        f"Words: mean {d['words_mean']:.1f}, median {d['words_median']:.1f}, min/max {d['words_min']}/{d['words_max']}",
        f"Chars: mean {d['chars_mean']:.1f}, median {d['chars_median']:.1f}, min/max {d['chars_min']}/{d['chars_max']}",
        "",
        f"Imbalance: zero-occ {d['zero_skills']}, <10 {d['lt10']}, <20 {d['lt20']}, Top10 cover {d['top10_cover']:.2f}%",
    ]

    ax3.text(0.0, 1.0, "\n".join(quality_lines), va="top", fontsize=11)

    # Footer note
    top_pair = acc["pair_counts"].most_common(1)
    if top_pair:
        (a, b), cnt = top_pair[0]
        fig.text(0.01, 0.01, f"Co-occurrence: Top pair ({a}, {b}) appears {cnt}/{total_samples}.", fontsize=10)
    else:
        fig.text(0.01, 0.01, "Co-occurrence: no pairs found.", fontsize=10)

    out_path = out_dir / "eda_dashboard.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("Saved dashboard to:", out_path)


# MAIN
def main():
    ROOT_DIR, JSONL_PATH, OUTPUT_DIR = setup_paths()
    GLOBAL_SKILL_VECTOR, global_set = load_global_skills()

    acc = init_accumulators()

    # SINGLE PASS OVER DATASET
    process_dataset(JSONL_PATH, global_set, acc)

    # REPORTS
    print_sanity_and_length(acc)
    plot_histograms(acc, OUTPUT_DIR)
    print_text_cleanliness(acc)

    print_label_sanity(acc)

    total_counts = print_top_bottom_skills(GLOBAL_SKILL_VECTOR, acc, TOP_K=20)
    print_imbalance_summary(GLOBAL_SKILL_VECTOR, total_counts)
    print_top_pairs(acc, TOP_PAIRS=10)

    print_sentence_length(acc)
    print_types_rates(acc)
    print_summary_percentages(acc)

    # DASHBOARD
    build_dashboard(GLOBAL_SKILL_VECTOR, acc, total_counts, OUTPUT_DIR)


if __name__ == "__main__":
    main()
