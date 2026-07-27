"""
AgentDDx Evaluation Suite for MedQA-USMLE Benchmark
====================================================
3-condition ablation study with proper statistical testing.

KEY FIX: Option shuffling to eliminate position bias.
The model had severe A-bias (45% of predictions = A vs 28% true rate).
We now shuffle option order before each query and map back, so the model
never sees options in a fixed A/B/C/D order.

Conditions:
  A) LLM-only           — Llama 3.3 70B, zero-shot MCQ
  B) LLM + PubMed       — Llama + raw PubMed retrieval (no filtering)
  C) Full AgentDDx      — Entity extraction + MeSH queries + relevance-filtered
                           PubMed + structured reasoning agent

Run:
  python evaluate.py --n 1273 --condition llm_only
  python evaluate.py --n 1273 --condition llm_pubmed
  python evaluate.py --n 1273 --condition full_system
  python evaluate.py --n 1273  # all three

Output (separate file per condition):
  eval_llm_only.json
  eval_llm_pubmed.json
  eval_full_system.json
  eval_report_<condition>.json
"""
import os, sys, json, time, argparse, random, re, math
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

# NOTE: agent/LLM imports are deferred into _lazy_imports() rather than done at
# module level. Reason: --merge only crunches existing JSON and makes zero API
# calls, so it must not require the openai/biopython stack to be installed.
extract_entities = None
fetch_evidence_for_candidates = None
_search_pubmed = None
run_reasoning = None
ClinicalCase = None
config = None
llm_call = None


def _lazy_imports():
    """Import the LLM/agent stack. Called only when actually running an eval."""
    global extract_entities, fetch_evidence_for_candidates, _search_pubmed
    global run_reasoning, ClinicalCase, config, llm_call

    from agents.entity_extractor import extract_entities as _ee
    from agents.pubmed_retrieval import (
        fetch_evidence_for_candidates as _fe,
        _search_pubmed as _sp,
    )
    from agents.reasoning_agent import run_reasoning as _rr
    from models import ClinicalCase as _cc
    import config as _cfg
    from utils import llm_call as _lc

    extract_entities = _ee
    fetch_evidence_for_candidates = _fe
    _search_pubmed = _sp
    run_reasoning = _rr
    ClinicalCase = _cc
    config = _cfg
    llm_call = _lc

# ── Output location ───────────────────────────────────────────
# All checkpoints and reports are written under results/ so the repository
# root stays clean. Override with AGENTDDX_RESULTS_DIR for scratch runs.
RESULTS_DIR = os.environ.get(
    "AGENTDDX_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)


def results_path(name: str) -> str:
    """Resolve an artifact name to its path under RESULTS_DIR."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


VALID_ANSWERS = {"A", "B", "C", "D"}
OPTION_LETTERS = ["A", "B", "C", "D"]

# ── Reproducibility ───────────────────────────────────────────
# Fixed seed so option shuffling is deterministic across runs.
# A reviewer re-running this command gets byte-identical shuffles.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Bump this whenever answer-extraction changes. Checkpoints record it, and a
# resume across a mismatch is REFUSED (starting fresh) so a run can never mix
# answers scored by different parser versions.
PARSER_VERSION = "fixed-2026-07-colon-boundary"

# ── Equalized hyperparameters (ablation controls) ─────────────
# These are IDENTICAL across all three conditions so that the only
# independent variable is retrieval quality — not token budget,
# temperature, or persona framing.
EVAL_MAX_TOKENS  = 900
EVAL_TEMPERATURE = 0.1

# Shared persona + task framing. Every condition uses this verbatim.
BASE_PERSONA = "You are a medical expert taking a clinical exam."
BASE_INSTRUCTION = (
    "Think through this step by step, then state your final answer.\n"
    "End your response with exactly: Answer: <letter>"
)


# ── Option Shuffling (position bias fix) ─────────────────────

def _shuffle_options(options: dict) -> tuple[dict, dict]:
    """Shuffle MCQ option order to eliminate position bias.

    Args:
        options: Original options dict, e.g. {"A": "Aspirin", "B": "Ibuprofen", ...}

    Returns:
        (shuffled_options, reverse_map)
        - shuffled_options: New dict with same keys A-D but randomized values
        - reverse_map: Maps shuffled letter → original letter so we can recover
                       the true answer after the model picks from shuffled order.

    Example:
        Original:  A=Aspirin  B=Ibuprofen  C=Tylenol  D=Morphine
        Shuffled:  A=Tylenol  B=Morphine   C=Aspirin  D=Ibuprofen
        reverse_map: {A→C, B→D, C→A, D→B}
        If model picks "C" from shuffled, reverse_map["C"] = "A" (original Aspirin)
    """
    original_items = list(options.items())  # [(A, val_a), (B, val_b), ...]
    values = [v for _, v in original_items]
    original_letters = [k for k, _ in original_items]

    # Shuffle the values
    shuffled_values = values.copy()
    random.shuffle(shuffled_values)

    # Build new options and reverse map
    shuffled_options = {}
    reverse_map = {}  # shuffled_letter → original_letter
    for i, new_val in enumerate(shuffled_values):
        new_letter = OPTION_LETTERS[i]
        # Find which original letter had this value
        orig_idx = values.index(new_val)
        orig_letter = original_letters[orig_idx]
        shuffled_options[new_letter] = new_val
        reverse_map[new_letter] = orig_letter
        # Mark used to handle duplicate values (rare but possible)
        values[orig_idx] = None
        original_letters[orig_idx] = None

    return shuffled_options, reverse_map


def _unshuffle_answer(predicted_letter: str, reverse_map: dict) -> str:
    """Map a prediction on shuffled options back to the original letter."""
    if predicted_letter in reverse_map:
        return reverse_map[predicted_letter]
    return predicted_letter  # fallback (X or unparsed)


# ── Answer Extraction ────────────────────────────────────────

def _extract_answer(raw: str) -> str:
    """Extract a single letter answer from LLM output.

    Handles markdown bold (**B**), various phrasings, and edge cases.
    Priority order (high → low confidence):
      1. Explicit "Answer: X" or "Final answer: X"
      2. "The answer/correct answer/best answer is X"
      3. Letter in parentheses (B)
      4. "Option X" / "choose X" / "select X"
      5. "X is the correct/best/most appropriate answer"
      6. Standalone bold letter **X** on last line
      7. Standalone letter on last line (≤5 chars after stripping bold/punct)
      8. Last bold letter **X** anywhere in text
      9. "X." or "X)" as a standalone token
    """
    text = raw.strip()

    # Strip markdown bold for matching — but keep original for fallbacks
    clean = re.sub(r'\*\*', '', text)

    # PRIMARY (fixed): explicit terminal "Answer: X" / "Final answer: X" tag.
    # Colon REQUIRED and letter must be standalone (negative lookahead), so hedge
    # words such as "the answer depends..." / "answer cannot..." are NOT mis-read
    # as an option letter. We take the LAST occurrence because the prompt asks the
    # model to END with its final answer. This resolves the mis-capture bug where
    # the old Strategy 1 grabbed the "d" in "depends".
    _primary = re.findall(
        r'(?:final\s+answer|answer)\s*:\s*\(?\*{0,2}([A-Da-d])\*{0,2}\)?(?![A-Za-z])',
        text, re.IGNORECASE)
    if _primary:
        return _primary[-1].upper()

    # Direct single letter (with or without bold)
    clean_stripped = clean.strip().rstrip('.')
    if len(clean_stripped) == 1 and clean_stripped.upper() in VALID_ANSWERS:
        return clean_stripped.upper()

    # Strategy 1 (hardened): "Answer: X" — now requires a colon and a standalone
    # letter, matching the PRIMARY rule above. Kept as a fallback for edge spacing.
    match = re.search(r'(?:answer|final answer)\s*:\s*\*?\*?\(?([A-Da-d])\)?\*?\*?(?![A-Za-z])', clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 2 (hardened): "The answer is X" — letter must be standalone so
    # phrasings like "the answer is difficult" cannot capture the leading letter.
    match = re.search(r'(?:the answer is|correct answer is|best answer is)\s*\*?\*?\(?([A-Da-d])\)?\*?\*?(?![A-Za-z])', clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 3: letter in parentheses — e.g. (B)
    match = re.search(r'\(([A-Da-d])\)', clean)
    if match:
        return match.group(1).upper()

    # Strategy 4: "Option X" / "choose X" / "select X"
    match = re.search(r'(?:option|choose|select|pick|go with|would be)\s+\(?([A-Da-d])\)?', clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 5: "X is the correct/best/most appropriate answer" (reverse order)
    match = re.search(r'\b([A-Da-d])\b[\s.]*(?:is the|is)\s+(?:correct|best|most appropriate|most likely|right)\s+(?:answer|option|choice)', clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Strategy 6: Last line standalone letter (strip bold/punctuation, allow up to 5 chars)
    last_line = clean.strip().split('\n')[-1].strip()
    last_clean = re.sub(r'[*._)\s]', '', last_line)
    if len(last_clean) <= 2:
        for c in last_clean.upper():
            if c in VALID_ANSWERS:
                return c

    # Strategy 7: Bold letter **X** anywhere — find the LAST one (closest to answer)
    bold_matches = re.findall(r'\*\*([A-Da-d])\*\*', text)
    if bold_matches:
        return bold_matches[-1].upper()

    # Strategy 8: Standalone "X." or "X)" token at end of text
    match = re.search(r'\b([A-Da-d])[.)]\s*$', clean, re.MULTILINE)
    if match:
        return match.group(1).upper()

    # Strategy 9: Last standalone letter A-D in the final 100 chars.
    #
    # BIAS FIX: the previous version filtered out *any* 'a'/'A', which made it
    # impossible for this fallback to EVER return "A" — reintroducing exactly the
    # position bias that option shuffling was added to remove. The real problem is
    # only the lowercase English article "a", not the answer letter "A".
    # So we match UPPERCASE letters only: the article "a" is naturally excluded,
    # while a genuine "A" answer is still recoverable.
    tail = clean[-100:] if len(clean) > 100 else clean
    tail_matches = re.findall(r'\b([A-D])\b', tail)  # uppercase only
    if tail_matches:
        return tail_matches[-1].upper()

    return "X"  # Could not parse — do NOT guess


# ── Condition A: LLM-only (zero-shot) ─────────────────────────

def baseline_llm_only(question: str, options: dict,
                      shuffled: dict, reverse_map: dict) -> str:
    """Zero-shot MCQ. Receives the SHARED shuffled layout for this question."""
    opts = "\n".join([f"{k}: {v}" for k, v in shuffled.items()])
    prompt = f"""{BASE_PERSONA}

Question: {question}

Options:
{opts}

{BASE_INSTRUCTION}"""

    raw = llm_call([{"role": "user", "content": prompt}],
                   max_tokens=EVAL_MAX_TOKENS, temperature=EVAL_TEMPERATURE)
    predicted = _extract_answer(raw)
    return {
        "pred": _unshuffle_answer(predicted, reverse_map),
        "raw": raw,
        "evidence_found": None,   # no retrieval in this condition
        "n_papers": None,
        "n_papers_total": None,
    }


# ── Condition B: LLM + raw PubMed (no filtering) ──────────────

def baseline_llm_pubmed(question: str, options: dict,
                        shuffled: dict, reverse_map: dict) -> str:
    """LLM + RAW PubMed retrieval (no MeSH, no relevance filtering).

    Identical persona / instruction / token budget / temperature as Condition A.
    The ONLY difference is the injected raw evidence block.
    """
    papers = _search_pubmed(question[:120], max_results=3)
    evidence = ""
    for p in papers:
        evidence += f"- {p.title}: {p.abstract[:300]}\n"

    opts = "\n".join([f"{k}: {v}" for k, v in shuffled.items()])
    prompt = f"""{BASE_PERSONA}

PubMed Evidence:
{evidence if evidence else 'No relevant papers found.'}

Question: {question}

Options:
{opts}

{BASE_INSTRUCTION}"""

    raw = llm_call([{"role": "user", "content": prompt}],
                   max_tokens=EVAL_MAX_TOKENS, temperature=EVAL_TEMPERATURE)
    predicted = _extract_answer(raw)
    return {
        "pred": _unshuffle_answer(predicted, reverse_map),
        "raw": raw,
        "evidence_found": bool(papers),   # did naive PubMed return anything
        "n_papers": len(papers),          # papers injected into the prompt
        "n_papers_total": len(papers),
    }


# ── Condition C: Full AgentDDx system ─────────────────────────

def full_system(question: str, options: dict,
                shuffled: dict, reverse_map: dict) -> str:
    """Full agentic pipeline: entity extraction + MeSH queries + relevance filter.

    Identical persona / instruction / token budget / temperature as Conditions A and B.
    The ONLY differences are the injected entity block and the quality-controlled
    evidence block — which is exactly the variable this ablation is testing.
    """
    case = ClinicalCase(id="eval", source="medqa", text=question)

    # Step 1 — Entity extraction
    entities = extract_entities(case)

    # Step 2 — Candidates = ALL FOUR MCQ options.
    # (Previously sliced to [:3], which starved the 4th option of evidence
    #  and biased the system against it whenever it was the correct answer.)
    candidates = list(options.values())[:4]

    # Step 3 — PubMed evidence WITH relevance filtering + MeSH queries
    evidence = fetch_evidence_for_candidates(
        candidates,                       # all 4, not [:3]
        entities.all_entities,
        max_papers=2,
        case_text=question,
        use_relevance_filter=True,
    )

    # Step 4 — Reasoning over the SHARED shuffled layout
    evidence_text = ""
    n_injected = 0          # papers actually placed into the prompt
    n_kept_total = 0        # all papers surviving the relevance filter
    relevance_scores = []   # per-paper relevance scores of surviving papers
    for dx, res in evidence.items():
        n_kept_total += len(res.articles)
        relevance_scores.extend(getattr(res, "relevance_scores", []) or [])
        for p in res.articles[:2]:
            evidence_text += f"- {dx}: [{p.pmid}] {p.title}\n  {p.abstract[:300]}\n"
            n_injected += 1

    opts = "\n".join([f"{k}: {v}" for k, v in shuffled.items()])
    entity_str = ", ".join(entities.all_entities[:10])

    prompt = f"""{BASE_PERSONA}

Clinical Entities: {entity_str}

PubMed Evidence:
{evidence_text if evidence_text else 'No specific evidence found.'}

Question: {question}

Options:
{opts}

{BASE_INSTRUCTION}"""

    raw = llm_call([{"role": "user", "content": prompt}],
                   max_tokens=EVAL_MAX_TOKENS, temperature=EVAL_TEMPERATURE)
    predicted = _extract_answer(raw)
    return {
        "pred": _unshuffle_answer(predicted, reverse_map),
        "raw": raw,
        "evidence_found": bool(evidence_text),   # did the pipeline inject any evidence
        "n_papers": n_injected,                   # papers placed in the prompt
        "n_papers_total": n_kept_total,           # papers surviving the filter
        "relevance_scores": relevance_scores,     # per-paper relevance of survivors
    }


# ── Statistical Tests ──────────────────────────────────────────

def mcnemar_test(correct_a: list, correct_b: list) -> dict:
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    if b + c == 0:
        return {"chi2": 0, "p_value": 1.0, "b": b, "c": c}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"chi2": round(chi2, 4), "p_value": round(p, 4), "b": b, "c": c}


def bootstrap_ci(correct: list, n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    n = len(correct)
    if n == 0:
        return {"mean": 0, "ci_lower": 0, "ci_upper": 0}
    # Own RNG stream: makes the CI reproducible no matter how many shuffles
    # preceded it (i.e. identical for a fresh run vs a resumed one).
    rng = random.Random(RANDOM_SEED)
    accs = []
    for _ in range(n_bootstrap):
        sample = [correct[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    alpha = (1 - ci) / 2
    lower = accs[int(alpha * n_bootstrap)]
    upper = accs[int((1 - alpha) * n_bootstrap)]
    return {
        "mean":     round(sum(correct) / n * 100, 1),
        "ci_lower": round(lower * 100, 1),
        "ci_upper": round(upper * 100, 1),
    }


# ── Category Detection ─────────────────────────────────────────

def categorize_question(question: str) -> str:
    q = question.lower()
    categories = {
        "diagnosis":    ["most likely diagnosis", "what is the diagnosis", "most likely cause",
                         "best explains", "most likely explanation"],
        "treatment":    ["treatment", "management", "therapy", "medication", "drug of choice",
                         "next best step", "most appropriate treatment"],
        "mechanism":    ["mechanism", "pathophysiology", "pathogenesis", "most likely involves",
                         "mediator", "receptor"],
        "anatomy":      ["structure", "nerve", "artery", "muscle", "ligament", "innervat"],
        "pharmacology": ["side effect", "adverse effect", "contraindic", "pharmacol"],
        "prevention":   ["screening", "prevention", "vaccine", "prophylax"],
    }
    for cat, keywords in categories.items():
        if any(kw in q for kw in keywords):
            return cat
    return "other"


def _checkpoint_filename(conditions: list) -> str:
    return results_path(f"eval_{'_'.join(conditions)}.json")


def _save_checkpoint(results: dict, log: list, conditions: list):
    fname = _checkpoint_filename(conditions)
    with open(fname, "w") as f:
        json.dump({"summary": results, "log": log, "conditions": conditions,
                   "parser_version": PARSER_VERSION}, f, indent=2)


def evaluate(n: int = 100, conditions: list = None, resume_file: str = None):
    _lazy_imports()   # pull in the LLM/agent stack only when actually evaluating
    conditions = conditions or ["llm_only", "llm_pubmed", "full_system"]

    print(f"\n{'='*60}")
    print(f"  AgentDDx MedQA Evaluation")
    print(f"  Conditions : {', '.join(conditions)}")
    print(f"  Questions  : {n}")
    print(f"  Debiasing  : Option shuffling enabled")
    print(f"  Output     : {_checkpoint_filename(conditions)}")
    print(f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    try:
        from datasets import load_dataset
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        print(f"Loaded MedQA test set: {len(ds)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Use ALL questions — no clinical filter so we test on the full 1273
    questions = list(ds)[:n]
    print(f"Using {len(questions)} questions (no clinical filter — full dataset)")

    log = []
    start_idx = 0
    resume_path = resume_file or _checkpoint_filename(conditions)
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        prev_ver = prev.get("parser_version")
        if prev_ver != PARSER_VERSION:
            # Old checkpoint was scored with a different (e.g. buggy) parser.
            # Resuming would mix parser versions, so start fresh instead.
            print(f"!! Checkpoint parser_version={prev_ver!r} != current "
                  f"{PARSER_VERSION!r}. NOT resuming — starting fresh to avoid "
                  f"mixing parser versions. (Old file: {resume_path})")
            log = []
            start_idx = 0
        else:
            log = prev.get("log", [])
            start_idx = len(log)
            if start_idx > 0:
                print(f"Resuming from question {start_idx + 1}")

    results = {c: {"correct": 0, "total": 0} for c in conditions}
    for entry in log:
        for c in conditions:
            if c in entry and entry[c] != "X":
                results[c]["total"] += 1
                if entry[c] == entry["correct"]:
                    results[c]["correct"] += 1

    # Track prediction distribution for bias monitoring
    pred_dist = {c: {"A": 0, "B": 0, "C": 0, "D": 0, "X": 0} for c in conditions}

    for i in range(start_idx, len(questions)):
        q        = questions[i]
        question = q["question"]
        options  = q["options"]
        correct  = q["answer_idx"]
        category = categorize_question(question)

        print(f"\n[{i+1}/{n}] Category: {category}")
        entry = {
            "idx":      i,
            "question": question[:120],
            "correct":  correct,
            "category": category,
        }

        # Shuffle ONCE per question, shared by all conditions.
        # RESUME-SAFE SEEDING: seed from the question index, not from global RNG
        # state. Without this, a resumed run would give questions different
        # shuffles than an uninterrupted run (the RNG stream depends on how many
        # shuffles already happened in the process), so results would silently
        # depend on where the job crashed. Seeding per-index makes question i's
        # shuffle identical whether it ran fresh or after 6 restarts.
        random.seed(RANDOM_SEED + i)
        shuffled, reverse_map = _shuffle_options(options)

        for cond in conditions:
            pred = "X"
            # Default metadata; overwritten by a successful call below.
            meta = {"raw": "", "evidence_found": None,
                    "n_papers": None, "n_papers_total": None}
            max_question_retries = 3
            for attempt in range(max_question_retries):
                try:
                    if cond == "llm_only":
                        out = baseline_llm_only(question, options,
                                                shuffled, reverse_map)
                    elif cond == "llm_pubmed":
                        out = baseline_llm_pubmed(question, options,
                                                  shuffled, reverse_map)
                    elif cond == "full_system":
                        out = full_system(question, options,
                                          shuffled, reverse_map)
                    else:
                        break

                    pred = out["pred"]
                    meta = out                 # capture raw text + evidence info
                    if pred != "X":
                        break  # Got a valid answer
                    # Got X (unparseable) — retry with backoff
                    if attempt < max_question_retries - 1:
                        print(f"  {cond} returned X — retrying ({attempt+2}/{max_question_retries})")
                        time.sleep(2 * (attempt + 1))
                except Exception as e:
                    print(f"  {cond} error (attempt {attempt+1}/{max_question_retries}): {e}")
                    if attempt < max_question_retries - 1:
                        time.sleep(5 * (attempt + 1))

            entry[cond] = pred
            # --- NEW: persist raw generation + retrieval metadata per condition ---
            entry[cond + "_raw"] = (meta.get("raw") or "")[:1500]
            if meta.get("evidence_found") is not None:
                entry[cond + "_evidence_found"] = meta.get("evidence_found")
                entry[cond + "_n_papers"] = meta.get("n_papers")
                entry[cond + "_n_papers_total"] = meta.get("n_papers_total")
                if meta.get("relevance_scores") is not None:
                    entry[cond + "_relevance_scores"] = meta.get("relevance_scores")
            # ---------------------------------------------------------------------
            if pred != "X":
                results[cond]["total"] += 1
                is_correct = pred == correct
                if is_correct:
                    results[cond]["correct"] += 1
            pred_dist[cond][pred] = pred_dist[cond].get(pred, 0) + 1

            symbol = "✓" if pred == correct else ("✗" if pred != "X" else "⚠")
            print(f"  {cond:<15} → {pred} ({symbol})")
            time.sleep(0.3)

        log.append(entry)

        if (i + 1) % 10 == 0:
            print(f"\n{'─'*50}")
            print(f"  Running accuracy after {i+1} questions:")
            for name, r in results.items():
                if r["total"] > 0:
                    acc = r["correct"] / r["total"] * 100
                    dist = pred_dist[name]
                    total_pred = sum(dist[l] for l in "ABCD")
                    a_pct = dist["A"] / total_pred * 100 if total_pred > 0 else 0
                    print(f"    {name:<15} {r['correct']}/{r['total']} = {acc:.1f}%  (A-rate: {a_pct:.0f}%)")
            print(f"{'─'*50}")

        if (i + 1) % 5 == 0:
            _save_checkpoint(results, log, conditions)

    print(f"\n\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for name, r in results.items():
        if r["total"] > 0:
            acc = r["correct"] / r["total"] * 100
            dist = pred_dist[name]
            total_pred = sum(dist[l] for l in "ABCD")
            print(f"  {name:<15} {r['correct']}/{r['total']} = {acc:.1f}%")
            if total_pred > 0:
                print(f"    Prediction dist: " + "  ".join(
                    f"{l}={dist[l]} ({dist[l]/total_pred*100:.0f}%)" for l in "ABCD"))

    report = _generate_report(results, log, conditions)
    _save_checkpoint(results, log, conditions)

    report_fname = results_path(f"eval_report_{'_'.join(conditions)}.json")
    with open(report_fname, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: {_checkpoint_filename(conditions)}")
    print(f"       {report_fname}")
    return report


def _generate_report(results: dict, log: list, conditions: list) -> dict:
    # Restrict to questions where EVERY condition has a recorded prediction, so
    # all conditions are scored over an identical denominator. (Previously each
    # condition was scored over only its own parseable rows, which silently gave
    # an advantage to whichever condition failed to parse least often.)
    paired_log = [e for e in log if all(c in e for c in conditions)]

    report = {
        "meta": {
            "date":             datetime.now().isoformat(),
            "total_questions":  len(log),
            "paired_questions": len(paired_log),
            "conditions":       conditions,
            "debiasing":        "option_shuffling (shared layout across conditions)",
            "seed":             RANDOM_SEED,
            "max_tokens":       EVAL_MAX_TOKENS,
            "temperature":      EVAL_TEMPERATURE,
            "scoring":          "unparseable ('X') counted as INCORRECT for all conditions",
        },
        "accuracy":          {},
        "parse_failures":    {},
        "per_category":      {},
        "statistical_tests": {},
        "error_analysis":    {},
    }

    condition_correct = {}
    for cond in conditions:
        # X counts as WRONG (not dropped) — same denominator for every condition.
        correct_list = [entry[cond] == entry["correct"] for entry in paired_log]
        condition_correct[cond] = correct_list

        n_x = sum(1 for entry in paired_log if entry[cond] == "X")
        report["parse_failures"][cond] = {
            "n_unparsed": n_x,
            "rate_pct":   round(100 * n_x / len(paired_log), 1) if paired_log else 0,
        }

        ci = bootstrap_ci(correct_list)
        report["accuracy"][cond] = {
            "correct":     sum(correct_list),
            "total":       len(correct_list),
            "accuracy":    ci["mean"],
            "ci_95_lower": ci["ci_lower"],
            "ci_95_upper": ci["ci_upper"],
        }
        print(f"\n  {cond}: {ci['mean']:.1f}% "
              f"(95% CI: {ci['ci_lower']:.1f}-{ci['ci_upper']:.1f}%) "
              f"[{n_x} unparsed, scored wrong]")

    # Everything downstream operates on the paired log.
    log = paired_log

    cond_list = list(conditions)
    for i, c1 in enumerate(cond_list):
        for c2 in cond_list[i+1:]:
            # No 'X' filter: an unparseable answer is simply a wrong answer.
            # Filtering X here would score McNemar on a different subset than
            # the accuracy table above, which is how the two used to disagree.
            paired_a = [entry[c1] == entry["correct"] for entry in log]
            paired_b = [entry[c2] == entry["correct"] for entry in log]
            if paired_a:
                test = mcnemar_test(paired_a, paired_b)
                key  = f"{c1}_vs_{c2}"
                report["statistical_tests"][key] = test
                sig  = ("***" if test["p_value"] < 0.001
                        else "**" if test["p_value"] < 0.01
                        else "*"  if test["p_value"] < 0.05
                        else "ns")
                print(f"  McNemar {c1} vs {c2}: "
                      f"chi2={test['chi2']}, p={test['p_value']} ({sig})")

    categories = set(e.get("category", "other") for e in log)
    for cat in sorted(categories):
        report["per_category"][cat] = {}
        for cond in conditions:
            cat_correct = [
                entry[cond] == entry["correct"]
                for entry in log
                if entry.get("category") == cat
            ]
            if cat_correct:
                report["per_category"][cat][cond] = {
                    "accuracy": round(sum(cat_correct) / len(cat_correct) * 100, 1),
                    "n":        len(cat_correct),
                }

    if "llm_only" in conditions and "full_system" in conditions:
        helped = hurt = both_right = both_wrong = 0
        for entry in log:
            if "llm_only" in entry and "full_system" in entry:
                lr = entry["llm_only"]    == entry["correct"]
                fr = entry["full_system"] == entry["correct"]
                if fr and not lr:   helped     += 1
                elif lr and not fr: hurt       += 1
                elif lr and fr:     both_right += 1
                else:               both_wrong += 1

        a = report["accuracy"].get("llm_only",    {}).get("accuracy", 0)
        b = report["accuracy"].get("llm_pubmed",  {}).get("accuracy", 0)
        c = report["accuracy"].get("full_system", {}).get("accuracy", 0)
        # RDR is only meaningful if naive retrieval actually degraded the baseline
        # by a non-trivial margin. Otherwise (a - b) ~ 0 and this explodes to
        # nonsense (we observed +440% and -160%). Report None rather than noise.
        rdr = round((c - b) / (a - b) * 100, 1) if (a - b) >= 2.0 else None

        report["error_analysis"] = {
            "retrieval_helped":          helped,
            "retrieval_hurt":            hurt,
            "both_correct":              both_right,
            "both_wrong":                both_wrong,
            "net_effect":                helped - hurt,
            "degradation_recovery_rate": rdr,
        }
        print(f"\n  Error analysis:")
        print(f"    Retrieval helped : {helped}")
        print(f"    Retrieval hurt   : {hurt}")
        print(f"    Net effect       : {'+' if helped >= hurt else ''}{helped - hurt}")
        if rdr is not None:
            print(f"    RDR              : {rdr}%")

    return report


# ── Merge separately-run conditions into ONE report ────────────

def merge_reports(files: list):
    """Merge per-condition eval files and produce a combined report.

    WHY THIS EXISTS: the statistical tests (McNemar) and error analysis compare
    conditions against EACH OTHER, so they can only run when all conditions live
    in one log. If you run `--condition llm_only`, then `--condition llm_pubmed`,
    etc., each output file contains a single condition and the stats sections come
    out empty ("statistical_tests": {}). This merges them on question index first.

    Usage:
      python evaluate.py --merge eval_llm_only.json eval_llm_pubmed.json eval_full_system.json
    """
    by_idx = {}
    all_conditions = []

    for path in files:
        if not os.path.exists(path):
            print(f"  !! missing: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        conds = data.get("conditions", [])
        for c in conds:
            if c not in all_conditions:
                all_conditions.append(c)
        for entry in data.get("log", []):
            idx = entry["idx"]
            if idx not in by_idx:
                by_idx[idx] = {
                    "idx":      idx,
                    "question": entry.get("question", ""),
                    "correct":  entry["correct"],
                    "category": entry.get("category", "other"),
                }
            for c in conds:
                if c in entry:
                    by_idx[idx][c] = entry[c]
                # Carry the enriched per-condition fields too (raw text,
                # evidence-found flag, paper counts) so the merged log is
                # self-contained for the retrieval-quality analysis.
                for suffix in ("_raw", "_evidence_found",
                               "_n_papers", "_n_papers_total", "_relevance_scores"):
                    k = c + suffix
                    if k in entry:
                        by_idx[idx][k] = entry[k]
        print(f"  loaded {path}: {len(data.get('log', []))} rows, conditions={conds}")

    # Keep only questions answered under EVERY condition (true paired set).
    merged = [e for e in by_idx.values() if all(c in e for c in all_conditions)]
    merged.sort(key=lambda e: e["idx"])

    print(f"\n  Conditions merged : {all_conditions}")
    print(f"  Paired questions  : {len(merged)} "
          f"(of {len(by_idx)} seen across all files)")

    if not merged:
        print("  !! No overlapping questions — nothing to compare.")
        return None

    report = _generate_report({}, merged, all_conditions)

    with open(results_path("eval_report_merged.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(results_path("eval_merged_log.json"), "w") as f:
        json.dump({"log": merged, "conditions": all_conditions}, f, indent=2)

    print("\n  Saved: eval_report_merged.json")
    print("         eval_merged_log.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentDDx MedQA Evaluation")
    parser.add_argument("--n",         type=int, default=100,
                        help="Number of questions (default 100, max 1273)")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["all", "llm_only", "llm_pubmed", "full_system"],
                        help="Which condition to run")
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to checkpoint file to resume from")
    parser.add_argument("--merge",     type=str, nargs="+", default=None,
                        help="Merge per-condition eval files into one report "
                             "with McNemar tests (no API calls made)")
    args = parser.parse_args()

    if args.merge:
        print(f"\n{'='*60}\n  Merging condition files\n{'='*60}\n")
        merge_reports(args.merge)
        sys.exit(0)

    conditions = (["llm_only", "llm_pubmed", "full_system"]
                  if args.condition == "all"
                  else [args.condition])

    evaluate(n=args.n, conditions=conditions, resume_file=args.resume)
