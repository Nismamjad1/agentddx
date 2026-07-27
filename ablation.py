"""
AgentDDx — Component Ablation Harness
=====================================
Isolates WHICH part of the full agentic pipeline causes the accuracy drop.
Kept SEPARATE from evaluate.py so the headline run is never touched.

Reuses evaluate.py's TESTED helpers (fixed parser, shuffling, seeding, report),
so the ablation is scored exactly like the main experiment.

Conditions (all share the same shuffled layout + fixed parser):
  llm_only        baseline (no entities, no evidence)          [reference]
  entities_only   entity block ONLY, no evidence
  evidence_only   filtered evidence ONLY, no entity block
  full_nofilter   entities + evidence, relevance filter OFF
  full_system     entities + evidence, relevance filter ON     [reference]

Read the pairwise McNemar results in the report like this:
  llm_only     vs entities_only  -> effect of the entity block alone
  llm_only     vs evidence_only  -> effect of the evidence alone
  full_nofilter vs full_system   -> effect of the relevance filter
  llm_only     vs full_system    -> the headline degradation, reproduced here

Run (defaults to FULL dataset; use --sample for a subset):
  python ablation.py                 # all conditions, full 1273
  python ablation.py --sample 400    # random fixed-seed subset of 400
  python ablation.py --n 1273 --sample 400

Then analyze / merge with the same tools as the main run:
  python analyze_retrieval.py eval_<...>.json    # (single-file, see note below)

NOTE ON SAMPLING: the subset is a RANDOM sample with a FIXED seed (reproducible).
It is NOT stratified by category, so per-category ablation breakdowns on a small
sample may be sparse. For per-category ablation, run the full dataset.
"""
import os, sys, json, time, argparse, random
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E   # reuse tested helpers: parser, shuffle, report, seeds

_results_path = E.results_path   # all artifacts land under results/


# ── New ablation conditions ───────────────────────────────────
# Each returns the same metadata dict shape as evaluate.py's conditions.

def cond_entities_only(question, options, shuffled, reverse_map):
    """Entity block ONLY — no evidence. Isolates the effect of entity injection."""
    case = E.ClinicalCase(id="eval", source="medqa", text=question)
    entities = E.extract_entities(case)
    opts = "\n".join(f"{k}: {v}" for k, v in shuffled.items())
    entity_str = ", ".join(entities.all_entities[:10])
    prompt = f"""{E.BASE_PERSONA}

Clinical Entities: {entity_str}

Question: {question}

Options:
{opts}

{E.BASE_INSTRUCTION}"""
    raw = E.llm_call([{"role": "user", "content": prompt}],
                     max_tokens=E.EVAL_MAX_TOKENS, temperature=E.EVAL_TEMPERATURE)
    predicted = E._extract_answer(raw)
    return {
        "pred": E._unshuffle_answer(predicted, reverse_map), "raw": raw,
        "evidence_found": None, "n_papers": None,
        "n_papers_total": None, "relevance_scores": None,
    }


def cond_evidence_only(question, options, shuffled, reverse_map):
    """Filtered evidence ONLY — no entity block in the prompt.
    (Entity extraction still runs, because it is needed to build MeSH queries;
    the entities are simply not shown to the model.)"""
    case = E.ClinicalCase(id="eval", source="medqa", text=question)
    entities = E.extract_entities(case)
    candidates = list(options.values())[:4]
    evidence = E.fetch_evidence_for_candidates(
        candidates, entities.all_entities, max_papers=2,
        case_text=question, use_relevance_filter=True)

    evidence_text = ""; n_injected = 0; n_kept_total = 0; relevance_scores = []
    for dx, res in evidence.items():
        n_kept_total += len(res.articles)
        relevance_scores.extend(getattr(res, "relevance_scores", []) or [])
        for p in res.articles[:2]:
            evidence_text += f"- {dx}: [{p.pmid}] {p.title}\n  {p.abstract[:300]}\n"
            n_injected += 1

    opts = "\n".join(f"{k}: {v}" for k, v in shuffled.items())
    prompt = f"""{E.BASE_PERSONA}

PubMed Evidence:
{evidence_text if evidence_text else 'No specific evidence found.'}

Question: {question}

Options:
{opts}

{E.BASE_INSTRUCTION}"""
    raw = E.llm_call([{"role": "user", "content": prompt}],
                     max_tokens=E.EVAL_MAX_TOKENS, temperature=E.EVAL_TEMPERATURE)
    predicted = E._extract_answer(raw)
    return {
        "pred": E._unshuffle_answer(predicted, reverse_map), "raw": raw,
        "evidence_found": bool(evidence_text), "n_papers": n_injected,
        "n_papers_total": n_kept_total, "relevance_scores": relevance_scores,
    }


def cond_full_nofilter(question, options, shuffled, reverse_map):
    """Entities + evidence, but relevance filter OFF. Compared against full_system
    (filter ON), this isolates the effect of the relevance filter."""
    case = E.ClinicalCase(id="eval", source="medqa", text=question)
    entities = E.extract_entities(case)
    candidates = list(options.values())[:4]
    evidence = E.fetch_evidence_for_candidates(
        candidates, entities.all_entities, max_papers=2,
        case_text=question, use_relevance_filter=False)   # <-- filter OFF

    evidence_text = ""; n_injected = 0; n_kept_total = 0
    for dx, res in evidence.items():
        n_kept_total += len(res.articles)
        for p in res.articles[:2]:
            evidence_text += f"- {dx}: [{p.pmid}] {p.title}\n  {p.abstract[:300]}\n"
            n_injected += 1

    opts = "\n".join(f"{k}: {v}" for k, v in shuffled.items())
    entity_str = ", ".join(entities.all_entities[:10])
    prompt = f"""{E.BASE_PERSONA}

Clinical Entities: {entity_str}

PubMed Evidence:
{evidence_text if evidence_text else 'No specific evidence found.'}

Question: {question}

Options:
{opts}

{E.BASE_INSTRUCTION}"""
    raw = E.llm_call([{"role": "user", "content": prompt}],
                     max_tokens=E.EVAL_MAX_TOKENS, temperature=E.EVAL_TEMPERATURE)
    predicted = E._extract_answer(raw)
    return {
        "pred": E._unshuffle_answer(predicted, reverse_map), "raw": raw,
        "evidence_found": bool(evidence_text), "n_papers": n_injected,
        "n_papers_total": n_kept_total,
        "relevance_scores": [],   # filter OFF => no relevance scores produced
    }


# dispatch table — llm_only and full_system reuse the tested evaluate.py fns
CONDITION_FNS = {
    "llm_only":      E.baseline_llm_only,
    "entities_only": cond_entities_only,
    "evidence_only": cond_evidence_only,
    "full_nofilter": cond_full_nofilter,
    "full_system":   E.full_system,
}
DEFAULT_CONDITIONS = ["llm_only", "entities_only", "evidence_only",
                      "full_nofilter", "full_system"]


def _ablation_filename(conditions):
    return _results_path(f"ablation_{'_'.join(conditions)}.json")


def run_ablation(n=1273, sample=0, conditions=None, resume_file=None):
    E._lazy_imports()   # pull in the real LLM/agent stack
    conditions = conditions or DEFAULT_CONDITIONS

    from datasets import load_dataset
    ds = list(load_dataset("GBaker/MedQA-USMLE-4-options", split="test"))[:n]

    # --- select the (original) indices to run ---
    if 0 < sample < len(ds):
        rng = random.Random(E.RANDOM_SEED)
        run_indices = sorted(rng.sample(range(len(ds)), sample))
        mode = f"random sample of {sample} (seed {E.RANDOM_SEED})"
    else:
        run_indices = list(range(len(ds)))
        mode = f"full dataset ({len(ds)})"

    print(f"\n{'='*60}\n  AgentDDx ABLATION\n  Conditions : {', '.join(conditions)}")
    print(f"  Questions  : {mode}")
    print(f"  Parser     : {E.PARSER_VERSION}\n{'='*60}\n")

    # --- resume (with parser-version guard, same policy as evaluate.py) ---
    log = []
    resume_path = resume_file or _ablation_filename(conditions)
    done_idx = set()
    if os.path.exists(resume_path):
        with open(resume_path) as f:
            prev = json.load(f)
        if prev.get("parser_version") != E.PARSER_VERSION:
            print(f"!! Checkpoint parser_version={prev.get('parser_version')!r} != "
                  f"{E.PARSER_VERSION!r}. NOT resuming — starting fresh.")
        else:
            log = prev.get("log", [])
            done_idx = {e["idx"] for e in log}
            if done_idx:
                print(f"Resuming — {len(done_idx)} questions already done.")

    results = {c: {"correct": 0, "total": 0} for c in conditions}
    for e in log:
        for c in conditions:
            if c in e and e[c] != "X":
                results[c]["total"] += 1
                if e[c] == e["correct"]:
                    results[c]["correct"] += 1

    def _save():
        with open(_ablation_filename(conditions), "w") as f:
            json.dump({"summary": results, "log": log, "conditions": conditions,
                       "parser_version": E.PARSER_VERSION}, f, indent=2)

    for count, i in enumerate(run_indices):
        if i in done_idx:
            continue
        q = ds[i]
        question, options, correct = q["question"], q["options"], q["answer_idx"]
        category = E.categorize_question(question)
        print(f"\n[{count+1}/{len(run_indices)}] idx={i} ({category})")

        entry = {"idx": i, "question": question[:120],
                 "correct": correct, "category": category}

        # SAME per-index shuffle seed as the main run -> comparable layouts.
        random.seed(E.RANDOM_SEED + i)
        shuffled, reverse_map = E._shuffle_options(options)

        for cond in conditions:
            pred = "X"
            meta = {"raw": "", "evidence_found": None, "n_papers": None,
                    "n_papers_total": None, "relevance_scores": None}
            for attempt in range(3):
                try:
                    out = CONDITION_FNS[cond](question, options, shuffled, reverse_map)
                    pred = out["pred"]; meta = out
                    if pred != "X":
                        break
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                except Exception as ex:
                    print(f"  {cond} error ({attempt+1}/3): {ex}")
                    if attempt < 2:
                        time.sleep(5 * (attempt + 1))

            entry[cond] = pred
            entry[cond + "_raw"] = (meta.get("raw") or "")[:1500]
            if meta.get("evidence_found") is not None:
                entry[cond + "_evidence_found"] = meta.get("evidence_found")
                entry[cond + "_n_papers"] = meta.get("n_papers")
                entry[cond + "_n_papers_total"] = meta.get("n_papers_total")
                if meta.get("relevance_scores") is not None:
                    entry[cond + "_relevance_scores"] = meta.get("relevance_scores")
            if pred != "X":
                results[cond]["total"] += 1
                if pred == correct:
                    results[cond]["correct"] += 1
            print(f"  {cond:<14} -> {pred} ({'OK' if pred==correct else ('x' if pred!='X' else 'X')})")
            time.sleep(0.2)

        log.append(entry)
        if (count + 1) % 5 == 0:
            _save()

    # reuse evaluate.py's TESTED report generator (accuracy, CIs, pairwise McNemar)
    report = E._generate_report(results, log, conditions)
    _save()
    with open(_results_path(f"ablation_report_{'_'.join(conditions)}.json"), "w") as f:
        json.dump(report, f, indent=2)
    # also write a merged-log name the analysis script understands
    with open(_results_path("ablation_merged_log.json"), "w") as f:
        json.dump({"log": log, "conditions": conditions}, f, indent=2)

    print("\nSaved: " + _ablation_filename(conditions))
    print("       ablation_merged_log.json  (feed to analyze_retrieval.py)")
    print("\nNOTE: the report's 'degradation_recovery_rate' is only meaningful "
          "with llm_pubmed present; ignore it in the ablation report.")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AgentDDx component ablation")
    ap.add_argument("--n", type=int, default=1273,
                    help="Cap on dataset size (default full 1273)")
    ap.add_argument("--sample", type=int, default=0,
                    help="Random fixed-seed subset size (0 = full dataset)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help=f"Subset of {DEFAULT_CONDITIONS}")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    conds = args.conditions or DEFAULT_CONDITIONS
    bad = [c for c in conds if c not in CONDITION_FNS]
    if bad:
        print(f"Unknown conditions: {bad}. Valid: {list(CONDITION_FNS)}")
        sys.exit(1)

    run_ablation(n=args.n, sample=args.sample, conditions=conds,
                 resume_file=args.resume)
