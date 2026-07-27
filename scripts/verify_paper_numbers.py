#!/usr/bin/env python3
"""Recompute every number reported in the paper from the released logs.

This script makes no API calls. It reads the per-question response logs shipped
in results/logs/ and recomputes accuracies, bootstrap confidence intervals,
McNemar tests, per-category breakdowns and retrieval statistics, then checks
each against the value printed in the paper.

Usage:
    python scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "results", "logs")

CONDITIONS = ["llm_only", "llm_pubmed", "full_system"]
ABLATION = [
    "llm_only",
    "entities_only",
    "evidence_only",
    "full_nofilter",
    "full_system",
]

BOOTSTRAP_RESAMPLES = 1000
SEED = 42


def load_log(name: str) -> list[dict]:
    """Load a per-question log, transparently handling gzip."""
    path = os.path.join(LOGS, name)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)["log"]


def accuracy(log: list[dict], cond: str) -> float:
    return 100.0 * sum(r[cond] == r["correct"] for r in log) / len(log)


def bootstrap_ci(log: list[dict], cond: str) -> tuple[float, float]:
    """Percentile bootstrap CI over questions, matching the paper's protocol."""
    rng = random.Random(SEED)
    hits = [1 if r[cond] == r["correct"] else 0 for r in log]
    n = len(hits)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        means.append(sum(hits[rng.randrange(n)] for _ in range(n)) / n * 100.0)
    means.sort()
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES)]


def mcnemar(log: list[dict], a: str, b: str) -> tuple[int, int, float]:
    """McNemar's test with continuity correction.

    Returns (lost, gained, p) where `lost` counts questions a answered correctly
    and b did not, and `gained` counts the reverse.
    """
    lost = sum(1 for r in log if r[a] == r["correct"] and r[b] != r["correct"])
    gained = sum(1 for r in log if r[a] != r["correct"] and r[b] == r["correct"])
    n = lost + gained
    if n == 0:
        return lost, gained, 1.0
    chi2 = (abs(lost - gained) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return lost, gained, p


def check(label: str, got, expected, tol=0.05, fmt="{:.1f}") -> bool:
    """Compare a computed value against the paper and print a verdict line."""
    if isinstance(expected, float):
        ok = abs(got - expected) <= tol
        shown, want = fmt.format(got), fmt.format(expected)
    else:
        ok = got == expected
        shown, want = str(got), str(expected)
    mark = "PASS" if ok else "FAIL"
    suffix = "" if ok else f"   <-- paper reports {want}"
    print(f"  [{mark}] {label:<52} {shown}{suffix}")
    return ok


def main() -> int:
    if not os.path.isdir(LOGS):
        print(f"error: {LOGS} not found", file=sys.stderr)
        return 2

    ok = True
    log = load_log("eval_merged_log.json.gz")

    print(f"\nMain experiment (n={len(log)})")
    print("-" * 72)
    ok &= check("Table 1  LLM-only accuracy (%)", accuracy(log, "llm_only"), 83.7)
    ok &= check("Table 1  LLM + naive PubMed accuracy (%)", accuracy(log, "llm_pubmed"), 83.8)
    ok &= check("Table 1  Full AgentDDx accuracy (%)", accuracy(log, "full_system"), 77.0)

    for cond, lo_exp, hi_exp in [
        ("llm_only", 81.6, 85.7),
        ("llm_pubmed", 81.7, 85.8),
        ("full_system", 74.5, 79.5),
    ]:
        lo, hi = bootstrap_ci(log, cond)
        ok &= check(f"Table 1  {cond} 95% CI lower", lo, lo_exp, tol=0.6)
        ok &= check(f"Table 1  {cond} 95% CI upper", hi, hi_exp, tol=0.6)

    print("\nPaired significance tests")
    print("-" * 72)
    lost, gained, p = mcnemar(log, "llm_only", "llm_pubmed")
    ok &= check("naive vs baseline: discordant lost", lost, 48)
    ok &= check("naive vs baseline: discordant gained", gained, 50)
    ok &= check("naive vs baseline: McNemar p", p, 0.92, tol=0.01, fmt="{:.2f}")

    lost, gained, p = mcnemar(log, "llm_only", "full_system")
    ok &= check("full vs baseline: discordant lost", lost, 140)
    ok &= check("full vs baseline: discordant gained", gained, 55)
    ok &= check("full vs baseline: McNemar p", p, 1.8e-9, tol=2e-10, fmt="{:.1e}")

    print("\nParsing and option-position controls")
    print("-" * 72)
    for cond in CONDITIONS:
        n_bad = sum(1 for r in log if r[cond] not in {"A", "B", "C", "D"})
        ok &= check(f"unparseable responses: {cond}", n_bad, 0)
    key = Counter(r["correct"] for r in log)
    pred = Counter(r["full_system"] for r in log)
    spread = max(abs(pred[k] - key[k]) for k in "ABCD")
    ok &= check("max |predicted - key| count, full system", spread <= 45, True)

    print("\nRetrieval volume")
    print("-" * 72)
    full_papers = [r["full_system_n_papers_total"] for r in log]
    ok &= check("mean papers injected, full pipeline", sum(full_papers) / len(log), 6.2, tol=0.05)
    ok &= check(
        "questions with evidence, full pipeline (%)",
        100.0 * sum(1 for r in log if r["full_system_evidence_found"]) / len(log),
        100.0,
    )
    naive_papers = [r["llm_pubmed_n_papers_total"] for r in log]
    ok &= check("mean papers injected, naive", sum(naive_papers) / len(log), 0.03, tol=0.01, fmt="{:.2f}")
    ok &= check(
        "questions with evidence, naive (%)",
        100.0 * sum(1 for r in log if r["llm_pubmed_evidence_found"]) / len(log),
        1.3,
        tol=0.1,
    )

    print("\nTable 2  accuracy by question category (%)")
    print("-" * 72)
    by_cat: dict[str, list] = defaultdict(list)
    for r in log:
        by_cat[r["category"]].append(r)
    expected_cat = {
        "diagnosis": (262, 88.2, 88.2, 82.4),
        "treatment": (476, 84.0, 84.5, 79.0),
        "mechanism": (40, 90.0, 90.0, 82.5),
        "anatomy": (52, 78.8, 73.1, 67.3),
        "prevention": (25, 84.0, 84.0, 68.0),
        "other": (412, 80.3, 81.1, 72.6),
    }
    for cat, (n_exp, *acc_exp) in expected_cat.items():
        rows = by_cat.get(cat, [])
        ok &= check(f"{cat}: N", len(rows), n_exp)
        for cond, exp in zip(CONDITIONS, acc_exp):
            ok &= check(f"{cat}: {cond}", accuracy(rows, cond), exp, tol=0.1)

    print("\nTable 3  component ablation (n=200)")
    print("-" * 72)
    abl = load_log("ablation_all_conditions.json.gz")
    ok &= check("ablation subset size", len(abl), 200)
    for cond, exp in zip(ABLATION, [82.5, 84.5, 76.5, 74.0, 73.5]):
        ok &= check(f"{cond} accuracy", accuracy(abl, cond), exp, tol=0.1)
    for cond, exp_p in [("entities_only", 0.45), ("evidence_only", 0.031), ("full_nofilter", 0.002)]:
        _, _, p = mcnemar(abl, "llm_only", cond)
        ok &= check(f"{cond} vs baseline: McNemar p", p, exp_p, tol=0.02, fmt="{:.3f}")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see FAIL lines above")
    print("=" * 72 + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
