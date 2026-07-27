"""
Retrieval-quality analysis for AgentDDx.
=======================================
Reads the enriched merged log produced by evaluate.py (--merge) and prints:

  PART 1  Characterizing the retrieval
          - evidence-hit rate per condition
          - mean papers injected / surviving the filter (full_system)

  PART 2  Connecting retrieval to the outcome
          - accuracy conditioned on whether evidence was injected
          - "lost question" analysis: of questions the baseline got right but
            the full system got wrong, how many actually received evidence?

Relevance-score analyses (distribution; accuracy-by-relevance-bin) are marked
TODO: they require per-paper relevance scores, which need a small change to
agents/pubmed_retrieval.py so fetch_evidence_for_candidates returns the scores.

Usage:
  python analyze_retrieval.py eval_merged_log.json
"""
import sys, json


def _acc(entries, cond):
    n = len(entries)
    if n == 0:
        return None
    c = sum(1 for e in entries if e.get(cond) == e.get("correct"))
    return 100.0 * c / n, c, n


def main(path):
    data = json.load(open(path))
    log = data["log"]
    conds = data["conditions"]
    N = len(log)
    print(f"Loaded {N} paired questions | conditions: {conds}\n")

    # ---------- PART 1: characterize the retrieval ----------
    print("=" * 60)
    print("PART 1  Retrieval characterization")
    print("=" * 60)
    for c in conds:
        ef_key = c + "_evidence_found"
        if not any(ef_key in e for e in log):
            print(f"{c:14s}  (no retrieval — no evidence fields)")
            continue
        got = [e for e in log if e.get(ef_key) is True]
        hit = 100.0 * len(got) / N
        npap = [e.get(c + "_n_papers") for e in log if e.get(c + "_n_papers") is not None]
        ntot = [e.get(c + "_n_papers_total") for e in log if e.get(c + "_n_papers_total") is not None]
        mean_inj = sum(npap) / len(npap) if npap else 0
        mean_tot = sum(ntot) / len(ntot) if ntot else 0
        print(f"{c:14s}  evidence-hit: {hit:5.1f}%  "
              f"mean papers injected: {mean_inj:.2f}  surviving filter: {mean_tot:.2f}")
    # relevance-score distribution (full_system)
    rel_all = []
    for e in log:
        rel_all.extend(e.get("full_system_relevance_scores", []) or [])
    if rel_all:
        rel_all.sort()
        n = len(rel_all)
        mean = sum(rel_all) / n
        median = rel_all[n // 2]
        print(f"\nrelevance scores (full_system survivors): n={n}  "
              f"mean={mean:.3f}  median={median:.3f}  "
              f"min={rel_all[0]:.2f}  max={rel_all[-1]:.2f}")
    else:
        print("\n  (no relevance scores found — is agents/pubmed_retrieval.py the "
              "updated version that returns scores?)")

    # ---------- PART 2: connect retrieval to outcome ----------
    print("\n" + "=" * 60)
    print("PART 2  Evidence effect on accuracy")
    print("=" * 60)
    for c in conds:
        ef_key = c + "_evidence_found"
        if not any(ef_key in e for e in log):
            continue
        with_ev = [e for e in log if e.get(ef_key) is True]
        no_ev = [e for e in log if e.get(ef_key) is False]
        a_with = _acc(with_ev, c)
        a_no = _acc(no_ev, c)
        print(f"\n{c}:")
        if a_with:
            print(f"  accuracy WHEN evidence injected : {a_with[0]:5.1f}%  ({a_with[1]}/{a_with[2]})")
        if a_no:
            print(f"  accuracy WHEN no evidence       : {a_no[0]:5.1f}%  ({a_no[1]}/{a_no[2]})")
        print("  (associational, not causal — the ablation isolates cause.)")

    # ---------- lost-question analysis ----------
    if "llm_only" in conds and "full_system" in conds:
        lost = [e for e in log
                if e.get("llm_only") == e.get("correct")
                and e.get("full_system") != e.get("correct")]
        ef_key = "full_system_evidence_found"
        lost_with_ev = [e for e in lost if e.get(ef_key) is True]
        print("\n" + "-" * 60)
        print(f"Lost-question analysis (baseline correct, full_system wrong):")
        print(f"  total lost: {len(lost)}")
        if lost:
            print(f"  of those, received evidence: {len(lost_with_ev)} "
                  f"({100.0*len(lost_with_ev)/len(lost):.1f}%)")
            print("  -> if most lost questions got confident evidence, that")
            print("     supports 'injected evidence competes with knowledge'.")
    # accuracy by mean-relevance bin (full_system)
    if "full_system" in conds:
        binned = {"low (<0.4)": [], "med (0.4-0.7)": [], "high (>=0.7)": []}
        for e in log:
            scores = e.get("full_system_relevance_scores", []) or []
            if not scores:
                continue
            m = sum(scores) / len(scores)
            b = "low (<0.4)" if m < 0.4 else ("med (0.4-0.7)" if m < 0.7 else "high (>=0.7)")
            binned[b].append(e)
        print("\n" + "-" * 60)
        print("Accuracy by mean evidence-relevance (full_system):")
        any_bin = False
        for b, items in binned.items():
            a = _acc(items, "full_system")
            if a:
                any_bin = True
                print(f"  {b:16s} acc={a[0]:5.1f}%  (n={a[2]})")
        if not any_bin:
            print("  (no relevance scores — using updated pubmed_retrieval.py?)")
        print("  NOTE: 'relevance' = your pipeline's own score, not ground truth;")
        print("        report N per bin, small bins have wide error bars.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python analyze_retrieval.py eval_merged_log.json")
        sys.exit(1)
    main(sys.argv[1])
