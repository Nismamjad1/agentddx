# AgentDDx

Code and evaluation logs for **"AgentDDx: When Agentic Evidence Injection Degrades Accuracy — A Paired Evaluation on Medical Question Answering."**

AgentDDx is an agentic differential-diagnosis system: it extracts structured clinical entities from a vignette, composes MeSH-aware PubMed queries for each candidate diagnosis, retrieves and relevance-filters the returned abstracts, and answers over the surviving evidence. This repository contains the system, the evaluation harness, and the per-question logs behind every number in the paper's quantitative evaluation. The eight illustrative cases of the paper's qualitative section are not part of this release.

The headline finding is negative. On the full MedQA-USMLE test set, retrieval does not improve closed-form accuracy for a strong backbone, and the full agentic pipeline significantly degrades it.

| Condition | Accuracy (%) | 95% CI | Δ vs. baseline | McNemar |
|---|---|---|---|---|
| LLM-only (baseline) | 83.7 | [81.6, 85.7] | — | — |
| LLM + naive PubMed | 83.8 | [81.7, 85.8] | +0.2 | p = 0.92 (48 lost / 50 gained) |
| Full AgentDDx | 77.0 | [74.5, 79.5] | −6.7 | p = 1.8 × 10⁻⁹ (140 lost / 55 gained) |

All three conditions see the same 1,273 questions under the same randomly shuffled answer-option layout, so the comparison is paired and option-position bias is controlled.

## Verify the paper's numbers without an API key

The per-question response logs are released with the code. Every accuracy, confidence interval, McNemar test, category breakdown and retrieval statistic in the paper can be recomputed from them offline:

```bash
pip install -r requirements.txt
python scripts/verify_paper_numbers.py
```

This makes no network calls and takes a few seconds. It recomputes each reported value and prints PASS/FAIL against the published figure.

### In a container

If you would rather not install anything, the default image needs no
dependencies and no API key:

```bash
docker build -t agentddx .
docker run --rm agentddx
```

The verification script imports only the Python standard library, so this image
carries just the released logs and builds in seconds.

## Reproducing the evaluation from scratch

Reproducing end to end requires an OpenRouter key and issues roughly 1,273 × 3 backbone calls plus PubMed traffic.

```bash
pip install -r requirements.txt
cp .env.example .env          # then add your OpenRouter key and contact email
./scripts/run_eval.sh         # all three conditions, resumable
```

`run_eval.sh` supervises the three conditions in parallel and restarts any that die, resuming from the last checkpoint — a full run takes many hours and will hit transient rate limits. Individual conditions can be run directly:

```bash
python evaluate.py --n 1273 --condition llm_only
python evaluate.py --n 1273 --condition llm_pubmed
python evaluate.py --n 1273 --condition full_system
python evaluate.py --merge results/eval_llm_only.json \
                           results/eval_llm_pubmed.json \
                           results/eval_full_system.json
```

The same run inside a container, with results written back to the host:

```bash
docker build --target full -t agentddx:full .
docker run --rm --env-file .env -v "$PWD/results:/app/results" \
    agentddx:full python evaluate.py --n 1273 --condition llm_only
```

`docker compose run --rm eval` wraps that, and `docker compose up demo` serves the
interactive branch on http://localhost:8501. Dependencies inside the image are
installed from `requirements.lock`, which pins exact versions; `requirements.txt`
carries the looser constraints for a local install, and `requirements-demo.txt`
adds Streamlit and FastAPI for the demo only.

The component ablation of Table 3 runs on a fixed 200-question subset (seed 42) drawn from the same test split:

```bash
python ablation.py --n 200
```

Retrieval statistics reported in Section 4 come from:

```bash
python analyze_retrieval.py results/eval_merged_log.json
```

## Experimental controls

These are the properties that make the comparison interpretable, and they are enforced in code rather than by convention.

**Shared shuffled layouts.** LLMs are not robust multiple-choice selectors, so a fixed A–D layout would confound any cross-system comparison. Each question's four option values are shuffled once, seeded per question index from a fixed base seed, and the same layout is used by every condition. The model's letter is mapped back through the stored permutation before scoring.

**Equalised generation settings.** Every condition uses the same persona, the same answer-format instruction, temperature 0.1 and a 900-token ceiling. These are defined once as module constants in `evaluate.py` and applied uniformly, so the only variable across conditions is the injected context.

**Conservative answer extraction.** The extractor requires an explicit terminal `Answer: <letter>` tag with a colon and a standalone letter, taking the last such tag when several appear. Responses from which no valid letter can be recovered are scored as incorrect rather than guessed, so parsing failures can never inflate accuracy. No response was unparseable in the main experiment.

**Parser versioning.** Each checkpoint records the answer-extraction version, and a resume across a version mismatch is refused rather than silently mixing scoring logic across runs.

**Common denominator.** Only questions answered under every condition are retained, so no condition benefits from parsing less often. In the main experiment this retained all 1,273 questions.

## Repository layout

```
.
├── Dockerfile               Two targets: verify (default) and full
├── docker-compose.yml       Wrappers for verify / eval / demo
├── evaluate.py              Main three-condition evaluation harness
├── ablation.py              Five-condition component ablation (Table 3)
├── analyze_retrieval.py     Retrieval-volume and relevance analysis
├── pipeline.py              Orchestrator wiring the agent stages together
├── config.py                Settings and demo cases
├── models.py                Shared dataclasses
├── utils.py                 Backbone API wrapper with retry/backoff
├── agents/
│   ├── entity_extractor.py  Stage 1 — vignette to typed JSON record
│   ├── pubmed_retrieval.py  Stages 2-4 — MeSH queries, Entrez, relevance filter
│   └── reasoning_agent.py   Stage 5 — differential and follow-up question
├── app.py                   Streamlit demo of the interactive branch
├── server.py                FastAPI backend
├── scripts/
│   ├── run_eval.sh          Resumable supervisor for the full run
│   ├── verify_paper_numbers.py   Offline check of every reported number
│   ├── check_all_evals.py   Quick accuracy summary across artifacts
│   └── stem_lengths.py      Question-stem length statistics
├── results/
│   ├── eval_report_*.json   Summary reports per condition and merged
│   └── logs/*.json.gz       Per-question response logs (gzipped)
└── docs/                    Architecture figure (SVG, PDF, LaTeX)
```

## Configuration

Runtime settings come from the environment; see `.env.example`. The defaults match the paper: `meta-llama/llama-3.3-70b-instruct` as the single backbone for every LLM stage, a relevance threshold of 0.4, and at most two retained papers per candidate diagnosis. `PUBMED_EMAIL` is required by NCBI's Entrez usage policy and is sent with every request.

## Data

The benchmark is MedQA-USMLE, four-option variant, redistributed as `GBaker/MedQA-USMLE-4-options` on the HuggingFace Hub under CC-BY-4.0, loaded through `datasets` at run time. It contains no patient-identifiable information and comprises retired licensing-examination items, so no ethics approval was required. The official test split is used without modification — no clinical or length filtering — so all reported accuracies are over the complete benchmark.

## Scope and limitations

The evaluation measures one narrow quantity: whether the correct option is selected on a multiple-choice benchmark. The interactive branch asks a follow-up question when the reasoning agent judges a case ambiguous (`needs_more_info` in `agents/reasoning_agent.py`); confidence is reported alongside but does not gate the question. That is the right instrument for the question the paper asks, and a poor one for the broader capabilities of an agentic diagnostic system. AgentDDx's open-ended differential, evidence attribution and confidence-triggered follow-up question cannot be scored by option accuracy; the interactive branch is exercised by `app.py` and assessed qualitatively in the paper rather than by this harness.

Three limitations bound the quantitative claims. The conditions differ in the quantity as well as the quality of injected text, so a context-length effect cannot be separated from a knowledge-conflict one. The naive condition's truncated queries retrieved on only 1.3% of questions, so it functions as a second no-retrieval baseline rather than a test of naive retrieval. And a single backbone on a single benchmark is evaluated; whether the effect holds for models with thinner parametric knowledge is untested.

## License

Released under the MIT License. See `LICENSE`.
