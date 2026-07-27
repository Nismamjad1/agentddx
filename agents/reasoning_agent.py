"""Reasoning agent — produces ranked differential diagnosis with evidence-backed reasoning.

Improvements over baseline:
  1. Self-consistency: runs reasoning N times and aggregates via majority vote
  2. Explicit instruction to IGNORE evidence that contradicts clinical presentation
  3. Numerical confidence scores (0-100) alongside categorical labels
  4. Evidence-aware prompting that distinguishes supportive vs. irrelevant literature
  5. Robust JSON parsing with multiple fallback strategies
"""
import json
import re
import time
from collections import Counter
from models import ClinicalCase, ExtractedEntities, PubMedResult, Differential, AgentResult
import config
from utils import llm_call

REASONING_PROMPT = """You are an expert clinical reasoning agent performing differential diagnosis.

CLINICAL NOTE:
{note}

EXTRACTED ENTITIES:
- Symptoms: {symptoms}
- Findings: {findings}
- Labs: {labs}
- Demographics: {demographics}
- Medications: {medications}

PUBMED EVIDENCE:
{evidence_block}

INSTRUCTIONS:
1. First, assess the clinical presentation INDEPENDENTLY of the PubMed evidence.
2. Then, check if the PubMed evidence supports, contradicts, or is irrelevant to your clinical assessment.
3. ONLY incorporate evidence that is directly relevant to THIS patient's presentation. Ignore generic or tangential papers.
4. Rank 3-5 diagnoses from most to least likely.
5. For each diagnosis:
   - Explain step-by-step clinical reasoning
   - List specific features that support it
   - List specific features that argue against it
   - Assign a confidence score from 0 to 100
   - Cite PMIDs only if the paper directly supports your reasoning
6. If the case is ambiguous, set needs_more_info to true and ask ONE specific follow-up question.

Return ONLY valid JSON:
{{
  "differentials": [
    {{
      "rank": 1,
      "diagnosis": "Disease name",
      "confidence": "High",
      "confidence_score": 85,
      "reasoning": "Step-by-step clinical reasoning...",
      "supporting_features": ["feature1", "feature2"],
      "features_against": ["feature1"],
      "evidence_summary": "Brief summary of relevant PubMed evidence",
      "key_references": ["PMID:12345678"]
    }}
  ],
  "reasoning_trace": "Overall clinical thinking process...",
  "needs_more_info": false,
  "follow_up_question": ""
}}"""


def _build_evidence_block(evidence: dict[str, PubMedResult]) -> str:
    """Format PubMed evidence for the prompt."""
    if not evidence:
        return "No PubMed evidence retrieved."

    parts = []
    for diagnosis, result in evidence.items():
        if result.articles:
            articles_text = "\n".join(
                f"  - [PMID:{a.pmid}] {a.title}\n    {a.abstract[:500]}"
                for a in result.articles[:3]
            )
            parts.append(f"For '{diagnosis}':\n{articles_text}")
        else:
            parts.append(f"For '{diagnosis}': No articles found.")
    return "\n\n".join(parts)


def _parse_json_response(raw: str) -> dict:
    """Robust JSON extraction with multiple fallback strategies."""
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the outermost JSON object
    brace_depth = 0
    start = None
    for i, c in enumerate(raw):
        if c == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif c == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    return json.loads(raw[start:i+1])
                except json.JSONDecodeError:
                    start = None

    # Strategy 3: regex for JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def _single_reasoning_pass(
    prompt: str,
    temperature: float = 0.3,
) -> dict:
    """Run one reasoning pass and return parsed JSON."""
    raw = llm_call(
        messages=[
            {"role": "system", "content": (
                "You are a senior clinical reasoning AI. Output only valid JSON. "
                "Be thorough and evidence-based. Prioritize clinical presentation "
                "over literature when they conflict."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=3000,
    ).strip()
    return _parse_json_response(raw)


def _aggregate_results(results: list[dict]) -> dict:
    """Aggregate multiple reasoning passes using majority vote on top diagnosis."""
    if not results:
        return {}
    if len(results) == 1:
        return results[0]

    # Count top-1 diagnosis across passes
    top1_counts = Counter()
    for r in results:
        diffs = r.get("differentials", [])
        if diffs:
            # Find rank-1 diagnosis
            sorted_diffs = sorted(diffs, key=lambda d: d.get("rank", 99))
            top1 = sorted_diffs[0].get("diagnosis", "").lower().strip()
            top1_counts[top1] += 1

    # Find the result whose top-1 matches the majority
    majority_dx = top1_counts.most_common(1)[0][0] if top1_counts else ""

    for r in results:
        diffs = r.get("differentials", [])
        if diffs:
            sorted_diffs = sorted(diffs, key=lambda d: d.get("rank", 99))
            if sorted_diffs[0].get("diagnosis", "").lower().strip() == majority_dx:
                # Add consistency info to reasoning trace
                total = len(results)
                agree = top1_counts[majority_dx]
                r["reasoning_trace"] = (
                    f"[Self-consistency: {agree}/{total} passes agreed on "
                    f"top diagnosis] " + r.get("reasoning_trace", "")
                )
                return r

    # Fallback: return the first result
    return results[0]


def run_reasoning(
    case: ClinicalCase,
    entities: ExtractedEntities,
    evidence: dict[str, PubMedResult],
    additional_context: str = "",
    num_passes: int = 0,
) -> AgentResult:
    """
    Run the reasoning agent to produce a differential diagnosis.

    Args:
        num_passes: Number of self-consistency passes (0 = use config default).
                    1 = single pass (fastest), 3 = recommended for evaluation.
    """
    n_passes = num_passes or getattr(config, 'SELF_CONSISTENCY_PASSES', 1)

    note_text = case.text
    if additional_context:
        note_text += f"\n\nAdditional information from clinician: {additional_context}"

    # Build entity strings, handling both list and missing attributes
    symptoms = ", ".join(getattr(entities, 'symptoms', []) or []) or "None extracted"
    findings = ", ".join(getattr(entities, 'findings', []) or []) or "None extracted"
    labs = ", ".join(getattr(entities, 'labs', []) or []) or "None extracted"
    medications = ", ".join(getattr(entities, 'medications', []) or []) or "None"
    demographics = json.dumps(entities.demographics) if entities.demographics else "Not specified"

    prompt = REASONING_PROMPT.format(
        note=note_text,
        symptoms=symptoms,
        findings=findings,
        labs=labs,
        demographics=demographics,
        medications=medications,
        evidence_block=_build_evidence_block(evidence),
    )

    # Run reasoning passes
    all_results = []
    for i in range(n_passes):
        temp = 0.1 if n_passes == 1 else 0.3 + (i * 0.1)  # Vary temperature
        try:
            data = _single_reasoning_pass(prompt, temperature=min(temp, 0.7))
            if data.get("differentials"):
                all_results.append(data)
        except Exception as e:
            print(f"  Reasoning pass {i+1} failed: {e}")

        if i < n_passes - 1:
            time.sleep(0.3)

    if not all_results:
        return AgentResult(reasoning_trace="All reasoning passes failed.")

    # Aggregate via majority vote
    data = _aggregate_results(all_results)

    differentials = []
    for d in data.get("differentials", []):
        differentials.append(Differential(
            rank=d.get("rank", 0),
            diagnosis=d.get("diagnosis", "Unknown"),
            confidence=d.get("confidence", "Low"),
            confidence_score=d.get("confidence_score", 50),
            reasoning=d.get("reasoning", ""),
            supporting_features=d.get("supporting_features", []),
            features_against=d.get("features_against", []),
            evidence_summary=d.get("evidence_summary", ""),
            key_references=d.get("key_references", []),
        ))

    differentials.sort(key=lambda x: x.rank)

    return AgentResult(
        differentials=differentials,
        reasoning_trace=data.get("reasoning_trace", ""),
        needs_more_info=data.get("needs_more_info", False),
        follow_up_question=data.get("follow_up_question", ""),
    )


# Legacy compatibility alias
def run_full_pipeline(case, entities, evidence_dict, kg_candidates=None):
    """Legacy wrapper for backward compatibility."""
    return run_reasoning(case, entities, evidence_dict)
