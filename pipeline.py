"""Pipeline orchestrator — wires all agents together with status callbacks."""
import sys
import os

# Add project root to path so agents can import models/config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import ClinicalCase, ExtractedEntities, PipelineResult
from agents.entity_extractor import extract_entities
from agents.pubmed_retrieval import fetch_evidence_for_candidates
from agents.reasoning_agent import run_reasoning


def run_pipeline(
    case: ClinicalCase,
    on_step=None,
    additional_context: str = "",
) -> PipelineResult:
    """
    Run the full agentic pipeline.

    on_step(step_name, status) is called at each stage:
        step_name: 'entities', 'candidates', 'pubmed', 'reasoning'
        status: 'running', 'done', 'error'
    """
    callback = on_step or (lambda *_: None)
    result = PipelineResult(case=case)

    # Step 1: Entity extraction
    try:
        callback("entities", "running")
        entities = extract_entities(case)
        result.entities = entities
        callback("entities", "done")
    except Exception as e:
        callback("entities", "error")
        result.error = f"Entity extraction failed: {e}"
        return result

    # Step 2: Generate candidate diagnoses from entities
    # (In full system, this calls knowledge graph. For now, we let the
    #  reasoning agent determine candidates from entities + note.)
    try:
        callback("candidates", "running")
        # Build candidate list from entity context
        candidate_query = " ".join(entities.symptoms[:3] + entities.findings[:2])
        # We'll use Groq to quickly suggest candidates
        import json, re
        from utils import llm_call
        raw = llm_call(
            messages=[
                {"role": "system", "content": "You are a clinical AI. Given a clinical presentation, suggest 3-5 candidate diagnoses. Return ONLY a JSON list of strings. Example: [\"Disease A\", \"Disease B\"]"},
                {"role": "user", "content": f"Clinical note: {case.text}\n\nEntities: {', '.join(entities.all_entities)}"},
            ],
            max_tokens=200,
        ).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        candidates = json.loads(raw)
        if not isinstance(candidates, list):
            candidates = [str(candidates)]
        callback("candidates", "done")
    except Exception as e:
        # Fallback: use entity terms as query
        candidates = entities.symptoms[:2] + entities.findings[:2]
        callback("candidates", "done")

    # Step 3: PubMed evidence retrieval (with relevance filtering)
    try:
        callback("pubmed", "running")
        evidence = fetch_evidence_for_candidates(
            candidates,
            entities.all_entities,
            case_text=case.text,
            use_relevance_filter=True,
        )
        result.evidence = evidence
        callback("pubmed", "done")
    except Exception as e:
        callback("pubmed", "error")
        result.error = f"PubMed retrieval failed: {e}"
        # Continue without evidence
        evidence = {}
        result.evidence = evidence
        callback("pubmed", "done")

    # Step 4: Clinical reasoning
    try:
        callback("reasoning", "running")
        agent_result = run_reasoning(case, entities, evidence, additional_context)
        result.result = agent_result
        callback("reasoning", "done")
    except Exception as e:
        callback("reasoning", "error")
        result.error = f"Reasoning failed: {e}"
        return result

    return result
