"""FastAPI server for AgentDDx — serves the UI and handles analysis requests."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.entity_extractor import extract_entities
from agents.pubmed_retrieval import fetch_evidence_for_candidates
from agents.reasoning_agent import run_reasoning
from models import ClinicalCase
import config
from utils import llm_call

app = FastAPI(title="AgentDDx", description="Agentic Differential Diagnosis System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyseRequest(BaseModel):
    text: str
    candidates: list[str] = []
    max_papers: int = 2


def _generate_candidates(note_text: str, entities) -> list[str]:
    """Use Llama to generate candidate diagnoses from clinical features."""
    symptoms = ", ".join(entities.symptoms[:5])
    findings = ", ".join(entities.findings[:5])
    prompt = (
        "You are a senior physician. Based on these clinical features, "
        "list the 5 most likely differential diagnoses.\n"
        f"Symptoms: {symptoms}\n"
        f"Findings: {findings}\n"
        "Return ONLY a JSON array of 5 diagnosis names, nothing else.\n"
        'Example: ["Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4", "Diagnosis 5"]'
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=150).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


@app.get("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "AgentDDx_UI_Mockup.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "AgentDDx API is running. POST to /analyse"}


@app.post("/analyse")
def analyse(req: AnalyseRequest):
    case = ClinicalCase(id="user_input", source="user", text=req.text)

    # Step 1 — entities
    entities = extract_entities(case)

    # Step 2 — generate candidates dynamically if not provided
    candidates = req.candidates if req.candidates else _generate_candidates(req.text, entities)

    # Step 3 — fetch evidence with relevance filtering
    evidence = fetch_evidence_for_candidates(
        candidates, entities.all_entities,
        max_papers=req.max_papers,
        case_text=req.text,
        use_relevance_filter=True,
    )

    # Step 4 — reasoning
    result = run_reasoning(case, entities, evidence)

    # Format evidence output
    evidence_out = {}
    for dx, res in evidence.items():
        papers = []
        for p in res.articles:
            papers.append({"pmid": p.pmid, "title": p.title, "url": p.url})
        evidence_out[dx] = {"query": res.query, "papers": papers}

    # Format differentials
    differentials = []
    for d in result.differentials:
        papers = evidence_out.get(d.diagnosis, {}).get("papers", [])
        if not papers:
            for key in evidence_out:
                if key.lower() in d.diagnosis.lower() or d.diagnosis.lower() in key.lower():
                    papers = evidence_out[key].get("papers", [])
                    break

        differentials.append({
            "rank":               d.rank,
            "name":               d.diagnosis,
            "confidence":         d.confidence.upper(),
            "confidence_score":   d.confidence_score,
            "reasoning":          d.reasoning,
            "supporting_features": d.supporting_features,
            "features_against":   d.features_against,
            "evidence_summary":   d.evidence_summary,
            "evidence":           papers,
        })

    return {
        "entities": {
            "symptoms":     entities.symptoms,
            "findings":     entities.findings,
            "labs":         entities.labs,
            "medications":  entities.medications,
            "demographics": list(entities.demographics.values()) if isinstance(entities.demographics, dict) else [],
            "genes":        entities.genes,
        },
        "differentials":       differentials,
        "reasoning_trace":     result.reasoning_trace,
        "needs_more_info":     result.needs_more_info,
        "follow_up_question":  result.follow_up_question,
    }
