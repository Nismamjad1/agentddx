"""Entity extraction agent — pulls structured medical entities from clinical notes."""
import json
import re
from models import ClinicalCase, ExtractedEntities
from utils import llm_call

EXTRACTION_PROMPT = """You are a clinical NLP system. Extract structured medical entities from the following clinical note.

Return ONLY valid JSON with these exact keys:
{
  "symptoms": ["list of symptoms/complaints"],
  "findings": ["list of physical exam and imaging findings"],
  "labs": ["list of laboratory values with numbers"],
  "genes": ["list of any genetic markers mentioned"],
  "medications": ["list of any medications/drugs mentioned"],
  "demographics": {"age": "", "sex": "", "other": "any relevant social/family history"}
}

Rules:
- Keep entity names concise but specific (e.g., "spiculated mass 3.2 cm" not just "mass")
- Include units for lab values
- Medications include any drugs, therapies, or supplements mentioned
- If a category has no entities, use an empty list or empty dict
- Do NOT add entities that are not in the note

Clinical note:
"""


def extract_entities(case: ClinicalCase) -> ExtractedEntities:
    """Extract medical entities from a clinical case using the shared backbone."""
    raw = llm_call(
        messages=[
            {"role": "system", "content": "You are a precise clinical NLP system. Output only valid JSON."},
            {"role": "user", "content": EXTRACTION_PROMPT + case.text},
        ],
        max_tokens=1000,
    ).strip()

    # Parse JSON — handle markdown code blocks
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            json_str = match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Fix common LLM JSON errors: trailing commas, single quotes
                fixed = re.sub(r',\s*}', '}', json_str)  # trailing comma before }
                fixed = re.sub(r',\s*]', ']', fixed)      # trailing comma before ]
                fixed = fixed.replace("'", '"')            # single → double quotes
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    return ExtractedEntities()
        else:
            return ExtractedEntities()

    demographics = data.get("demographics", {})
    if isinstance(demographics, list):
        demographics = {"info": ", ".join(str(d) for d in demographics)}

    return ExtractedEntities(
        symptoms=data.get("symptoms", []),
        findings=data.get("findings", []),
        labs=data.get("labs", []),
        genes=data.get("genes", []),
        medications=data.get("medications", []),
        demographics=demographics,
    )
