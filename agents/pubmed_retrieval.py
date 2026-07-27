"""PubMed retrieval agent — fetches real evidence for candidate diagnoses.

Improvements over baseline:
  1. Structured MeSH-aware query construction (not naive concatenation)
  2. Relevance filtering — LLM scores each abstract against the case
  3. Dual-strategy search: structured query first, simplified fallback
  4. Smarter abstract handling — keeps Results/Conclusions, not just first 1500 chars
"""
import re, json, time
from Bio import Entrez
from models import PubMedArticle, PubMedResult
import config
from utils import llm_call

Entrez.email = config.PUBMED_EMAIL


# ── Query Construction ──────────────────────────────────────────

def _build_mesh_query(diagnosis: str, entities: list[str]) -> str:
    """Use LLM to build a structured PubMed query with MeSH terms."""

    entity_str = ", ".join(entities[:6]) if entities else "none"

    raw = llm_call(
        messages=[
            {"role": "system", "content": (
                "You are a medical librarian. Convert the diagnosis and clinical "
                "features into a PubMed search query using MeSH terms and Boolean "
                "operators. Use [MeSH] tags where appropriate.\n"
                "Rules:\n"
                "- Max 4 terms connected by AND/OR\n"
                "- Prefer MeSH headings over free text\n"
                "- Focus on the diagnosis + 1-2 key distinguishing features\n"
                "- Output ONLY the query string, nothing else\n\n"
                "Example input: Pulmonary embolism, dyspnea, D-dimer elevated, tachycardia\n"
                'Example output: "Pulmonary Embolism"[MeSH] AND ("D-dimer" OR "diagnosis")'
            )},
            {"role": "user", "content": f"Diagnosis: {diagnosis}\nClinical features: {entity_str}"},
        ],
        temperature=0.0,
        max_tokens=80,
    )
    return raw.strip().strip('"\'')



def _simplify_query(query: str) -> str:
    """Fallback: shorten a long query into a PubMed-friendly search."""
    if len(query.split()) <= 8:
        return query

    raw = llm_call(
        messages=[
            {"role": "system", "content": "Convert the input into a short PubMed search query (max 5 terms). Output ONLY the query, nothing else."},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=60,
    )
    return raw.strip().strip('"')


# ── PubMed Search ───────────────────────────────────────────────

def _search_pubmed(query: str, max_results: int = 5) -> list[PubMedArticle]:
    """Search PubMed and return article details."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()

        ids = record.get("IdList", [])
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="xml")
        articles_xml = Entrez.read(handle)
        handle.close()

        articles = []
        for article in articles_xml.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            art = medline.get("Article", {})
            pmid = str(medline.get("PMID", ""))
            title = str(art.get("ArticleTitle", ""))

            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""

            # Smart truncation: keep conclusions if present
            if len(abstract) > 2000:
                abstract = _smart_truncate(abstract, max_len=2000)

            articles.append(PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))
        return articles
    except Exception as e:
        print(f"PubMed search error for '{query}': {e}")
        return []


def _smart_truncate(abstract: str, max_len: int = 2000) -> str:
    """Truncate abstract but preserve Results/Conclusions sections."""
    # Try to find section markers in structured abstracts
    sections = re.split(r'(?i)(BACKGROUND|METHODS|RESULTS|CONCLUSIONS?|OBJECTIVE|PURPOSE)', abstract)

    if len(sections) <= 1:
        # Unstructured abstract — take beginning and end
        half = max_len // 2
        return abstract[:half] + " [...] " + abstract[-half:]

    # Structured: prioritize Results and Conclusions
    priority_text = ""
    other_text = ""
    current_section = ""

    for part in sections:
        upper = part.strip().upper()
        if upper in ("BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS", "CONCLUSION", "OBJECTIVE", "PURPOSE"):
            current_section = upper
            continue
        if current_section in ("RESULTS", "CONCLUSIONS", "CONCLUSION"):
            priority_text += f" {current_section}: {part.strip()}"
        else:
            other_text += f" {current_section}: {part.strip()}"

    if priority_text:
        result = priority_text.strip()
        remaining = max_len - len(result)
        if remaining > 200:
            result = other_text[:remaining].strip() + " " + result
        return result[:max_len]

    return abstract[:max_len]


# ── Relevance Filtering ────────────────────────────────────────

def _score_relevance(article: PubMedArticle, diagnosis: str, case_text: str) -> float:
    """Score how relevant an article is to this specific case (0.0–1.0)."""
    if not article.abstract:
        return 0.5  # Neutral if no abstract to score

    raw = llm_call(
        messages=[
            {"role": "system", "content": (
                "Rate how relevant this PubMed abstract is to the clinical case and "
                "suspected diagnosis. Consider: does it discuss the same condition? "
                "Does it provide diagnostic criteria, clinical features, or treatment "
                "evidence that could help confirm or rule out this diagnosis?\n"
                "Return ONLY a JSON object: {\"score\": 0.0-1.0, \"reason\": \"one sentence\"}"
            )},
            {"role": "user", "content": (
                f"Suspected diagnosis: {diagnosis}\n"
                f"Case summary: {case_text[:300]}\n"
                f"Article title: {article.title}\n"
                f"Abstract excerpt: {article.abstract[:400]}"
            )},
        ],
        temperature=0.0,
        max_tokens=80,
    ).strip()
    try:
        # Handle markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return float(data.get("score", 0.5))
    except (json.JSONDecodeError, ValueError):
        return 0.5


def _filter_by_relevance(
    articles: list[PubMedArticle],
    diagnosis: str,
    case_text: str,
    threshold: float = 0.4,
    max_keep: int = 3,
) -> list[PubMedArticle]:
    """Keep only articles that score above the relevance threshold."""
    if not articles:
        return []

    scored = []
    for art in articles:
        score = _score_relevance(art, diagnosis, case_text)
        art.relevance_score = score          # ← ADD THIS LINE
        scored.append((score, art))
        time.sleep(0.1)  # Rate limiting

    # Sort by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep articles above threshold, up to max_keep
    filtered = [art for score, art in scored if score >= threshold][:max_keep]

    # Always keep at least 1 article if any were found
    if not filtered and scored:
        filtered = [scored[0][1]]

    return filtered


# ── Main Interface ──────────────────────────────────────────────

def fetch_evidence_for_candidates(
    candidates: list[str],
    entities: list[str],
    max_papers: int | None = None,
    case_text: str = "",
    use_relevance_filter: bool = True,
) -> dict[str, PubMedResult]:
    """
    Fetch PubMed evidence for each candidate diagnosis.

    Two-pass strategy:
      1. MeSH-structured query → fetch up to max_papers * 2 articles
      2. Relevance filter → keep only the top max_papers relevant ones

    Args:
        candidates: List of diagnosis names to search for
        entities: Extracted clinical entities for context
        max_papers: Max papers to keep per diagnosis (after filtering)
        case_text: Original clinical note (used for relevance scoring)
        use_relevance_filter: Whether to score and filter articles
    """
    max_papers = max_papers or config.MAX_PAPERS_PER_DIAGNOSIS
    results = {}

    for diagnosis in candidates:
        # Strategy 1: Structured MeSH query
        mesh_query = _build_mesh_query(diagnosis, entities)
        fetch_count = max_papers * 2 if use_relevance_filter else max_papers
        articles = _search_pubmed(mesh_query, max_results=fetch_count)

        # Strategy 2: Fallback with simplified diagnosis name
        if len(articles) < 2:
            fallback_query = _simplify_query(f"{diagnosis} diagnosis clinical presentation")
            fallback_articles = _search_pubmed(fallback_query, max_results=fetch_count)
            # Merge, avoiding duplicate PMIDs
            seen_pmids = {a.pmid for a in articles}
            for art in fallback_articles:
                if art.pmid not in seen_pmids:
                    articles.append(art)
                    seen_pmids.add(art.pmid)

        # Relevance filtering
        if use_relevance_filter and case_text and articles:
            articles = _filter_by_relevance(
                articles, diagnosis, case_text,
                threshold=0.4, max_keep=max_papers,
            )
        else:
            articles = articles[:max_papers]

        summary = ""
        if articles:
            titles = "; ".join(a.title for a in articles[:3])
            summary = f"Found {len(articles)} relevant articles: {titles}"

        kept_scores = [a.relevance_score for a in articles
                       if getattr(a, "relevance_score", None) is not None]
        results[diagnosis] = PubMedResult(
            query=mesh_query,
            articles=articles,
            summary=summary,
            relevance_scores=kept_scores,       # ← ADD THIS ARG

        )
        time.sleep(0.3)  # Rate limiting between candidates

    return results
