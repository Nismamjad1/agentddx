"""Shared data models for AgentDDx."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClinicalCase:
    id: str
    source: str
    text: str
    diagnosis: str = ""
    category: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ExtractedEntities:
    symptoms: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    labs: list[str] = field(default_factory=list)
    genes: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    demographics: dict = field(default_factory=dict)

    @property
    def all_entities(self) -> list[str]:
        demo_strs = [str(v) for v in self.demographics.values() if v]
        return self.symptoms + self.findings + self.labs + self.genes + self.medications + demo_strs

    @property
    def is_empty(self) -> bool:
        return not any([self.symptoms, self.findings, self.labs, self.genes, self.medications, self.demographics])


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    url: str
    relevance_score: float = None        # ← ADD THIS LINE


@dataclass
class PubMedResult:
    query: str
    articles: list[PubMedArticle] = field(default_factory=list)
    summary: str = ""
    relevance_scores: list = field(default_factory=list)   # ← ADD THIS LINE

@dataclass
class Differential:
    rank: int
    diagnosis: str
    confidence: str  # High, Medium, Low
    reasoning: str
    confidence_score: int = 50  # 0-100 numerical confidence
    supporting_features: list[str] = field(default_factory=list)
    features_against: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    key_references: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    differentials: list[Differential] = field(default_factory=list)
    reasoning_trace: str = ""
    needs_more_info: bool = False
    follow_up_question: str = ""


@dataclass
class PipelineResult:
    case: Optional[ClinicalCase] = None
    entities: Optional[ExtractedEntities] = None
    evidence: dict = field(default_factory=dict)  # diagnosis -> PubMedResult
    result: Optional[AgentResult] = None
    error: str = ""
