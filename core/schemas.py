from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


BiasLabel = Literal["Left", "Lean Left", "Center", "Lean Right", "Right", "Mixed", "Undetermined"]


class LLMKeys(BaseModel):
    """Ephemeral keys supplied by the caller.

    The Streamlit app stores these in session state only. The core code
    does not read model-provider keys from environment variables.
    """

    anthropic_token: str | None = None
    openai_token: str | None = None
    google_token: str | None = None
    gnews_token: str | None = None
    ollama_host: str = "http://localhost:11434"


class StructuredQuery(BaseModel):
    query: str
    date_from: str | None = None
    date_to: str | None = None


class Article(BaseModel):
    id: str
    title: str
    url: str
    source: str = "unknown"
    text: str

    @field_validator("text")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("article text cannot be empty")
        return value


class Citation(BaseModel):
    article_id: str
    span_text: str = Field(min_length=8)


class StructuredSummary(BaseModel):
    headline: str
    neutral_summary: str
    key_points: list[str]
    framing_notes: list[Citation]


class StructuredBiasJudgment(BaseModel):
    label: BiasLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[Citation]
    proxy_features: list[str] = Field(default_factory=list)


class StructuredCritique(BaseModel):
    refined_label: BiasLabel
    agree_with_detector: bool
    reasoning: str
    trigger_phrases: list[Citation]
    proxy_notes: str


class ReconciledReport(BaseModel):
    headline: str
    executive_summary: str
    final_label: BiasLabel
    confidence: float = Field(ge=0.0, le=1.0)
    caveats: list[str]
    article_count: int


class StageRecord(BaseModel):
    name: str
    implementation: str
    status: Literal["ok", "warning", "error"] = "ok"
    output: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class PipelineTrace(BaseModel):
    subject: str
    implementation: str
    provider: str
    model: str
    structured_query: StructuredQuery
    articles: list[Article]
    summary: StructuredSummary
    bias_judgment: StructuredBiasJudgment
    critique: StructuredCritique
    report: ReconciledReport
    stages: list[StageRecord]
    citation_errors: list[str] = Field(default_factory=list)
    framework_notes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        caveats = "\n".join(f"- {item}" for item in self.report.caveats)
        return (
            f"# {self.report.headline}\n\n"
            f"**Final label:** {self.report.final_label} "
            f"({self.report.confidence:.2f})\n\n"
            f"{self.report.executive_summary}\n\n"
            "## Detector\n\n"
            f"{self.bias_judgment.label}: {self.bias_judgment.rationale}\n\n"
            "## Critic\n\n"
            f"{self.critique.refined_label}: {self.critique.reasoning}\n\n"
            "## Caveats\n\n"
            f"{caveats}\n"
        )
