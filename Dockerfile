# syntax=docker/dockerfile:1
#
# AgentDDx — reproducible environment for the paper's evaluation.
#
# Two targets:
#
#   verify  (default, built last)  Recompute every number in the paper from
#                      the released logs. Needs no dependencies and no API
#                      key: the verification script is pure standard library,
#                      so this image builds in seconds and stays near 150 MB.
#
#   full                           Adds the runtime dependencies needed to
#                      re-run the evaluation against OpenRouter and PubMed.
#
# Build and verify:
#     docker build -t agentddx .
#     docker run --rm agentddx
#
# Re-run the evaluation from scratch (hours, needs a key):
#     docker build --target full -t agentddx:full .
#     docker run --rm --env-file .env -v "$PWD/results:/app/results" \
#         agentddx:full python evaluate.py --n 1273 --condition llm_only

FROM python:3.11-slim-bookworm AS base

# Fail fast and keep the image small.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Run as a non-root user; container writes land in results/ only.
RUN useradd --create-home --uid 1000 agentddx


# ── full ──────────────────────────────────────────────────────────────────
# Everything needed to re-run the evaluation end to end. Dependencies are
# installed from the lock file so the environment is reproducible; pass
# --build-arg REQUIREMENTS=requirements.txt for the loosely-pinned set.
FROM base AS full

ARG REQUIREMENTS=requirements.lock
COPY requirements.txt requirements.lock ./
RUN pip install -r "${REQUIREMENTS}"

COPY --chown=agentddx:agentddx . .

# HuggingFace datasets caches the benchmark here; mount a volume to persist it
# across runs rather than re-downloading MedQA each time.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p "${HF_HOME}" && chown -R agentddx:agentddx "${HF_HOME}"

USER agentddx
CMD ["python", "scripts/verify_paper_numbers.py"]


# ── verify ────────────────────────────────────────────────────────────────
# Dependency-free target. scripts/verify_paper_numbers.py imports nothing
# outside the standard library, so no pip install is needed to check the
# paper's claims.
FROM base AS verify

COPY --chown=agentddx:agentddx scripts/ ./scripts/
COPY --chown=agentddx:agentddx results/ ./results/
COPY --chown=agentddx:agentddx README.md LICENSE ./

USER agentddx
CMD ["python", "scripts/verify_paper_numbers.py"]
