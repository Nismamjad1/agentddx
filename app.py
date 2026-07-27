"""AgentDDx — Agentic Differential Diagnosis System.

Run: streamlit run app.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from models import ClinicalCase, ExtractedEntities, Differential, AgentResult
from config import SAMPLE_CASES
from pipeline import run_pipeline

# ─────────────── Page config ───────────────
st.set_page_config(page_title="AgentDDx", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

# ─────────────── Theme — light, warm, medical ───────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Reset & globals */
    .stApp { background: #FAFBFD; font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.2rem; max-width: 1320px; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Top bar ── */
    .topbar {
        display: flex; align-items: center; justify-content: space-between;
        padding: 14px 0 14px; margin-bottom: 4px;
        border-bottom: 1px solid #EEF0F4;
    }
    .topbar-left { display: flex; align-items: center; gap: 14px; }
    .topbar-logo {
        width: 38px; height: 38px; background: linear-gradient(135deg, #3B82F6, #6366F1);
        border-radius: 11px; display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 900; font-size: 13px; letter-spacing: -0.5px;
    }
    .topbar-title { font-size: 19px; font-weight: 800; color: #1F2937; letter-spacing: -0.5px; }
    .topbar-sub { font-size: 11.5px; color: #94A3B8; font-weight: 500; margin-top: 1px; }
    .topbar-badge {
        font-size: 10px; background: #EEF2FF; color: #4F46E5;
        padding: 4px 12px; border-radius: 99px; font-weight: 700; letter-spacing: 0.2px;
    }

    /* ── Pipeline stepper ── */
    .stepper {
        display: flex; align-items: center; gap: 0;
        padding: 14px 20px; margin: 8px 0 16px;
        background: white; border-radius: 14px;
        border: 1px solid #EEF0F4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .step-item { display: flex; align-items: center; gap: 9px; flex: 1; }
    .step-dot {
        width: 30px; height: 30px; min-width: 30px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 800;
        border: 2px solid #E2E8F0; color: #CBD5E1; background: #F8FAFC;
        transition: all 0.3s ease;
    }
    .step-dot.active {
        background: #3B82F6; border-color: #3B82F6; color: white;
        animation: stepPulse 1.8s ease-in-out infinite;
    }
    .step-dot.done { background: #10B981; border-color: #10B981; color: white; }
    .step-dot.error { background: #F43F5E; border-color: #F43F5E; color: white; }
    .step-name { font-size: 12px; color: #CBD5E1; font-weight: 600; }
    .step-name.active { color: #3B82F6; font-weight: 700; }
    .step-name.done { color: #10B981; }
    .step-arrow { color: #E2E8F0; margin: 0 4px; font-size: 13px; }
    @keyframes stepPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.25); }
        50% { box-shadow: 0 0 0 8px rgba(59,130,246,0); }
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 11px; font-weight: 700; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.7px;
        padding: 6px 0 8px; margin-bottom: 8px;
    }

    /* ── Entity tags ── */
    .entity-tag {
        display: inline-block; font-size: 12px; padding: 5px 11px;
        border-radius: 99px; font-weight: 600; margin: 2px 3px;
    }
    .tag-symptom { background: #FFF1F2; color: #E11D48; }
    .tag-finding { background: #F5F3FF; color: #7C3AED; }
    .tag-lab { background: #EFF6FF; color: #2563EB; }
    .tag-demo { background: #ECFDF5; color: #059669; }
    .tag-gene { background: #FFFBEB; color: #D97706; }

    /* ── DDx card ── */
    .ddx-card {
        background: white; border: 1px solid #EEF0F4; border-radius: 14px;
        padding: 20px 22px; margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        transition: all 0.2s;
    }
    .ddx-card:hover { border-color: #D1D5DB; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .ddx-header { display: flex; align-items: center; gap: 14px; margin-bottom: 2px; }
    .ddx-rank {
        width: 38px; height: 38px; min-width: 38px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 900;
    }
    .rank-1 { background: #ECFDF5; color: #059669; }
    .rank-2 { background: #FFFBEB; color: #D97706; }
    .rank-3 { background: #FFF1F2; color: #E11D48; }
    .rank-other { background: #F1F5F9; color: #94A3B8; }
    .ddx-name { font-size: 16px; font-weight: 700; color: #1F2937; flex: 1; letter-spacing: -0.2px; }
    .conf-badge {
        font-size: 10px; font-weight: 800; padding: 4px 13px;
        border-radius: 99px; letter-spacing: 0.3px;
    }
    .conf-high { background: #ECFDF5; color: #059669; }
    .conf-med { background: #FFFBEB; color: #D97706; }
    .conf-low { background: #FFF1F2; color: #E11D48; }

    /* Reasoning block */
    .ddx-reasoning {
        font-size: 13.5px; line-height: 1.75; color: #475569;
        margin: 12px 0 14px; padding: 14px 16px;
        background: #F8FAFC; border-radius: 10px;
        border-left: 3px solid #3B82F6;
    }

    /* Feature tags */
    .features-section { margin-bottom: 12px; }
    .features-label {
        font-size: 10px; font-weight: 700; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .feature-tag {
        display: inline-block; font-size: 11.5px; padding: 4px 10px;
        border-radius: 8px; font-weight: 600; margin: 2px 3px;
    }
    .feat-for { background: #ECFDF5; color: #059669; }
    .feat-against { background: #FFF1F2; color: #E11D48; }

    /* Evidence section */
    .evidence-section {
        margin-top: 12px; padding-top: 12px;
        border-top: 1px solid #F1F5F9;
    }
    .evidence-label {
        font-size: 10px; font-weight: 700; color: #94A3B8;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
    }
    .evidence-summary {
        font-size: 12.5px; line-height: 1.6; color: #64748B;
        margin-bottom: 8px; font-style: italic;
    }
    .evidence-link {
        display: flex; align-items: flex-start; gap: 8px;
        font-size: 12.5px; color: #3B82F6; text-decoration: none;
        padding: 7px 10px; border-radius: 8px;
        transition: background 0.15s; line-height: 1.4;
        margin-bottom: 2px;
    }
    .evidence-link:hover { background: #EFF6FF; }
    .evidence-link .ev-icon { flex-shrink: 0; margin-top: 1px; }
    .evidence-link .ev-title { flex: 1; }
    .evidence-link .ev-pmid { color: #94A3B8; font-size: 11px; white-space: nowrap; }

    /* ── Follow-up bar ── */
    .followup-bar {
        background: #FFFBEB; border: 1px solid #FDE68A;
        border-radius: 12px; padding: 16px 20px; margin-top: 14px;
    }
    .followup-label {
        font-size: 10px; font-weight: 800; color: #D97706;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }
    .followup-text {
        font-size: 14px; color: #92400E; font-weight: 500; line-height: 1.5;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        font-size: 11px; color: #94A3B8; text-align: center;
        padding: 14px 0; margin-top: 20px;
        border-top: 1px solid #F1F5F9;
    }

    /* ── Note display ── */
    .note-display {
        background: white; border: 1px solid #EEF0F4; border-radius: 12px;
        padding: 18px; font-size: 13.5px; line-height: 1.8; color: #374151;
        min-height: 140px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6, #6366F1); color: white; border: none;
        border-radius: 11px; font-weight: 700; font-size: 15px;
        padding: 12px 24px; width: 100%;
        transition: all 0.2s; letter-spacing: -0.2px;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 14px rgba(59,130,246,0.3);
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }
    .stButton > button[disabled] { background: #CBD5E1; box-shadow: none; }

    .stTextArea textarea {
        border-radius: 12px; border: 1px solid #E2E8F0;
        font-size: 13.5px; line-height: 1.75; background: white;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6; box-shadow: 0 0 0 3px rgba(59,130,246,0.08);
    }
    .stSelectbox > div > div { border-radius: 10px; }
    .stRadio > div { gap: 8px; }
    .stTextInput > div > div > input { border-radius: 10px; }

    /* Empty state */
    .empty-state {
        text-align: center; padding: 70px 30px;
    }
    .empty-icon { font-size: 52px; margin-bottom: 16px; opacity: 0.35; }
    .empty-title { font-size: 16px; font-weight: 700; color: #475569; margin-bottom: 8px; }
    .empty-desc {
        font-size: 13px; color: #94A3B8; line-height: 1.6;
        max-width: 340px; margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────── Session state ───────────────
defaults = {
    "pipeline_steps": {}, "entities": None, "result": None,
    "evidence": {}, "running": False, "round": 0,
    "case_text": "", "additional_context": "", "error_msg": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────── Top bar ───────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-logo">Dx</div>
        <div>
            <div class="topbar-title">AgentDDx</div>
            <div class="topbar-sub">Agentic Differential Diagnosis</div>
        </div>
    </div>
    <span class="topbar-badge">MICCAI 2026</span>
</div>
""", unsafe_allow_html=True)


# ─────────────── Stepper ───────────────
def render_stepper():
    steps = [("input","Input"),("entities","Entities"),("candidates","Candidates"),("pubmed","PubMed"),("reasoning","Reasoning")]
    ps = st.session_state.pipeline_steps
    parts = []
    for i, (key, label) in enumerate(steps):
        s = ps.get(key, "")
        dc = "step-dot" + (" done" if s=="done" else " active" if s=="running" else " error" if s=="error" else "")
        nc = "step-name" + (" done" if s=="done" else " active" if s=="running" else "")
        icon = "✓" if s=="done" else "⋯" if s=="running" else "✗" if s=="error" else str(i+1)
        parts.append(f'<div class="step-item"><div class="{dc}">{icon}</div><span class="{nc}">{label}</span></div>')
        if i < len(steps)-1:
            parts.append('<span class="step-arrow">→</span>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)

render_stepper()


# ─────────────── Layout ───────────────
left_col, right_col = st.columns([2, 3], gap="large")


# ═══════ LEFT: Input + Entities ═══════
with left_col:
    st.markdown('<div class="section-label">Clinical Note</div>', unsafe_allow_html=True)

    source = st.radio("Source", ["Sample Cases", "Free Text"], horizontal=True, label_visibility="collapsed")

    if source == "Sample Cases":
        case_idx = st.selectbox(
            "case", range(len(SAMPLE_CASES)),
            format_func=lambda i: f"{SAMPLE_CASES[i]['category']}  ·  {SAMPLE_CASES[i]['title']}",
            label_visibility="collapsed",
        )
        case_text = SAMPLE_CASES[case_idx]["text"]
        st.markdown(f'<div class="note-display">{case_text}</div>', unsafe_allow_html=True)
    else:
        case_text = st.text_area("note", height=200, placeholder="A 45-year-old female presents with...", label_visibility="collapsed")

    # API key — uses OpenRouter (set in .env)
    from utils import has_api_key
    api_key = has_api_key()
    if not api_key:
        entered_key = st.text_input("OpenRouter API key (get one at openrouter.ai)", type="password", placeholder="sk-or-v1-...")
        if entered_key:
            os.environ["OPENROUTER_API_KEY"] = entered_key
            api_key = True

    can_run = bool(case_text.strip()) and api_key and not st.session_state.running
    if st.button("▶  Analyse Case" if not st.session_state.running else "⏳  Analysing...", disabled=not can_run):
        st.session_state.running = True
        st.session_state.round += 1
        st.session_state.pipeline_steps = {"input": "done"}
        st.session_state.entities = None
        st.session_state.result = None
        st.session_state.evidence = {}
        st.session_state.error_msg = ""
        st.session_state.case_text = case_text
        st.rerun()

    # ── Entities ──
    st.markdown('<div class="section-label" style="margin-top:14px;">Extracted Entities</div>', unsafe_allow_html=True)

    ent = st.session_state.entities
    if ent and not ent.is_empty:
        tag_map = {
            "Symptoms": ("symptoms", "tag-symptom"),
            "Findings": ("findings", "tag-finding"),
            "Labs": ("labs", "tag-lab"),
            "Demographics": ("demographics", "tag-demo"),
            "Genes": ("genes", "tag-gene"),
        }
        for label, (attr, cls) in tag_map.items():
            vals = getattr(ent, attr, None)
            if isinstance(vals, dict):
                vals = [f"{k}: {v}" for k, v in vals.items() if v]
            if vals:
                tags = "".join(f'<span class="entity-tag {cls}">{v}</span>' for v in vals)
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:5px;">{label}</div>'
                    f'<div>{tags}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div style="font-size:12px;color:#CBD5E1;padding:8px 0;">Run analysis to extract entities</div>', unsafe_allow_html=True)


# ═══════ RIGHT: DDx Results ═══════
with right_col:
    round_html = f'<span style="font-size:10px;background:#EEF2FF;color:#4F46E5;padding:3px 10px;border-radius:99px;font-weight:700;margin-left:10px;">Round {st.session_state.round}</span>' if st.session_state.round > 0 else ""
    st.markdown(f'<div class="section-label">Ranked Differential Diagnosis{round_html}</div>', unsafe_allow_html=True)

    res: AgentResult | None = st.session_state.result

    if res and res.differentials:
        for d in res.differentials:
            rank_cls = f"rank-{d.rank}" if d.rank <= 3 else "rank-other"
            conf_cls = {"High":"conf-high","Medium":"conf-med","Low":"conf-low"}.get(d.confidence, "conf-low")

            # ── Build supporting/against features ──
            for_html = ""
            if d.supporting_features:
                tags = "".join(f'<span class="feature-tag feat-for">✓ {f}</span>' for f in d.supporting_features)
                for_html = f'<div class="features-section"><div class="features-label">Supporting</div>{tags}</div>'

            against_html = ""
            if d.features_against:
                tags = "".join(f'<span class="feature-tag feat-against">✗ {f}</span>' for f in d.features_against)
                against_html = f'<div class="features-section"><div class="features-label">Against</div>{tags}</div>'

            # ── Build evidence section (properly structured) ──
            evidence_html = ""
            has_evidence = False

            # Evidence summary from reasoning agent
            summary_html = ""
            if d.evidence_summary:
                summary_html = f'<div class="evidence-summary">{d.evidence_summary}</div>'
                has_evidence = True

            # PubMed articles from evidence dict
            articles_html = ""
            ev_data = st.session_state.evidence.get(d.diagnosis)
            if ev_data and ev_data.articles:
                for art in ev_data.articles[:3]:
                    title_text = art.title[:90] + ("..." if len(art.title) > 90 else "") if art.title else "Untitled"
                    articles_html += (
                        f'<a class="evidence-link" href="{art.url}" target="_blank">'
                        f'<span class="ev-icon">📄</span>'
                        f'<span class="ev-title">{title_text}</span>'
                        f'<span class="ev-pmid">PMID {art.pmid}</span>'
                        f'</a>'
                    )
                has_evidence = True

            # PMID references from reasoning agent (that aren't already shown)
            shown_pmids = set(a.pmid for a in (ev_data.articles[:3] if ev_data and ev_data.articles else []))
            refs_html = ""
            if d.key_references:
                for ref in d.key_references:
                    pmid = ref.replace("PMID:", "").replace("PMID", "").strip()
                    if pmid and pmid not in shown_pmids:
                        refs_html += (
                            f'<a class="evidence-link" href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/" target="_blank">'
                            f'<span class="ev-icon">📄</span>'
                            f'<span class="ev-title">PubMed Reference</span>'
                            f'<span class="ev-pmid">PMID {pmid}</span>'
                            f'</a>'
                        )
                        has_evidence = True

            if has_evidence:
                evidence_html = (
                    f'<div class="evidence-section">'
                    f'<div class="evidence-label">Evidence</div>'
                    f'{summary_html}{articles_html}{refs_html}'
                    f'</div>'
                )

            # ── Render card ──
            st.markdown(f"""
            <div class="ddx-card">
                <div class="ddx-header">
                    <div class="ddx-rank {rank_cls}">#{d.rank}</div>
                    <span class="ddx-name">{d.diagnosis}</span>
                    <span class="conf-badge {conf_cls}">{d.confidence.upper()}</span>
                </div>
                <div class="ddx-reasoning">{d.reasoning}</div>
                {for_html}
                {against_html}
                {evidence_html}
            </div>
            """, unsafe_allow_html=True)

        # Reasoning trace
        if res.reasoning_trace:
            with st.expander("🧠  Full Reasoning Trace", expanded=False):
                st.markdown(f'<div style="font-size:13px;line-height:1.75;color:#475569;">{res.reasoning_trace}</div>', unsafe_allow_html=True)

        # Follow-up question
        if res.needs_more_info and res.follow_up_question:
            st.markdown(f"""
            <div class="followup-bar">
                <div class="followup-label">💬 Agent Follow-Up Question</div>
                <div class="followup-text">{res.follow_up_question}</div>
            </div>
            """, unsafe_allow_html=True)

            answer = st.text_input("Your answer:", placeholder="Type additional clinical information...", key="followup_input")
            if st.button("Submit & Re-analyse", key="followup_btn"):
                if answer.strip():
                    st.session_state.additional_context = answer.strip()
                    st.session_state.running = True
                    st.session_state.round += 1
                    st.session_state.pipeline_steps = {"input": "done"}
                    st.session_state.result = None
                    st.rerun()

    elif st.session_state.running:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🩺</div>
            <div class="empty-title">Analysing clinical case...</div>
            <div class="empty-desc">The agentic pipeline is running. Each step processes sequentially — entities, candidates, evidence, then reasoning.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🩺</div>
            <div class="empty-title">No analysis yet</div>
            <div class="empty-desc">Select or enter a clinical case and press <strong>Analyse Case</strong> to generate a ranked differential diagnosis with evidence.</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)


# ─────────────── Disclaimer ───────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Research prototype</strong> — not for clinical decision-making.
    AgentDDx is an experimental tool for MICCAI 2026 research purposes only.
</div>
""", unsafe_allow_html=True)


# ─────────────── Pipeline execution ───────────────
if st.session_state.running and st.session_state.result is None:
    case = ClinicalCase(id=f"user_{int(time.time())}", source="user", text=st.session_state.case_text)

    def on_step(step_name, status):
        st.session_state.pipeline_steps[step_name] = status

    try:
        result = run_pipeline(case, on_step=on_step, additional_context=st.session_state.get("additional_context", ""))
        st.session_state.entities = result.entities
        st.session_state.evidence = result.evidence or {}
        st.session_state.result = result.result
        if result.error:
            st.session_state.error_msg = result.error
    except Exception as e:
        st.session_state.error_msg = f"Pipeline error: {str(e)}"

    st.session_state.running = False
    st.session_state.additional_context = ""
    st.rerun()
