"""
AgriDoctor – AI-powered Crop Disease Diagnosis & Advisory
==========================================================
Two-agent workflow:
  1. Pathologist Agent  – analyses a crop/leaf image and produces a diagnosis.
  2. Extension Officer Agent – converts the diagnosis into a farmer-friendly
     remedial-action report.

Both agents call the IBM watsonx.ai REST API with the IBM Granite 4 H Small
model.  The Pathologist Agent sends the image as a base-64 encoded data URL
alongside the prompt; the Extension Officer Agent sends plain text.
"""

import base64
import io
import json
import textwrap
import time
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WATSONX_ENDPOINT = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "5247ecfe-b435-4f50-93e9-2ed52faedf1d"
MODEL_ID = "ibm/granite-4-h-small"          # Granite 4 H Small identifier on watsonx
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Text generation endpoint (supports multimodal for Granite 4)
GENERATE_URL = f"{WATSONX_ENDPOINT}/ml/v1/text/chat?version=2024-05-31"

# ---------------------------------------------------------------------------
# IBM IAM Token Helper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3000, show_spinner=False)
def get_iam_token(api_key: str) -> str:
    """Exchange an IBM Cloud API key for a short-lived IAM bearer token."""
    resp = requests.post(
        IAM_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": api_key,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"IAM token request failed ({resp.status_code}): {resp.text}"
        )
    return resp.json()["access_token"]


# ---------------------------------------------------------------------------
# Core API call  (now returns metrics alongside the text)
# ---------------------------------------------------------------------------

def call_watsonx(
    api_key: str,
    messages: list,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> dict:
    """
    Send a chat-completion request to watsonx.ai and return a dict:
      {
        "text":             str   – assistant response,
        "latency_s":        float – wall-clock seconds for the API call,
        "input_tokens":     int,
        "output_tokens":    int,
        "total_tokens":     int,
        "stop_reason":      str,
        "model_id":         str,
        "temperature":      float,
        "max_tokens":       int,
      }
    """
    token = get_iam_token(api_key)

    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    resp = requests.post(
        GENERATE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=120,
    )
    latency = time.perf_counter() - t0

    if resp.status_code != 200:
        raise RuntimeError(
            f"watsonx API error ({resp.status_code}): {resp.text}"
        )

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
        stop_reason = data["choices"][0].get("finish_reason", "unknown")
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected response structure: {json.dumps(data)}")

    usage = data.get("usage", {})
    input_tokens  = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens  = usage.get("total_tokens", input_tokens + output_tokens)

    return {
        "text":          text,
        "latency_s":     round(latency, 3),
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
        "stop_reason":   stop_reason,
        "model_id":      MODEL_ID,
        "temperature":   temperature,
        "max_tokens":    max_tokens,
    }


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def extract_image_context(uploaded_file) -> dict:
    """
    Extract visual metadata from the uploaded image using PIL.
    Returns a dict with filename, dimensions, mode, dominant colours, and
    brightness — used as structured context for the text-based LLM.
    """
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    filename = uploaded_file.name

    # Sample a grid of pixels for colour analysis
    img_small = img.resize((50, 50))
    pixels = list(img_small.getdata())

    # Average colour
    avg_r = int(sum(p[0] for p in pixels) / len(pixels))
    avg_g = int(sum(p[1] for p in pixels) / len(pixels))
    avg_b = int(sum(p[2] for p in pixels) / len(pixels))

    # Brightness (0-255)
    brightness = int((avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114))

    # Colour tone description
    if avg_g > avg_r and avg_g > avg_b:
        if avg_g > 120:
            tone = "predominantly green (healthy-looking foliage)"
        else:
            tone = "dark green (possible shade stress or fungal darkening)"
    elif avg_r > avg_g and avg_r > avg_b:
        tone = "reddish/brownish (possible leaf scorch, rust, or blight)"
    elif avg_r > 150 and avg_g > 130 and avg_b < 100:
        tone = "yellowish-green (possible chlorosis or nutrient deficiency)"
    elif avg_r > 160 and avg_g > 140 and avg_b < 80:
        tone = "yellow (likely chlorosis, nitrogen deficiency, or viral infection)"
    elif avg_r < 80 and avg_g < 80 and avg_b < 80:
        tone = "very dark/black (possible sooty mould, necrosis, or severe blight)"
    else:
        tone = f"mixed tones (R:{avg_r} G:{avg_g} B:{avg_b})"

    brightness_desc = (
        "well-lit" if brightness > 120
        else "dim/underexposed (may affect analysis accuracy)"
    )

    return {
        "filename": filename,
        "dimensions": f"{width}x{height} pixels",
        "colour_tone": tone,
        "avg_rgb": f"R:{avg_r} G:{avg_g} B:{avg_b}",
        "brightness": f"{brightness}/255 ({brightness_desc})",
    }


# ---------------------------------------------------------------------------
# Agent 1 – Pathologist Agent
# ---------------------------------------------------------------------------

PATHOLOGIST_SYSTEM_PROMPT = textwrap.dedent("""
You are an expert plant pathologist and crop disease diagnostician with decades
of field and laboratory experience. A farmer has submitted a crop or plant-leaf
image for diagnosis. You will be given:
  1. Image metadata (dimensions, colour analysis, brightness) extracted
     automatically from the photo.
  2. A description of symptoms as observed by the farmer.

Use both sources of information to deliver a rigorous, evidence-based
preliminary diagnosis.

GUIDELINES:
- Base your analysis on the colour tone data, brightness, and farmer-reported
  symptoms. Colour shifts (yellowing, browning, darkening) are key indicators.
- Identify the most likely disease(s) and state the evidence that supports each.
- Provide a confidence level: High / Medium / Low.
- List up to three alternative differential diagnoses when relevant.
- If the information is insufficient for a reliable assessment, say so clearly
  and ask the farmer for more details rather than guessing.
- Always include the disclaimer that this is an AI-generated preliminary
  assessment and is NOT a confirmed laboratory diagnosis.

OUTPUT FORMAT (use exactly these headings):
## Primary Diagnosis
## Observed Symptoms
## Confidence Level
## Possible Causes
## Differential Diagnoses
## Disclaimer
""").strip()


def run_pathologist_agent(
    api_key: str,
    image_context: dict,
    symptom_description: str,
) -> dict:
    """
    Agent 1: sends image metadata + farmer symptom description to Granite 4
    and returns the full metrics dict from call_watsonx().
    """
    context_block = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in image_context.items()
    )

    user_message = (
        "Please analyse the following crop image data and farmer observations, "
        "then provide a detailed disease diagnosis following the specified "
        "output format.\n\n"
        "=== IMAGE ANALYSIS DATA ===\n"
        f"{context_block}\n\n"
        "=== FARMER-REPORTED SYMPTOMS ===\n"
        f"{symptom_description if symptom_description.strip() else 'No additional description provided by farmer.'}\n\n"
        "Based on the colour analysis and symptoms above, provide your diagnosis."
    )

    messages = [
        {"role": "system", "content": PATHOLOGIST_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    result = call_watsonx(api_key, messages, max_tokens=1200, temperature=0.2)
    result["agent"] = "Pathologist"
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


# ---------------------------------------------------------------------------
# Agent 2 – Extension Officer Agent
# ---------------------------------------------------------------------------

EXTENSION_OFFICER_SYSTEM_PROMPT = textwrap.dedent("""
You are a friendly and knowledgeable agricultural extension officer with
extensive experience advising smallholder and commercial farmers.  A plant
pathologist has provided you with a crop disease diagnosis report.  Your job is
to convert that technical report into a clear, practical, farmer-friendly
advisory in simple English that even a first-time farmer can understand.

GUIDELINES:
- Use plain, non-technical language wherever possible.
- Provide immediate remedial actions the farmer can take right now.
- Include preventive measures to stop future outbreaks.
- Give crop-care and general best-practice recommendations.
- When chemical treatments are appropriate, mention only well-known, widely
  available options; refer to product labels and local regulations for exact
  dosages – DO NOT invent specific dosages.
- Always recommend Personal Protective Equipment (PPE) when chemicals are
  mentioned.
- End with a safety notice and a recommendation to seek professional help if
  symptoms persist or worsen.
- If the pathologist indicated the image was unclear, advise the farmer to
  obtain a clearer image or consult a local expert directly.

OUTPUT FORMAT (use exactly these headings):
## Summary for the Farmer
## Immediate Remedial Actions
## Preventive Measures
## Crop-Care Recommendations
## Treatment Options
## Safety Warnings
## When to Seek Expert Help
""").strip()


def run_extension_officer_agent(api_key: str, diagnosis_report: str) -> dict:
    """
    Agent 2: receives the Pathologist's report and produces a farmer-friendly
    advisory. Returns the full metrics dict from call_watsonx().
    """
    messages = [
        {
            "role": "system",
            "content": EXTENSION_OFFICER_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "The plant pathologist has provided the following crop disease "
                "diagnosis report. Please convert it into a practical, "
                "farmer-friendly advisory:\n\n"
                f"{diagnosis_report}"
            ),
        },
    ]

    result = call_watsonx(api_key, messages, max_tokens=1400, temperature=0.4)
    result["agent"] = "Extension Officer"
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return result


# ---------------------------------------------------------------------------
# Report Builder
# ---------------------------------------------------------------------------

def build_final_report(
    diagnosis: str,
    advisory: str,
    filename: str,
) -> str:
    """Combine both agent outputs into a single downloadable text report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "=" * 70

    report = textwrap.dedent(f"""
        {divider}
        AGRI DOCTOR – CROP DISEASE ADVISORY REPORT
        Generated: {timestamp}
        Image File: {filename}
        {divider}

        SECTION 1 – PATHOLOGIST AGENT: DISEASE DIAGNOSIS
        {divider}
        {diagnosis}

        {divider}
        SECTION 2 – EXTENSION OFFICER AGENT: FARMER ADVISORY
        {divider}
        {advisory}

        {divider}
        IMPORTANT NOTICE
        {divider}
        This report is an AI-generated preliminary assessment produced by the
        AgriDoctor application using IBM Granite 4 H Small on IBM watsonx.ai.
        It is NOT a confirmed laboratory diagnosis. Farmers are strongly advised
        to consult a certified plant pathologist or local agricultural extension
        officer for confirmation, especially before applying chemical treatments.
        Always follow local agricultural regulations and pesticide label
        instructions. Use appropriate Personal Protective Equipment (PPE) when
        handling agrochemicals.
        {divider}
    """).strip()

    return report


# ---------------------------------------------------------------------------
# LLM Evaluation Helpers
# ---------------------------------------------------------------------------

def _score_response(text: str, agent: str) -> dict:
    """
    Derive lightweight quality-proxy scores from the response text.
    All scores are 0-100.

    - completeness : fraction of expected section headings present
    - verbosity    : log-scaled word count normalised to 100
    - coherence    : ratio of sentences that are >4 words (proxy)
    """
    if agent == "Pathologist":
        expected_headings = [
            "Primary Diagnosis", "Observed Symptoms", "Confidence Level",
            "Possible Causes", "Differential Diagnoses", "Disclaimer",
        ]
    else:
        expected_headings = [
            "Summary for the Farmer", "Immediate Remedial Actions",
            "Preventive Measures", "Crop-Care Recommendations",
            "Treatment Options", "Safety Warnings", "When to Seek Expert Help",
        ]

    found = sum(1 for h in expected_headings if h.lower() in text.lower())
    completeness = round(found / len(expected_headings) * 100)

    words = text.split()
    word_count = len(words)
    import math
    verbosity = min(100, round(math.log1p(word_count) / math.log1p(600) * 100))

    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    long_sentences = sum(1 for s in sentences if len(s.split()) > 4)
    coherence = round(long_sentences / max(len(sentences), 1) * 100)

    overall = round((completeness * 0.5 + verbosity * 0.2 + coherence * 0.3))

    return {
        "completeness": completeness,
        "verbosity":    verbosity,
        "coherence":    coherence,
        "overall":      overall,
        "word_count":   word_count,
    }


def _bar(value: int, max_val: int = 100, width: int = 20) -> str:
    """Return a Unicode progress-bar string."""
    filled = round(value / max_val * width)
    return "█" * filled + "░" * (width - filled)


def _colour_for(score: int) -> str:
    if score >= 80:
        return "#2e7d32"
    if score >= 55:
        return "#f57c00"
    return "#c62828"


# ---------------------------------------------------------------------------
# Dashboard Tab
# ---------------------------------------------------------------------------

def render_eval_dashboard():
    """Render the LLM Evaluation Dashboard tab."""

    st.title("📊 LLM Evaluation Dashboard")
    st.markdown(
        "Real-time performance metrics for the **IBM Granite 4 H Small** model "
        "running on **IBM watsonx.ai** — collected during this session's "
        "AgriDoctor diagnoses."
    )

    # ── Model identity card ───────────────────────────────────────────────────
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("🤖 Model", "IBM Granite 4 H Small")
        c2.metric("🌐 Endpoint", "us-south.ml.cloud.ibm.com")
        c3.metric("📡 API Version", "2024-05-31")

    st.divider()

    calls = st.session_state.get("eval_calls", [])

    if not calls:
        st.info(
            "📌 No evaluation data yet.  "
            "Run a crop diagnosis on the **🌾 Diagnosis** tab first — "
            "metrics will appear here automatically.",
            icon="ℹ️",
        )

        # ── Static reference card (always visible) ────────────────────────────
        st.subheader("📋 Evaluation Metrics Reference")
        ref_cols = st.columns(2)
        with ref_cols[0]:
            with st.container(border=True):
                st.markdown("**Latency (s)**")
                st.caption(
                    "Wall-clock time from API call start to first byte received. "
                    "Lower is better; typical range 3–15 s for Granite 4 H Small."
                )
            with st.container(border=True):
                st.markdown("**Input / Output Tokens**")
                st.caption(
                    "Tokens consumed from the prompt vs. generated in the "
                    "response. Reported by the watsonx.ai usage object."
                )
            with st.container(border=True):
                st.markdown("**Stop Reason**")
                st.caption(
                    "`stop` = natural end of response. "
                    "`length` = truncated at max_tokens limit. "
                    "`content_filter` = safety filter triggered."
                )
        with ref_cols[1]:
            with st.container(border=True):
                st.markdown("**Completeness Score**")
                st.caption(
                    "Fraction of expected section headings present in the "
                    "response (0–100). A score of 100 means all required "
                    "headings were generated."
                )
            with st.container(border=True):
                st.markdown("**Coherence Score**")
                st.caption(
                    "Proportion of sentences longer than 4 words — a lightweight "
                    "proxy for grammatical completeness (0–100)."
                )
            with st.container(border=True):
                st.markdown("**Overall Quality Score**")
                st.caption(
                    "Weighted composite: 50 % Completeness + 30 % Coherence + "
                    "20 % Verbosity. Range 0–100."
                )

        st.subheader("⚙️ Model Configuration Reference")
        cfg_data = {
            "Parameter": ["model_id", "temperature (Agent 1)", "temperature (Agent 2)",
                           "max_tokens (Agent 1)", "max_tokens (Agent 2)",
                           "API version", "IAM token TTL"],
            "Value": [MODEL_ID, "0.2", "0.4", "1 200", "1 400",
                      "2024-05-31", "3 000 s"],
            "Notes": [
                "IBM Granite 4 H Small multimodal",
                "Lower → more deterministic diagnosis",
                "Slightly higher → more natural advisory tone",
                "~900 words output capacity",
                "~1 050 words output capacity",
                "Latest stable chat-completion version",
                "Cached IAM bearer token lifetime",
            ],
        }
        st.table(cfg_data)
        return

    # ── Aggregate stats ───────────────────────────────────────────────────────
    total_calls    = len(calls)
    total_sessions = len({c.get("session_id", 0) for c in calls})
    avg_latency    = round(sum(c["latency_s"] for c in calls) / total_calls, 2)
    total_tokens   = sum(c["total_tokens"] for c in calls)
    avg_quality    = round(sum(c["scores"]["overall"] for c in calls) / total_calls)

    st.subheader("📈 Session Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total API Calls",    total_calls)
    m2.metric("Diagnosis Sessions", total_sessions)
    m3.metric("Avg Latency",        f"{avg_latency} s")
    m4.metric("Total Tokens Used",  f"{total_tokens:,}")
    m5.metric("Avg Quality Score",  f"{avg_quality}/100")

    st.divider()

    # ── Per-call detail cards ─────────────────────────────────────────────────
    st.subheader("🔍 Per-Call Breakdown")

    for i, call in enumerate(reversed(calls), 1):
        agent   = call["agent"]
        scores  = call["scores"]
        icon    = "🔬" if agent == "Pathologist" else "📋"
        session = call.get("session_id", "—")

        with st.expander(
            f"{icon} Call #{total_calls - i + 1} · {agent} Agent  "
            f"| Session {session} · {call['timestamp']}",
            expanded=(i == 1),
        ):
            # ── Top metrics row ───────────────────────────────────────────────
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("⏱ Latency",       f"{call['latency_s']} s")
            col_b.metric("📥 Input Tokens",  call["input_tokens"])
            col_c.metric("📤 Output Tokens", call["output_tokens"])
            col_d.metric("🔢 Total Tokens",  call["total_tokens"])

            col_e, col_f, col_g, col_h = st.columns(4)
            col_e.metric("🌡 Temperature",   call["temperature"])
            col_f.metric("📏 Max Tokens",    call["max_tokens"])
            col_g.metric("🛑 Stop Reason",   call["stop_reason"])
            col_h.metric("📝 Word Count",    scores["word_count"])

            st.markdown("---")

            # ── Quality scores ────────────────────────────────────────────────
            st.markdown("**Response Quality Scores**")
            sq1, sq2, sq3, sq4 = st.columns(4)
            sq1.metric("✅ Overall",      f"{scores['overall']}/100")
            sq2.metric("📑 Completeness", f"{scores['completeness']}/100")
            sq3.metric("🔗 Coherence",    f"{scores['coherence']}/100")
            sq4.metric("📖 Verbosity",    f"{scores['verbosity']}/100")

            # ── Visual bars ───────────────────────────────────────────────────
            bar_labels = ["Overall", "Completeness", "Coherence", "Verbosity"]
            bar_values = [
                scores["overall"], scores["completeness"],
                scores["coherence"], scores["verbosity"],
            ]
            for label, val in zip(bar_labels, bar_values):
                col_lbl, col_bar, col_pct = st.columns([2, 6, 1])
                col_lbl.caption(label)
                bar_html = (
                    f"<div style='background:#e5e7eb;border-radius:4px;height:16px;margin-top:6px;'>"
                    f"<div style='width:{val}%;background:{_colour_for(val)};"
                    f"height:16px;border-radius:4px;'></div></div>"
                )
                col_bar.markdown(bar_html, unsafe_allow_html=True)
                col_pct.caption(f"**{val}**")

            # ── Token efficiency ──────────────────────────────────────────────
            if call["total_tokens"] > 0:
                st.markdown("---")
                eff_col1, eff_col2 = st.columns(2)
                words_per_token = round(scores["word_count"] / max(call["output_tokens"], 1), 2)
                pct_output = round(call["output_tokens"] / call["total_tokens"] * 100, 1)
                eff_col1.metric("Words / Output Token",  words_per_token)
                eff_col2.metric("Output % of Total Tokens", f"{pct_output} %")

    st.divider()

    # ── Trend charts (sparklines via st.line_chart) ───────────────────────────
    if total_calls >= 2:
        st.subheader("📉 Trend Charts (This Session)")

        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("**Latency per Call (s)**")
            latency_data = {
                "Latency (s)": [c["latency_s"] for c in calls],
                "Call #":      list(range(1, total_calls + 1)),
            }
            st.line_chart(
                data={c["agent"]: [] for c in calls},
            )
            # Simple approach: build a labelled dict for line_chart
            agent_latency: dict = {}
            for c in calls:
                agent_latency.setdefault(c["agent"], []).append(c["latency_s"])
            st.line_chart(agent_latency)

        with t_col2:
            st.markdown("**Overall Quality Score per Call**")
            agent_quality: dict = {}
            for c in calls:
                agent_quality.setdefault(c["agent"], []).append(
                    c["scores"]["overall"]
                )
            st.line_chart(agent_quality)

    # ── Token usage summary table ─────────────────────────────────────────────
    st.subheader("🔢 Token Usage Summary")
    table_data = {
        "Call #":          [],
        "Agent":           [],
        "Timestamp":       [],
        "Input Tokens":    [],
        "Output Tokens":   [],
        "Total Tokens":    [],
        "Latency (s)":     [],
        "Quality Score":   [],
        "Stop Reason":     [],
    }
    for i, c in enumerate(calls, 1):
        table_data["Call #"].append(i)
        table_data["Agent"].append(c["agent"])
        table_data["Timestamp"].append(c["timestamp"])
        table_data["Input Tokens"].append(c["input_tokens"])
        table_data["Output Tokens"].append(c["output_tokens"])
        table_data["Total Tokens"].append(c["total_tokens"])
        table_data["Latency (s)"].append(c["latency_s"])
        table_data["Quality Score"].append(c["scores"]["overall"])
        table_data["Stop Reason"].append(c["stop_reason"])

    st.dataframe(table_data, use_container_width=True)

    # ── Export eval log ───────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Export Evaluation Log (.json)",
        data=json.dumps(calls, indent=2),
        file_name=f"AgriDoctor_EvalLog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Clear session data ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Evaluation Data", use_container_width=True):
        st.session_state["eval_calls"] = []
        st.session_state["session_counter"] = 0
        st.rerun()


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AgriDoctor – Crop Disease AI",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialise evaluation state
    if "eval_calls" not in st.session_state:
        st.session_state["eval_calls"] = []
    if "session_counter" not in st.session_state:
        st.session_state["session_counter"] = 0

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🌿 AgriDoctor")
        st.caption("AI-Powered Crop Disease Diagnosis")
        st.divider()

        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "IBM Cloud API Key",
            type="password",
            placeholder="Enter your IBM Cloud API key",
            help="Your IBM Cloud API key is used to authenticate with watsonx.ai. "
                 "It is never stored or logged.",
        )

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **AgriDoctor** uses a two-agent AI workflow:

            **🔬 Agent 1 – Pathologist**
            Analyses crop/leaf images to detect diseases and symptoms.

            **📋 Agent 2 – Extension Officer**
            Converts the diagnosis into practical farmer-friendly advice.

            **Powered by:**
            IBM Granite 4 H Small on IBM watsonx.ai
            """
        )
        st.divider()

        # Live call counter in sidebar
        n_calls = len(st.session_state.get("eval_calls", []))
        if n_calls:
            st.metric("📊 API Calls This Session", n_calls)
            avg_lat = round(
                sum(c["latency_s"] for c in st.session_state["eval_calls"]) / n_calls, 2
            )
            st.metric("⏱ Avg Latency", f"{avg_lat} s")
        st.divider()
        st.caption(
            "⚠️ AI-generated preliminary assessment only. "
            "Consult a certified expert for confirmation."
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_diag, tab_eval = st.tabs(["🌾 Diagnosis", "📊 LLM Evaluation Dashboard"])

    # =========================================================================
    # TAB 1 – DIAGNOSIS
    # =========================================================================
    with tab_diag:
        st.title("🌾 AgriDoctor – Crop Disease Diagnosis & Advisory")
        st.markdown(
            "Upload a photo of your crop or plant leaf. AgriDoctor's two AI agents "
            "will diagnose the disease and provide a practical management advisory."
        )

        st.info(
            "📌 **How it works:** "
            "① Upload image → ② Pathologist Agent diagnoses the disease → "
            "③ Extension Officer Agent generates farmer-friendly advice → "
            "④ Download your advisory report.",
            icon="ℹ️",
        )

        st.divider()

        # ── Image Upload ──────────────────────────────────────────────────────
        col_upload, col_preview = st.columns([1, 1])

        with col_upload:
            st.subheader("📷 Upload Crop Image")
            uploaded_file = st.file_uploader(
                "Choose a JPG, JPEG, or PNG image",
                type=["jpg", "jpeg", "png"],
                help="For best results, upload a clear, well-lit close-up of the "
                     "affected plant part (leaf, stem, fruit, etc.).",
            )
            st.subheader("📝 Describe What You See (Optional but Recommended)")
            symptom_description = st.text_area(
                "Describe the visible symptoms on your crop",
                placeholder=(
                    "e.g. 'Leaves are turning yellow with brown spots, edges are "
                    "drying out, some leaves have white powdery coating, plant "
                    "looks wilted in the morning...'"
                ),
                height=130,
                help="Describe any discolouration, spots, lesions, wilting, mould, "
                     "or unusual appearance you can see. The more detail you provide, "
                     "the more accurate the diagnosis will be.",
            )

        with col_preview:
            if uploaded_file is not None:
                st.subheader("🖼️ Uploaded Image Preview")
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
                uploaded_file.seek(0)  # Reset file pointer after PIL reads it

        st.divider()

        # ── Diagnose Button ───────────────────────────────────────────────────
        diagnose_btn = st.button(
            "🔍 Diagnose Crop",
            type="primary",
            disabled=(uploaded_file is None),
            use_container_width=True,
        )

        if uploaded_file is None:
            st.warning("Please upload a crop or plant-leaf image to get started.")

        # ── Run Agents ────────────────────────────────────────────────────────
        if diagnose_btn:
            if not api_key:
                st.error(
                    "🔐 API key is required. Please enter your IBM Cloud API key "
                    "in the sidebar."
                )
                st.stop()

            # Increment session counter
            st.session_state["session_counter"] += 1
            session_id = st.session_state["session_counter"]

            # Extract image metadata using PIL
            try:
                image_context = extract_image_context(uploaded_file)
            except Exception as exc:
                st.error(f"Failed to process the uploaded image: {exc}")
                st.stop()

            # ── Agent 1: Pathologist ──────────────────────────────────────────
            st.subheader("🔬 Agent 1 – Pathologist Agent")
            st.caption("Analysing your crop image for disease symptoms…")

            path_result = None
            with st.spinner("🧬 Pathologist Agent is examining the image…"):
                try:
                    path_result = run_pathologist_agent(
                        api_key, image_context, symptom_description
                    )
                except RuntimeError as exc:
                    error_msg = str(exc)
                    if "401" in error_msg or "403" in error_msg:
                        st.error(
                            "❌ Authentication failed. Please check your IBM Cloud "
                            "API key and try again."
                        )
                    elif "model" in error_msg.lower():
                        st.error(
                            "❌ The selected model is not available or not supported "
                            f"for this request.\n\nDetails: {error_msg}"
                        )
                    else:
                        st.error(f"❌ Pathologist Agent error: {error_msg}")
                    st.stop()
                except Exception as exc:
                    st.error(f"❌ Unexpected error in Pathologist Agent: {exc}")
                    st.stop()

            diagnosis_result = path_result["text"]

            # Store eval metrics for Pathologist call
            path_scores = _score_response(diagnosis_result, "Pathologist")
            path_result["scores"] = path_scores
            path_result["session_id"] = session_id
            st.session_state["eval_calls"].append(path_result)

            # ── Inline latency badge ──────────────────────────────────────────
            st.success(
                f"✅ Pathologist Agent completed in **{path_result['latency_s']} s** · "
                f"{path_result['total_tokens']} tokens used · "
                f"Quality score: **{path_scores['overall']}/100**"
            )

            with st.expander("📄 Pathologist Agent Report", expanded=True):
                st.markdown(diagnosis_result)

            # ── Agent 2: Extension Officer ────────────────────────────────────
            st.subheader("📋 Agent 2 – Extension Officer Agent")
            st.caption(
                "Converting the diagnosis into a farmer-friendly advisory…"
            )

            ext_result = None
            with st.spinner("📝 Extension Officer Agent is preparing your advisory…"):
                try:
                    ext_result = run_extension_officer_agent(
                        api_key, diagnosis_result
                    )
                except RuntimeError as exc:
                    error_msg = str(exc)
                    if "401" in error_msg or "403" in error_msg:
                        st.error(
                            "❌ Authentication failed. Please check your IBM Cloud "
                            "API key and try again."
                        )
                    else:
                        st.error(f"❌ Extension Officer Agent error: {error_msg}")
                    st.stop()
                except Exception as exc:
                    st.error(f"❌ Unexpected error in Extension Officer Agent: {exc}")
                    st.stop()

            advisory_result = ext_result["text"]

            # Store eval metrics for Extension Officer call
            ext_scores = _score_response(advisory_result, "Extension Officer")
            ext_result["scores"] = ext_scores
            ext_result["session_id"] = session_id
            st.session_state["eval_calls"].append(ext_result)

            st.success(
                f"✅ Extension Officer Agent completed in **{ext_result['latency_s']} s** · "
                f"{ext_result['total_tokens']} tokens used · "
                f"Quality score: **{ext_scores['overall']}/100**"
            )

            with st.expander("📄 Extension Officer Advisory", expanded=True):
                st.markdown(advisory_result)

            # ── Combined Final Report ─────────────────────────────────────────
            st.divider()
            st.subheader("📑 Combined Final Report")

            final_report = build_final_report(
                diagnosis_result,
                advisory_result,
                uploaded_file.name,
            )

            st.success("✅ Multi-agent analysis complete!")

            with st.expander("📋 View Full Combined Report", expanded=False):
                st.text(final_report)

            # ── Download Button ───────────────────────────────────────────────
            st.download_button(
                label="⬇️ Download Advisory Report (.txt)",
                data=final_report.encode("utf-8"),
                file_name=f"AgriDoctor_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            # ── Footer Warning ────────────────────────────────────────────────
            st.divider()
            st.warning(
                "⚠️ **Important Disclaimer:** This report is an AI-generated "
                "preliminary assessment and is NOT a confirmed laboratory diagnosis. "
                "Always consult a certified plant pathologist or local agricultural "
                "extension officer before taking action, especially regarding "
                "chemical treatments. Follow all pesticide label instructions and "
                "local agricultural regulations. Use appropriate Personal Protective "
                "Equipment (PPE) when handling agrochemicals.",
                icon="⚠️",
            )

            st.info(
                "📊 Switch to the **LLM Evaluation Dashboard** tab to view "
                "detailed performance metrics for this diagnosis session.",
                icon="📊",
            )

    # =========================================================================
    # TAB 2 – LLM EVALUATION DASHBOARD
    # =========================================================================
    with tab_eval:
        render_eval_dashboard()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
