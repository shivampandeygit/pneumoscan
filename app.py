"""
Pneumonia Detection Portal — Streamlit Frontend
Sends chest X-ray images to the FastAPI backend and visualizes results.
"""

import streamlit as st
import requests
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import base64
import io
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BACKEND_URL = "https://pneumoscan-api-zxkp.onrender.com"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Crimson+Pro:ital,wght@0,300;0,400;1,300&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0, 180, 219, 0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0, 255, 180, 0.04) 0%, transparent 50%);
}

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero Header ── */
.hero-wrapper {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(0, 180, 219, 0.15);
    margin-bottom: 2rem;
}
.hero-left {}
.hero-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00b4db;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    color: #f0f6ff;
    letter-spacing: -0.02em;
    margin: 0;
}
.hero-title span {
    color: #00b4db;
}
.hero-subtitle {
    font-family: 'Crimson Pro', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: rgba(180, 200, 230, 0.6);
    margin-top: 0.5rem;
    font-weight: 300;
}
.hero-model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(0, 180, 219, 0.06);
    border: 1px solid rgba(0, 180, 219, 0.2);
    border-radius: 2rem;
    padding: 0.5rem 1.1rem;
    font-size: 0.72rem;
    color: rgba(160, 210, 240, 0.8);
    letter-spacing: 0.05em;
    margin-top: 1rem;
}
.dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00b4db;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

/* ── Status Bar ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 1.2rem;
    border-radius: 0.5rem;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.status-ok {
    background: rgba(0, 255, 150, 0.06);
    border: 1px solid rgba(0, 255, 150, 0.2);
    color: #00ff96;
}
.status-warn {
    background: rgba(255, 200, 0, 0.06);
    border: 1px solid rgba(255, 200, 0, 0.2);
    color: #ffc800;
}
.status-err {
    background: rgba(255, 60, 60, 0.06);
    border: 1px solid rgba(255, 60, 60, 0.2);
    color: #ff4444;
}

/* ── Upload Zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(160, 200, 240, 0.7);
    margin-bottom: 0.75rem;
}

[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(0, 180, 219, 0.25) !important;
    border-radius: 0.75rem !important;
    background: rgba(0, 180, 219, 0.02) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 180, 219, 0.5) !important;
    background: rgba(0, 180, 219, 0.05) !important;
}

/* ── Cards ── */
.card {
    background: rgba(10, 20, 40, 0.6);
    border: 1px solid rgba(0, 180, 219, 0.1);
    border-radius: 0.75rem;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(0, 180, 219, 0.7);
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(0, 180, 219, 0.1);
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
}
.info-key { color: rgba(180, 200, 230, 0.5); }
.info-val { color: #e0f0ff; font-weight: 500; }

/* ── Analyze Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00b4db 0%, #00897b 100%) !important;
    color: #050810 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    border: none !important;
    border-radius: 0.5rem !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(0, 180, 219, 0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 35px rgba(0, 180, 219, 0.4) !important;
}

/* ── Result Verdict Cards ── */
.verdict-high {
    background: rgba(255, 50, 50, 0.06);
    border: 1.5px solid rgba(255, 50, 50, 0.35);
    border-radius: 0.75rem;
    padding: 1.5rem 1.8rem;
}
.verdict-medium {
    background: rgba(255, 180, 0, 0.06);
    border: 1.5px solid rgba(255, 180, 0, 0.35);
    border-radius: 0.75rem;
    padding: 1.5rem 1.8rem;
}
.verdict-low {
    background: rgba(0, 220, 130, 0.06);
    border: 1.5px solid rgba(0, 220, 130, 0.35);
    border-radius: 0.75rem;
    padding: 1.5rem 1.8rem;
}
.verdict-risk-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.verdict-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.3;
}
.verdict-rec {
    font-family: 'Crimson Pro', serif;
    font-size: 1rem;
    font-weight: 300;
    font-style: italic;
    margin-top: 0.6rem;
    opacity: 0.8;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.25rem 0;
}
.metric-card {
    background: rgba(10, 20, 40, 0.8);
    border: 1px solid rgba(0, 180, 219, 0.12);
    border-radius: 0.65rem;
    padding: 1.1rem;
    text-align: center;
}
.metric-label {
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(160, 200, 240, 0.5);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f0f6ff;
    line-height: 1;
}
.metric-value.accent { color: #00b4db; }
.metric-value.danger { color: #ff5555; }
.metric-value.success { color: #00dc82; }
.metric-value.warning { color: #ffc800; }

/* ── Inference time tag ── */
.infer-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0, 180, 219, 0.06);
    border: 1px solid rgba(0, 180, 219, 0.15);
    border-radius: 2rem;
    padding: 0.35rem 0.9rem;
    font-size: 0.68rem;
    color: rgba(160, 210, 240, 0.7);
    letter-spacing: 0.08em;
}

/* ── Section Label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(0, 180, 219, 0.6);
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0, 180, 219, 0.12);
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255, 200, 0, 0.04);
    border: 1px solid rgba(255, 200, 0, 0.15);
    border-radius: 0.5rem;
    padding: 0.9rem 1.2rem;
    font-size: 0.75rem;
    color: rgba(255, 200, 0, 0.7);
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* ── Divider ── */
hr { border-color: rgba(0, 180, 219, 0.08) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00b4db !important; }

/* ── Column gaps ── */
[data-testid="column"] { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def img_to_b64(pil_img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def check_backend() -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=4)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-left">
    <div class="hero-tag">AI Medical Imaging · v2.0</div>
    <div class="hero-title">Pneumo<span>Scan</span></div>
    <div class="hero-subtitle">Vision Transformer · Chest X-Ray Analysis</div>
    <div class="hero-model-badge">
      <span class="dot"></span>
      nickmuchi/vit-finetuned-chest-xray-pneumonia
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Backend Status ────────────────────────────────────────────────────────────
health = check_backend()

if health is None:
    st.markdown("""
    <div class="status-bar status-err">
      ⬤ &nbsp; Backend offline — run: <code>uvicorn main:app --reload</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
elif not health.get("model_loaded"):
    st.markdown("""
    <div class="status-bar status-warn">
      ◌ &nbsp; Model loading... Please wait a moment.
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    st.rerun()
else:
    device = health.get("device", "cpu").upper()
    load_ms = health.get("model_load_time_ms", "—")
    st.markdown(f"""
    <div class="status-bar status-ok">
      ⬤ &nbsp; Model ready &nbsp;·&nbsp; Device: {device} &nbsp;·&nbsp; Loaded in {load_ms} ms
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 2], gap="large")

with col_up:
    st.markdown('<div class="upload-label">Upload Chest X-Ray</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("""
    <div class="card">
      <div class="card-title">How to Use</div>
      <div class="info-row"><span class="info-key">Step 1</span><span class="info-val">Upload a chest X-ray image</span></div>
      <div class="info-row"><span class="info-key">Step 2</span><span class="info-val">Click Analyze</span></div>
      <div class="info-row"><span class="info-key">Step 3</span><span class="info-val">Review AI prediction</span></div>
      <div class="info-row"><span class="info-key">Formats</span><span class="info-val">JPG · PNG · WebP</span></div>
      <div class="info-row"><span class="info-key">Max Size</span><span class="info-val">10 MB</span></div>
      <div class="info-row"><span class="info-key">Model</span><span class="info-val">ViT (Vision Transformer)</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── If Image Uploaded ─────────────────────────────────────────────────────────
if uploaded:
    img = Image.open(uploaded)
    file_kb = len(uploaded.getvalue()) // 1024

    st.markdown("<hr>", unsafe_allow_html=True)

    col_img, col_meta = st.columns([3, 2], gap="large")

    with col_img:
        st.markdown('<div class="section-label">X-Ray Preview</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    with col_meta:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Image Details</div>
          <div class="info-row"><span class="info-key">Filename</span><span class="info-val">{uploaded.name}</span></div>
          <div class="info-row"><span class="info-key">Dimensions</span><span class="info-val">{img.size[0]} × {img.size[1]} px</span></div>
          <div class="info-row"><span class="info-key">Mode</span><span class="info-val">{img.mode}</span></div>
          <div class="info-row"><span class="info-key">Format</span><span class="info-val">{uploaded.type.split("/")[-1].upper()}</span></div>
          <div class="info-row"><span class="info-key">File Size</span><span class="info-val">{file_kb} KB</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔬  Analyze X-Ray", use_container_width=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze:
        with st.spinner("Running ViT inference..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/predict",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
                result = resp.json()
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Backend error: {result.get('detail', 'Unknown error')}")
            st.stop()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

        # ── Verdict Card ──────────────────────────────────────────────────────
        risk = result["risk_level"]
        if risk == "HIGH":
            verdict_class = "verdict-high"
            risk_color = "#ff5555"
            icon = "🔴"
        elif risk == "MEDIUM":
            verdict_class = "verdict-medium"
            risk_color = "#ffc800"
            icon = "🟡"
        else:
            verdict_class = "verdict-low"
            risk_color = "#00dc82"
            icon = "🟢"

        st.markdown(f"""
        <div class="{verdict_class}">
          <div class="verdict-risk-label" style="color:{risk_color};">{icon} RISK: {risk}</div>
          <div class="verdict-title" style="color:{risk_color};">{result['verdict']}</div>
          <div class="verdict-rec">{result['recommendation']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        conf_pct = result['confidence'] * 100
        label = result['label']
        infer_ms = result['inference_time_ms']

        if risk == "HIGH":
            conf_cls = "danger"
        elif risk == "MEDIUM":
            conf_cls = "warning"
        else:
            conf_cls = "success"

        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value accent" style="font-size:1.1rem;">{label}</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value {conf_cls}">{conf_pct:.1f}%</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value {'danger' if risk=='HIGH' else 'warning' if risk=='MEDIUM' else 'success'}">{risk}</div>
          </div>
        </div>
        <div style="text-align:right; margin-top:-0.5rem; margin-bottom:1rem;">
          <span class="infer-tag">⏱ Inference: {infer_ms} ms</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence Chart ──────────────────────────────────────────────────
        st.markdown('<div class="section-label">Confidence Breakdown</div>', unsafe_allow_html=True)

        all_scores = result["all_scores"]
        labels_chart = [s["label"] for s in all_scores]
        scores_chart = [round(s["score"] * 100, 2) for s in all_scores]
        bar_colors = ["#ff5555" if "PNEUMONIA" in l.upper() else "#00dc82" for l in labels_chart]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels_chart,
            y=scores_chart,
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=[f"{s:.2f}%" for s in scores_chart],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=13, color="#e0f0ff"),
            width=0.4,
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="rgba(180,200,240,0.7)", size=11),
            yaxis=dict(
                range=[0, 115],
                title="Confidence (%)",
                gridcolor="rgba(0,180,219,0.07)",
                zeroline=False,
                tickfont=dict(size=10),
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(family="Syne, sans-serif", size=13, color="#e0f0ff"),
            ),
            showlegend=False,
            bargap=0.5,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── Confidence Gauge ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">Prediction Confidence Gauge</div>', unsafe_allow_html=True)

        gauge_color = "#ff5555" if risk == "HIGH" else "#ffc800" if risk == "MEDIUM" else "#00dc82"
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_pct,
            number=dict(suffix="%", font=dict(family="Syne, sans-serif", size=36, color="#f0f6ff")),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(160,200,240,0.3)",
                          tickfont=dict(family="DM Mono", size=10, color="rgba(160,200,240,0.5)")),
                bar=dict(color=gauge_color, thickness=0.22),
                bgcolor="rgba(10,20,40,0.8)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="rgba(0,180,219,0.04)"),
                    dict(range=[50, 85], color="rgba(0,180,219,0.08)"),
                    dict(range=[85, 100], color="rgba(0,180,219,0.04)"),
                ],
                threshold=dict(
                    line=dict(color=gauge_color, width=2),
                    thickness=0.75,
                    value=conf_pct,
                ),
            ),
        ))
        fig2.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono, monospace", color="rgba(180,200,240,0.7)"),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
          ⚠️ <strong>Medical Disclaimer:</strong> This tool is strictly for educational and research purposes only.
          It is <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment.
          Always consult a qualified physician or radiologist for clinical decisions.
        </div>
        """, unsafe_allow_html=True)