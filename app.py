"""
NeuroScan — Brain Tumor Detection.
Hybrid quantum-classical model: ResNet18 + 4-qubit variational quantum circuit.
"""

import torch
import streamlit as st
from PIL import Image

from src.data.data_loader import get_transforms
from src.models.hybrid_model import HybridModel

st.set_page_config(
    page_title="NeuroScan · Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = ["no_tumor", "tumor"]
device = torch.device("cpu")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f8fafc; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* NAV */
.ns-nav {
    background: #fff;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 2.5rem;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.ns-logo { font-size: 1.05rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }
.ns-logo span { color: #4f46e5; }
.ns-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    color: #4f46e5; background: #eef2ff;
    border: 1px solid #c7d2fe;
    padding: 0.28rem 0.75rem;
    border-radius: 6px;
    letter-spacing: 0.05em;
}

/* HERO BANNER */
.ns-hero {
    background: linear-gradient(135deg, #4f46e5 0%, #6d28d9 100%);
    padding: 3.5rem 2.5rem;
    text-align: center;
}
.ns-hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600;
    color: #a5b4fc; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 1rem;
}
.ns-hero h1 {
    font-size: 2.6rem; font-weight: 900;
    color: #fff; letter-spacing: -0.04em;
    line-height: 1.1; margin: 0 0 0.8rem 0;
}
.ns-hero p {
    font-size: 0.95rem; color: #c7d2fe;
    line-height: 1.7; max-width: 500px;
    margin: 0 auto 2rem auto;
}
.ns-hero-stats {
    display: inline-flex; gap: 0;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px; overflow: hidden;
}
.ns-hstat {
    padding: 0.9rem 2rem;
    border-right: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}
.ns-hstat:last-child { border-right: none; }
.ns-hstat-n {
    font-size: 1.6rem; font-weight: 900;
    color: #fff; letter-spacing: -0.03em; line-height: 1;
}
.ns-hstat-l { font-size: 0.68rem; color: #a5b4fc; margin-top: 3px; font-weight: 500; }

/* MAIN CONTENT */
.ns-body {
    max-width: 740px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem 1rem 1.5rem;
}

/* STEPS */
.ns-steps {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.ns-step {
    padding: 1.2rem 1rem;
    border-right: 1px solid #f1f5f9;
}
.ns-step:last-child { border-right: none; }
.ns-step-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; font-weight: 700;
    color: #4f46e5; margin-bottom: 0.4rem;
}
.ns-step-t { font-size: 0.8rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
.ns-step-d { font-size: 0.72rem; color: #94a3b8; line-height: 1.5; }

/* UPLOAD HEAD */
.ns-upload-head {
    background: linear-gradient(135deg, #4f46e5, #6d28d9);
    padding: 1rem 1.4rem;
    border-radius: 14px 14px 0 0;
    display: flex; align-items: center;
    justify-content: space-between;
}
.ns-upload-head-title { font-size: 0.9rem; font-weight: 700; color: #fff; }
.ns-upload-head-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; color: #c4b5fd; letter-spacing: 0.04em;
}

[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 1.4rem !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6d28d9) !important;
    color: #fff !important; border: none !important;
    padding: 0.7rem !important; border-radius: 10px !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    width: 100% !important; letter-spacing: -0.01em !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.3) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(79,70,229,0.38) !important;
}

/* IMAGE */
.stImage img { border-radius: 10px !important; }

/* SCAN READY PANEL */
.ns-ready {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.65;
}
.ns-ready-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 700;
    color: #4f46e5; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.35rem;
}

/* RESULTS */
.ns-result-tumor {
    background: #fff1f2;
    border: 1px solid #fca5a5;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.2rem;
}
.ns-result-healthy {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin-top: 1.2rem;
}
.ns-result-title { font-size: 1rem; font-weight: 800; color: #0f172a; margin: 0 0 0.2rem 0; }
.ns-conf-t {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; font-weight: 600; color: #dc2626;
}
.ns-conf-h {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem; font-weight: 600; color: #16a34a;
}
.ns-result-note { font-size: 0.74rem; color: #64748b; margin-top: 0.3rem; }

/* PROB */
.ns-prob-row {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 0.25rem;
}
.ns-prob-name { font-size: 0.82rem; font-weight: 600; color: #334155; }
.ns-prob-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; color: #64748b;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border-radius: 4px !important;
}
.stProgress > div > div {
    background: #e2e8f0 !important;
    border-radius: 4px !important;
    height: 6px !important;
}

/* FOOTER */
.ns-footer {
    max-width: 740px; margin: 1.5rem auto 0 auto;
    padding: 1.2rem 1.5rem 2rem 1.5rem;
    border-top: 1px solid #e2e8f0;
    display: flex; align-items: center; justify-content: space-between;
}
.ns-footer-l { font-size: 0.78rem; font-weight: 700; color: #1e293b; }
.ns-footer-l span { color: #4f46e5; }
.ns-footer-r { font-size: 0.68rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = HybridModel(n_classes=2)
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
transform = get_transforms()


# ── NAV
st.markdown("""
<div class="ns-nav">
    <div class="ns-logo">Neuro<span>Scan</span></div>
    <div class="ns-badge">⚛ ResNet18 + 4-qubit VQC</div>
</div>
""", unsafe_allow_html=True)


# ── HERO
st.markdown("""
<div class="ns-hero">
    <div class="ns-hero-tag">Hybrid Quantum · Classical · Medical Imaging</div>
    <h1>Brain Tumor Detection</h1>
    <p>Upload a brain MRI scan. A ResNet18 CNN extracts features;
       a variational quantum circuit classifies them. Result in under 2 seconds.</p>
    <div class="ns-hero-stats">
        <div class="ns-hstat">
            <div class="ns-hstat-n">88%</div>
            <div class="ns-hstat-l">Accuracy</div>
        </div>
        <div class="ns-hstat">
            <div class="ns-hstat-n">4</div>
            <div class="ns-hstat-l">Qubits</div>
        </div>
        <div class="ns-hstat">
            <div class="ns-hstat-n">&lt;2s</div>
            <div class="ns-hstat-l">Inference</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── BODY
st.markdown('<div class="ns-body">', unsafe_allow_html=True)

# ── STEPS
st.markdown("""
<div class="ns-steps">
    <div class="ns-step">
        <div class="ns-step-n">01</div>
        <div class="ns-step-t">Upload MRI</div>
        <div class="ns-step-d">Brain scan — JPG or PNG.</div>
    </div>
    <div class="ns-step">
        <div class="ns-step-n">02</div>
        <div class="ns-step-t">CNN Encode</div>
        <div class="ns-step-d">ResNet18 → 512-dim features.</div>
    </div>
    <div class="ns-step">
        <div class="ns-step-n">03</div>
        <div class="ns-step-t">Quantum Layer</div>
        <div class="ns-step-d">RY + CNOT + ⟨Z⟩ readout.</div>
    </div>
    <div class="ns-step">
        <div class="ns-step-n">04</div>
        <div class="ns-step-t">Result</div>
        <div class="ns-step-d">Softmax → class probability.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD HEAD (separate, self-closing — no leaking)
st.markdown("""
<div class="ns-upload-head">
    <div class="ns-upload-head-title">Quantum MRI Analyser</div>
    <div class="ns-upload-head-sub">ResNet18 + 4-qubit VQC · CPU · &lt;2s</div>
</div>
""", unsafe_allow_html=True)

# File uploader — NOT wrapped in HTML so it renders cleanly
uploaded_file = st.file_uploader(
    "Drop a brain MRI image here — JPG or PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1.1, 1])

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded scan", use_container_width=True)

    with col2:
        st.markdown("""
        <div class="ns-ready">
            <div class="ns-ready-label">Scan loaded</div>
            ResNet18 encodes this scan into a 512-dim feature vector.
            The 4-qubit variational circuit then classifies it and returns probabilities.
        </div>
        """, unsafe_allow_html=True)
        analyze = st.button("Run quantum analysis")
        if st.button("Clear"):
            st.rerun()

    if analyze:
        with st.spinner("Running quantum circuit…"):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = int(probabilities.argmax())
                confidence = float(probabilities[predicted_idx])

        prediction = CLASS_NAMES[predicted_idx]
        tumor_prob = float(probabilities[1])
        healthy_prob = float(probabilities[0])

        if prediction == "tumor":
            st.markdown(f"""
            <div class="ns-result-tumor">
                <div class="ns-result-title">Tumor signal detected</div>
                <div class="ns-conf-t">Confidence · {confidence:.1%}</div>
                <div class="ns-result-note">Review with a neuroradiologist before any clinical action.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ns-result-healthy">
                <div class="ns-result-title">No tumor signal found</div>
                <div class="ns-conf-h">Confidence · {confidence:.1%}</div>
                <div class="ns-result-note">No evidence of a mass detected.</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        st.markdown(f"""
        <div class="ns-prob-row" style="margin-top:1rem;">
            <span class="ns-prob-name">Tumor</span>
            <span class="ns-prob-val">{tumor_prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(tumor_prob)

        st.markdown(f"""
        <div class="ns-prob-row" style="margin-top:0.6rem;">
            <span class="ns-prob-name">No tumor</span>
            <span class="ns-prob-val">{healthy_prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(healthy_prob)

else:
    st.markdown("""
    <div style="text-align:center;padding:2.2rem 1rem;
                border:2px dashed #e2e8f0;border-radius:10px;
                background:#fff;margin-top:0.8rem;">
        <div style="font-size:0.85rem;color:#94a3b8;">
            Upload a brain MRI above to begin analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # ns-body close


# ── FOOTER
st.markdown("""
<div class="ns-footer">
    <div class="ns-footer-l">Neuro<span>Scan</span> · Research prototype</div>
    <div class="ns-footer-r">PennyLane · PyTorch · Streamlit</div>
</div>
""", unsafe_allow_html=True)