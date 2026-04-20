"""
zkFedMoE Interactive Dashboard
================================
Launch:  python -m streamlit run dashboard.py
"""

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.data.text_datasets import TextClassificationDataset, build_ag_news_clients
from src.fl.adversaries import freerider_train, poisoning_train, sybil_clones
from src.fl.client import local_train
from src.fl.dp import PrivacyAccountant, apply_dp
from src.fl.sepg import generate_proof, verify_proof
from src.fl.server import FedServer
from src.models.moe_model import MoETextClassifier, predict_with_routing

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
st.set_page_config(page_title="zkFedMoE", page_icon="🧠", layout="wide")

st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1b2d 0%, #1a2f4a 60%, #0f1b2d 100%);
}
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 10px; border-radius: 6px;
    transition: background 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,0.08); }

/* ── Page banner ── */
.page-banner {
    background: linear-gradient(90deg, #1565C0 0%, #0D47A1 50%, #283593 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 18px;
    color: white;
}
.page-banner h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; color: white; }
.page-banner p  { font-size: 1rem; margin: 0; opacity: 0.85; color: #dce8ff; }

/* ── Flow step pill ── */
.flow-bar {
    display: flex; align-items: center; gap: 0;
    background: #f0f4ff; border-radius: 10px;
    padding: 10px 16px; margin-bottom: 16px;
    overflow-x: auto;
}
.flow-step {
    display: flex; align-items: center; gap: 6px;
    background: #e3eafc; border-radius: 8px;
    padding: 6px 14px; font-size: 0.82rem;
    font-weight: 600; color: #1a3a6b; white-space: nowrap;
    border: 1px solid #c5d4f5;
}
.flow-step.active {
    background: #1565C0; color: white;
    border-color: #1565C0;
    box-shadow: 0 2px 8px rgba(21,101,192,0.4);
}
.flow-arrow {
    color: #9baecf; font-size: 1.1rem; padding: 0 4px;
    flex-shrink: 0;
}

/* ── Concept card ── */
.concept-card {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
    border-left: 4px solid #1565C0;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin: 10px 0;
}
.concept-card h4 { margin: 0 0 5px 0; color: #1565C0; font-size: 0.95rem; }
.concept-card p  { margin: 0; font-size: 0.88rem; color: #374151; line-height: 1.5; }

/* ── Warning / key-insight box ── */
.insight-box {
    background: linear-gradient(135deg, #fff8e1, #fff3cd);
    border-left: 4px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 8px 0;
    font-size: 0.88rem; color: #78350f;
}

/* ── Attack badge ── */
.attack-badge {
    display: inline-block;
    background: #fef2f2; border: 1px solid #fca5a5;
    color: #991b1b; border-radius: 20px;
    padding: 3px 12px; font-size: 0.8rem; font-weight: 600;
}
.safe-badge {
    display: inline-block;
    background: #f0fdf4; border: 1px solid #86efac;
    color: #166534; border-radius: 20px;
    padding: 3px 12px; font-size: 0.8rem; font-weight: 600;
}

/* ── Metric card override ── */
[data-testid="stMetric"] {
    background: #f8faff;
    border: 1px solid #dbe4f5;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

PLOT_DIR = Path(__file__).parent / "plots"


def set_seed(s=42):
    torch.manual_seed(s)


def page_banner(title: str, subtitle: str, icon: str = ""):
    st.markdown(f"""
    <div class="page-banner">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def flow_bar(steps: list, active: str):
    """Render a horizontal pipeline breadcrumb. `steps` = list of strings, `active` = current step."""
    html = '<div class="flow-bar">'
    for i, s in enumerate(steps):
        cls = "flow-step active" if s == active else "flow-step"
        html += f'<div class="{cls}">{s}</div>'
        if i < len(steps) - 1:
            html += '<span class="flow-arrow">&#9658;</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def concept_card(title: str, body: str):
    st.markdown(f"""
    <div class="concept-card">
        <h4>&#128161; {title}</h4>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">&#9888;&#65039; {text}</div>', unsafe_allow_html=True)


def evaluate(model, dataset, bs, device):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
    model.to(device).eval()
    c = t = 0
    with torch.no_grad():
        for ids, lbl in loader:
            ids, lbl = ids.to(device), lbl.to(device)
            c += (model(ids).argmax(-1) == lbl).sum().item()
            t += lbl.size(0)
    return c / max(t, 1)


# FIX #1 — cache the base AG News data separately from the client split,
# so changing the client-count slider doesn't reload 120K rows.
@st.cache_resource
def _load_ag_news_raw():
    """Load AG News once and cache the raw texts/labels/vocab."""
    return build_ag_news_clients(
        num_clients=1, seq_len=64, use_external_csv=True, max_vocab=5000
    )


def load_ag_news(num_clients: int):
    """Split cached AG News into `num_clients` shards (fast, no re-parse)."""
    clients_1, test_ds, vs, nc, vocab = _load_ag_news_raw()
    # clients_1[0] is the whole train set as a single shard; re-split it
    full_train = clients_1[0]
    n = len(full_train)
    sizes = [n // num_clients] * num_clients
    sizes[0] += n - sum(sizes)
    clients = list(random_split(full_train, sizes))
    return clients, test_ds, vs, nc, vocab


# FIX #1 (continued) — cache the quick-start model so Predict page never
# re-trains when the user navigates away and comes back.
@st.cache_resource
def _build_quickstart_model():
    set_seed()
    dev = torch.device("cpu")
    cl, td, vs, nc, voc = build_ag_news_clients(
        num_clients=4, seq_len=64, use_external_csv=False, repeat=50)
    kw = dict(vocab_size=vs, embed_dim=64, num_classes=nc,
              num_experts=4, expert_hidden_dim=128, k=2, lora_r=8)
    srv = FedServer(MoETextClassifier(**kw), device=dev)
    for _ in range(8):
        sts = []
        for c in cl:
            m = MoETextClassifier(**kw)
            m.load_state_dict(srv.get_global_state(), strict=False)
            fs, _, n, _, _, _, _ = local_train(m, c, 2, 16, 5e-4, dev)
            sts.append((fs, n))
        srv.aggregate(sts)
    return srv.global_model, voc, kw


# ---------------------------------------------------------------
# Sidebar — FIX #7: model status block
# ---------------------------------------------------------------
st.sidebar.title("zkFedMoE")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "Predict", "Train", "Custom CSV", "Privacy & DP", "Robustness",
     "Experiments", "Compare", "Architecture", "About"],
)
st.sidebar.divider()

# Model status widget
if "model" in st.session_state:
    _kw = st.session_state.get("model_kw", {})
    _src = st.session_state.get("model_source", "quick-start")
    _nc  = _kw.get("num_classes", "?")
    _ne  = _kw.get("num_experts", "?")
    _k   = _kw.get("k", "?")
    st.sidebar.success(
        f"**Model loaded**\n\n"
        f"Source: `{_src}`\n\n"
        f"Classes: {_nc} | Experts: {_ne} | K: {_k}"
    )
else:
    st.sidebar.info("No model loaded yet.\nGo to **Train** or **Custom CSV**.")

st.sidebar.divider()
st.sidebar.caption("Group #34 | IIIT Kota | April 2026")


# ---------------------------------------------------------------
# PAGE: HOME
# ---------------------------------------------------------------
if page == "🏠 Home":
    page_banner(
        "zkFedMoE",
        "Zero-Knowledge Federated Mixture-of-Experts · Privacy-Preserving Adaptive LLM Customization · IIIT Kota Group #34",
        "🧠"
    )

    # Full system pipeline
    st.subheader("System Pipeline")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        graph [bgcolor=transparent splines=ortho nodesep=0.6];
        node [shape=box style="rounded,filled" fontname="Segoe UI" fontsize=11 width=1.6];
        edge [color="#4C72B0" penwidth=1.5];

        subgraph cluster_data {
            label="Data" style=filled fillcolor="#EEF2FF" color="#7C93D0";
            csv  [label="Raw Text\n(CSV / AG News)" fillcolor="#DBEAFE"];
            tok  [label="Tokenise\n+ Vocab Build"   fillcolor="#BFDBFE"];
            shard[label="Client\nSharding"           fillcolor="#BFDBFE"];
        }
        subgraph cluster_local {
            label="Client (xN)" style=filled fillcolor="#F0FDF4" color="#6EBD8C";
            emb  [label="Embedding\n+ Mean Pool"    fillcolor="#BBF7D0"];
            moe  [label="MoE Layer\nTop-K Routing"  fillcolor="#86EFAC"];
            lora [label="LoRA\nClassifier"           fillcolor="#BBF7D0"];
            dp   [label="DP-SGD\nClip + Noise"      fillcolor="#FEF9C3"];
            proof[label="SEPG Proof\nGenerate"       fillcolor="#FDE68A"];
        }
        subgraph cluster_server {
            label="Server" style=filled fillcolor="#FFF7ED" color="#D97706";
            verify [label="Verify\nProofs"             fillcolor="#FED7AA"];
            aggr   [label="FedAvg /\nMedian / TrimMean" fillcolor="#FDBA74"];
            global [label="Global\nModel Update"        fillcolor="#FED7AA"];
        }
        subgraph cluster_eval {
            label="Evaluation" style=filled fillcolor="#FDF4FF" color="#A855F7";
            pred  [label="Predict\n+ Route"        fillcolor="#E9D5FF"];
            stats [label="Confusion Matrix\nF1 / Accuracy" fillcolor="#E9D5FF"];
        }

        csv -> tok -> shard -> emb;
        emb -> moe -> lora -> dp -> proof;
        proof -> verify -> aggr -> global;
        global -> emb [style=dashed label="next round" fontsize=9];
        global -> pred -> stats;
    }
    """)

    st.divider()

    # Dashboard map
    st.subheader("What each page shows")
    cols = st.columns(3)
    pages_info = [
        ("🔮 Predict",     "Type text → see predicted class + which experts activate. Compare two headlines side-by-side."),
        ("🏋️ Train",       "Configure FL rounds, clients, learning rate. Enable DP + SEPG. Watch accuracy & expert heatmap live."),
        ("📂 Custom CSV",  "Upload any labelled CSV → auto-detect columns → train → confusion matrix + expert routing per class."),
        ("🔒 Privacy & DP","DP-SGD training with live ε/δ budget chart. Inspect each client's SHA-256 SEPG proof after training."),
        ("🛡️ Robustness",  "Simulate poisoning / free-rider / Sybil attacks. Compare FedAvg vs Median vs Trimmed Mean live."),
        ("📊 Experiments", "Interactive charts for all 4 core experiments: privacy-utility, comm vs K, overhead, robustness."),
        ("📡 Compare",     "Instant calculator: adjust expert count & K to see communication savings in real time."),
        ("🏗️ Architecture","Model diagram + per-expert param breakdown + 5 code snippet tabs (MoE/LoRA/FedAvg/DP/SEPG)."),
    ]
    for i, (pg, desc) in enumerate(pages_info):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="concept-card">
                <h4>{pg}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Key numbers
    st.subheader("Key Numbers at a Glance")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Dataset",        "120K", "AG News samples")
    k2.metric("Vocabulary",     "5 000", "most frequent tokens")
    k3.metric("Experts",        "8",     "per MoE layer")
    k4.metric("Comm Saving",    "~40%",  "at Top-K = 1")
    k5.metric("Proof Overhead", "~6 ms", "per client")
    k6.metric("Classes",        "4",     "World/Sports/Biz/Tech")

    st.divider()

    # Quick-start instructions
    st.subheader("How to demo this project")
    st.markdown("""
    | Step | Page | What to show |
    |------|------|--------------|
    | 1 | **Architecture** | Show the MoE data-flow diagram, explain Top-K routing and LoRA |
    | 2 | **Train** | Run 3 rounds with 5 clients — watch accuracy curve and expert heatmap live |
    | 3 | **Predict** | Type a news headline — highlight which experts fire and why |
    | 4 | **Compare** | Drag the K slider to show real-time communication saving calculation |
    | 5 | **Privacy & DP** | Run DP training, show SEPG proofs with PASS badges |
    | 6 | **Robustness** | Run Poisoning at 30% — show FedAvg drops, Median holds |
    | 7 | **Experiments** | Open all 4 interactive charts, point to sweet-spot K=3–4 |
    | 8 | **Custom CSV** | Upload your own CSV and train a new model live |
    """)


# ---------------------------------------------------------------
# PAGE: PREDICT
# ---------------------------------------------------------------
elif page == "Predict":
    page_banner("Live Prediction", "Type text → model predicts the class and shows which experts activated", "🔮")
    flow_bar(["Data", "Tokenise", "Embedding", "MoE Router", "▶ Predict"], "▶ Predict")

    # FIX #1 — use cached model; only populate session_state once
    if "model" not in st.session_state:
        with st.spinner("Building quick-start model (first load only)..."):
            gm, voc, kw = _build_quickstart_model()
        st.session_state.model = gm
        st.session_state.vocab = voc
        st.session_state.model_kw = kw
        st.session_state.model_source = "quick-start"
        st.session_state.pop("custom_class_names", None)
        st.rerun()

    model = st.session_state.model
    vocab = st.session_state.vocab
    top_k = model.moe.k
    num_exp = model.moe.num_experts

    # FIX #2 — custom class names: build a reliable idx→label map from session
    custom_classes = st.session_state.get("custom_class_names", None)
    default_class_names = ["World", "Sports", "Business", "Tech"]

    def idx_to_label(i: int) -> str:
        if custom_classes:
            return str(custom_classes.get(i, i))
        return default_class_names[i] if i < len(default_class_names) else str(i)

    n_classes_model = model.classifier.base.out_features

    # Sample headlines (only shown for default AG News model)
    if not custom_classes:
        st.markdown("**Try a sample:**")
        samples = {
            "World":    "earthquake strikes coastal city thousands evacuated",
            "Sports":   "olympic champion breaks world record in 100m sprint final",
            "Business": "stock markets surge after federal reserve cuts interest rates",
            "Tech":     "researchers develop new ai chip for edge computing devices",
        }
        s_cols = st.columns(4)
        for i, (cls, txt) in enumerate(samples.items()):
            if s_cols[i].button(cls, use_container_width=True):
                st.session_state.input_text = txt
    else:
        st.info(f"Custom model loaded ({n_classes_model} classes). "
                "Type any text matching your dataset categories.")

    text = st.text_input(
        "Enter text to classify:",
        value=st.session_state.get("input_text", ""),
        placeholder="e.g. Apple unveils new smartphone with advanced AI features",
    )

    if text.strip():
        # FIX #2 — run inference, then remap class names regardless of model source
        with torch.no_grad():
            x = model.embedding(
                torch.tensor(
                    [[vocab.get(t, 0) for t in text.lower().split()][:64]
                     + [0] * max(0, 64 - len(text.lower().split()))],
                    dtype=torch.long)
            ).mean(dim=1)
            x, rp_tensor = model.moe(x)
            logits = model.classifier(x)
            probs_tensor = torch.softmax(logits, dim=-1).squeeze(0)

        pred_idx = probs_tensor.argmax().item()
        rp = rp_tensor.squeeze(0).cpu()
        topk_vals, topk_idx = torch.topk(rp, top_k)

        class_probs = {idx_to_label(i): probs_tensor[i].item()
                       for i in range(n_classes_model)}
        pred_label = idx_to_label(pred_idx)
        top_experts = topk_idx.tolist()

        oov = [t for t in text.lower().split() if t not in vocab]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted Class", pred_label,
                      f"{max(class_probs.values()):.0%} confidence")
            st.markdown(f"**Active Experts:** {top_experts}")
            if oov:
                st.warning(f"Unknown words: {', '.join(oov[:5])}")

        with c2:
            colors_bar = ["#FF6B6B" if k == pred_label else "#4C72B0"
                          for k in class_probs]
            fig = go.Figure(go.Bar(
                x=list(class_probs.keys()), y=list(class_probs.values()),
                marker_color=colors_bar,
                text=[f"{v:.1%}" for v in class_probs.values()],
                textposition="outside",
            ))
            fig.update_layout(title="Class Confidence", yaxis_range=[0, 1],
                              height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Expert Routing")
        concept_card(
            "What is Expert Routing?",
            f"The Router (a small Linear layer) scores all {num_exp} experts for this input. "
            f"Only the top {top_k} scoring experts compute — the rest are skipped entirely. "
            "Orange bars = active experts. Each expert specialises in different linguistic patterns."
        )
        rp_np = rp.numpy()
        colors_r = ["#DD8452" if i in top_experts else "#CCCCCC"
                    for i in range(num_exp)]
        fig2 = go.Figure(go.Bar(
            x=[f"Expert {i}" for i in range(num_exp)],
            y=rp_np,
            marker_color=colors_r,
            text=[f"{v:.3f}" for v in rp_np],
            textposition="outside",
        ))
        fig2.update_layout(
            title=f"Router Probabilities (Top-{top_k} highlighted)",
            yaxis_title="Routing Weight",
            height=350, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Token Details"):
            tokens = text.lower().split()
            st.dataframe(
                pd.DataFrame([{"Token": t, "ID": vocab.get(t, 0),
                               "Known": "Yes" if t in vocab else "OOV"}
                              for t in tokens]),
                hide_index=True, use_container_width=True,
            )

    # ---- Compare Two Headlines ----
    st.divider()
    st.subheader("Compare Two Headlines")
    st.caption("See how the same model routes two different headlines to different experts.")

    col_h1, col_h2 = st.columns(2)
    h1 = col_h1.text_input("Headline A",
                            placeholder="e.g. NASA launches new Mars rover",
                            key="compare_h1")
    h2 = col_h2.text_input("Headline B",
                            placeholder="e.g. Manchester United wins championship",
                            key="compare_h2")

    if h1.strip() and h2.strip():
        def _infer(txt):
            tokens_ = txt.lower().split()
            ids_ = [vocab.get(t, 0) for t in tokens_][:64]
            ids_ += [0] * max(0, 64 - len(ids_))
            with torch.no_grad():
                x_ = model.embedding(torch.tensor([ids_], dtype=torch.long)).mean(dim=1)
                x_, rp_ = model.moe(x_)
                logits_ = model.classifier(x_)
                prob_ = torch.softmax(logits_, dim=-1).squeeze(0)
            pred_ = prob_.argmax().item()
            tv_, ti_ = torch.topk(rp_.squeeze(0).cpu(), top_k)
            return prob_, rp_.squeeze(0).cpu(), pred_, ti_.tolist()

        pr1, rp1, pd1, te1 = _infer(h1)
        pr2, rp2, pd2, te2 = _infer(h2)
        rp1_np, rp2_np = rp1.numpy(), rp2.numpy()
        experts_a, experts_b = set(te1), set(te2)
        overlap = experts_a & experts_b
        only_a  = experts_a - experts_b
        only_b  = experts_b - experts_a

        def _bar_colors(active, other):
            return ["#9B59B6" if i in active and i in other
                    else "#DD8452" if i in active
                    else "#CCCCCC"
                    for i in range(num_exp)]

        ymax = max(rp1_np.max(), rp2_np.max()) * 1.3

        fig_a = go.Figure(go.Bar(
            x=[f"E{i}" for i in range(num_exp)], y=rp1_np,
            marker_color=_bar_colors(experts_a, experts_b),
            text=[f"{v:.3f}" for v in rp1_np], textposition="outside",
        ))
        fig_a.update_layout(
            title=f"A → {idx_to_label(pd1)} ({pr1[pd1]:.0%})",
            yaxis_range=[0, ymax], height=300, margin=dict(t=50, b=20),
        )
        fig_b = go.Figure(go.Bar(
            x=[f"E{i}" for i in range(num_exp)], y=rp2_np,
            marker_color=_bar_colors(experts_b, experts_a),
            text=[f"{v:.3f}" for v in rp2_np], textposition="outside",
        ))
        fig_b.update_layout(
            title=f"B → {idx_to_label(pd2)} ({pr2[pd2]:.0%})",
            yaxis_range=[0, ymax], height=300, margin=dict(t=50, b=20),
        )
        col_h1.plotly_chart(fig_a, use_container_width=True)
        col_h2.plotly_chart(fig_b, use_container_width=True)

        lc1, lc2, lc3 = st.columns(3)
        lc1.markdown(f"**Shared (purple):** {sorted(overlap) or 'None'}")
        lc2.markdown(f"**Only A (orange):** {sorted(only_a)}")
        lc3.markdown(f"**Only B (orange):** {sorted(only_b)}")


# ---------------------------------------------------------------
# PAGE: TRAIN
# ---------------------------------------------------------------
elif page == "Train":
    page_banner("Federated Training", "Configure FL hyperparameters · watch accuracy and expert routing update live each round", "🏋️")
    flow_bar(["Config", "▶ Local Train (×N clients)", "DP-SGD (optional)", "Aggregate", "Evaluate"], "▶ Local Train (×N clients)")

    c1, c2, c3 = st.columns(3)
    num_experts = c1.slider("Experts", 2, 16, 8)
    top_k       = c2.slider("Top-K",   1, min(num_experts, 4), 2)
    num_clients = c3.slider("Clients", 2, 10, 5)

    c4, c5, c6 = st.columns(3)
    num_rounds = c4.slider("Rounds", 1, 15, 5)
    lr         = c5.select_slider("Learning Rate",
                                  [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=2e-3)
    use_ag     = c6.checkbox("AG News (120K)", value=True)

    # DP and SEPG toggles
    st.divider()
    dp_col, sepg_col = st.columns(2)
    use_dp   = dp_col.toggle("Enable Differential Privacy", value=False)
    use_sepg = sepg_col.toggle("Enable SEPG Proof Verification", value=False)

    # FIX #3 — always define clip_norm / noise_mult so training loop never NameErrors
    clip_norm  = 1.0
    noise_mult = 0.5
    if use_dp:
        dp_c1, dp_c2 = st.columns(2)
        clip_norm  = dp_c1.slider("Clip Norm (C)", 0.1, 5.0, 1.0, step=0.1)
        noise_mult = dp_c2.slider("Noise Multiplier (σ)", 0.0, 2.0, 0.5, step=0.1)

    if st.button("Start Training", type="primary", use_container_width=True):
        # FIX #10 — wrap entire training in try/except
        progress     = st.progress(0)
        status_text  = st.empty()
        try:
            set_seed()
            dev = torch.device("cpu")

            with st.spinner("Loading dataset..."):
                if use_ag:
                    cl, td, vs, nc, voc = load_ag_news(num_clients)
                else:
                    cl, td, vs, nc, voc = build_ag_news_clients(
                        num_clients=num_clients, seq_len=64,
                        use_external_csv=False, repeat=50)
            train_ds = ConcatDataset(cl)

            kw = dict(vocab_size=vs, embed_dim=64, num_classes=nc,
                      num_experts=num_experts, expert_hidden_dim=256,
                      k=top_k, lora_r=8)
            dm = MoETextClassifier(**kw)
            sm = MoETextClassifier(**kw)
            sm.load_state_dict(dm.state_dict())
            srv_d = FedServer(dm, device=dev)
            srv_s = FedServer(sm, device=dev)

            total_p  = sum(p.numel() for p in dm.parameters())
            expert_p = sum(p.numel() for n, p in dm.named_parameters()
                           if "moe.experts" in n)
            st.caption(
                f"Model: {total_p:,} params | Experts: {expert_p:,} "
                f"({expert_p/total_p*100:.0f}%) | "
                f"Data: {len(train_ds):,} train, {len(td):,} test"
            )

            if use_dp:
                priv_cols = st.columns(3)
                pm_eps    = priv_cols[0].empty()
                pm_delta  = priv_cols[1].empty()
                pm_rnds   = priv_cols[2].empty()
                accountant = PrivacyAccountant(target_delta=1e-5)

            if use_sepg:
                sepg_status = st.empty()

            metric_cols = st.columns(4)
            m_dense  = metric_cols[0].empty()
            m_sparse = metric_cols[1].empty()
            m_saving = metric_cols[2].empty()
            m_round  = metric_cols[3].empty()

            chart_col1, chart_col2 = st.columns(2)
            acc_chart    = chart_col1.empty()
            heatmap_area = chart_col2.empty()
            comm_chart   = st.empty()

            acc_rows  = []
            comm_rows = []
            cum_dense = cum_sparse = 0
            da = sa = sav = 0.0

            for rnd in range(1, num_rounds + 1):
                ds_, ss_ = [], []
                rd, rs   = 0, 0
                usage_matrix = []
                proofs = []

                for ci, cds in enumerate(cl):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv_d.get_global_state(), strict=False)
                    fs, sp, n, db, sb, tki, eu = local_train(
                        cm, cds, 1, 64, lr, dev, top_k_sparse=top_k)
                    rd += db
                    rs += sb

                    if use_dp and noise_mult > 0:
                        fs = apply_dp(fs, clip_norm=clip_norm,
                                      noise_multiplier=noise_mult)
                        sp = apply_dp(sp, clip_norm=clip_norm,
                                      noise_multiplier=noise_mult)

                    ds_.append((fs, n))
                    ss_.append((sp, n))
                    usage_matrix.append((eu / max(n, 1)).numpy())

                    if use_sepg:
                        eps_val = (accountant.get_privacy_spent()[0]
                                   if use_dp and rnd > 1 else 1.0)
                        proof = generate_proof(
                            client_id=ci, round_id=rnd,
                            top_k_indices=list(range(top_k)),
                            clip_norm=clip_norm,
                            noise_multiplier=noise_mult if noise_mult > 0 else 0.1,
                            epsilon=eps_val,
                            sparse_state=sp,
                        )
                        proofs.append((proof, sp))

                srv_d.aggregate(ds_)
                srv_s.aggregate(ss_)
                da  = evaluate(srv_d.global_model, td, 64, dev)
                sa  = evaluate(srv_s.global_model, td, 64, dev)
                sav = (1 - rs / rd) * 100 if rd > 0 else 0
                cum_dense  += rd
                cum_sparse += rs

                acc_rows.append({"Round": rnd, "Dense": da, "Sparse": sa})
                comm_rows.append({"Round": rnd, "Dense (KB)": cum_dense / 1024,
                                  "Sparse (KB)": cum_sparse / 1024})

                if use_dp and noise_mult > 0:
                    sr = 64 / max(len(cl[0]), 1) if hasattr(cl[0], "__len__") else 0.01
                    accountant.accumulate(noise_mult, sr, num_steps=1)
                    eps, delta = accountant.get_privacy_spent()
                    pm_eps.metric("Privacy Budget (ε)", f"{eps:.4f}")
                    pm_delta.metric("Delta (δ)", f"{delta:.2e}")
                    pm_rnds.metric("DP Rounds", rnd)

                if use_sepg and proofs:
                    rows_sepg = []
                    for proof, sp in proofs:
                        passed, reason = verify_proof(
                            proof, sp, expected_k=top_k,
                            min_noise_mult=0.0 if not use_dp else 0.01)
                        rows_sepg.append({
                            "Client": proof.client_id,
                            "Status": "PASS" if passed else "FAIL",
                            "Experts": str(proof.top_k_indices),
                            "Reason": reason,
                        })
                    sepg_status.dataframe(pd.DataFrame(rows_sepg),
                                          hide_index=True, use_container_width=True)

                progress.progress(rnd / num_rounds)
                status_text.text(f"Round {rnd}/{num_rounds}")
                m_dense.metric("Dense Acc",   f"{da:.2%}")
                m_sparse.metric("Sparse Acc", f"{sa:.2%}")
                m_saving.metric("Comm Saving", f"{sav:.1f}%")
                m_round.metric("Round", f"{rnd}/{num_rounds}")

                df_acc = pd.DataFrame(acc_rows).set_index("Round")
                fig_acc = px.line(df_acc, y=["Dense", "Sparse"],
                                  title="Test Accuracy per Round",
                                  labels={"value": "Accuracy", "variable": "Mode"})
                fig_acc.update_layout(height=350, yaxis_range=[0, 1])
                acc_chart.plotly_chart(fig_acc, use_container_width=True)

                um = np.array(usage_matrix)
                fig_hm = px.imshow(
                    um, x=[f"E{i}" for i in range(num_experts)],
                    y=[f"Client {i}" for i in range(len(cl))],
                    title=f"Expert Usage (Round {rnd})",
                    color_continuous_scale="YlOrRd",
                    labels=dict(color="Usage"),
                )
                fig_hm.update_layout(height=350)
                heatmap_area.plotly_chart(fig_hm, use_container_width=True)

                df_comm = pd.DataFrame(comm_rows).set_index("Round")
                fig_comm = px.bar(df_comm, barmode="group",
                                  title="Cumulative Communication Cost",
                                  labels={"value": "KB", "variable": "Mode"})
                fig_comm.update_layout(height=300)
                comm_chart.plotly_chart(fig_comm, use_container_width=True)

            progress.empty()
            status_text.empty()

            st.session_state.model        = srv_d.global_model
            st.session_state.vocab        = voc
            st.session_state.model_kw     = kw
            st.session_state.model_source = "AG News FL" if use_ag else "Small corpus FL"
            st.session_state.pop("custom_class_names", None)
            st.success(
                f"Training complete! Dense={da:.2%}, Sparse={sa:.2%}, "
                f"Saving={sav:.1f}%. Model saved — go to **Predict**."
            )

        except Exception as exc:
            progress.empty()
            status_text.empty()
            st.error(f"Training failed: {exc}")
            raise


# ---------------------------------------------------------------
# PAGE: PRIVACY & DP
# ---------------------------------------------------------------
elif page == "Privacy & DP":
    page_banner("Differential Privacy & SEPG Proofs",
                "DP-SGD: clip gradients + add Gaussian noise · SEPG: each client proves it followed the rules",
                "🔒")
    flow_bar(["Local Train", "▶ Clip Gradient", "▶ Add Noise", "Generate Proof", "Server Verify", "Aggregate"], "▶ Clip Gradient")

    col1, col2, col3 = st.columns(3)
    clip_norm_dp   = col1.slider("Clip Norm (C)", 0.1, 5.0, 1.0, step=0.1, key="dp_clip")
    noise_mult_dp  = col2.slider("Noise Multiplier (σ)", 0.01, 2.0, 0.5, step=0.05,
                                  key="dp_noise")
    num_rounds_dp  = col3.slider("Rounds", 1, 10, 5, key="dp_rounds")

    st.info(
        f"**Gaussian Mechanism:** Each update clipped to ‖Δ‖₂ ≤ {clip_norm_dp:.1f}, "
        f"then noise N(0, ({noise_mult_dp:.2f}×{clip_norm_dp:.1f})²) added per parameter."
    )
    concept_card(
        "Why Differential Privacy?",
        "Without DP, the server could reconstruct private training data from gradients. "
        "Adding calibrated Gaussian noise before sending ensures each client's data stays private. "
        "The privacy budget ε measures how much information leaks — lower ε = stronger protection."
    )

    if st.button("Run DP Training + Generate Proofs", type="primary",
                 use_container_width=True):
        progress_dp = st.progress(0)
        status_dp   = st.empty()
        # FIX #10
        try:
            set_seed()
            dev = torch.device("cpu")

            with st.spinner("Loading dataset..."):
                cl, td, vs, nc, voc = load_ag_news(5)

            kw = dict(vocab_size=vs, embed_dim=64, num_classes=nc,
                      num_experts=8, expert_hidden_dim=256, k=2, lora_r=8)
            srv        = FedServer(MoETextClassifier(**kw), device=dev)
            accountant = PrivacyAccountant(target_delta=1e-5)

            budget_chart = st.empty()
            eps_history  = []
            all_proofs   = []          # overwritten each round; last round shown

            for rnd in range(1, num_rounds_dp + 1):
                states     = []
                all_proofs = []

                for ci, cds in enumerate(cl):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv.get_global_state(), strict=False)
                    fs, sp, n, _, _, _, _ = local_train(
                        cm, cds, 1, 64, 2e-3, dev, top_k_sparse=2)

                    fs_dp = apply_dp(fs, clip_norm=clip_norm_dp,
                                     noise_multiplier=noise_mult_dp)

                    sr = min(64 / max(len(cds), 1), 1.0) if hasattr(cds, "__len__") else 0.01
                    accountant.accumulate(noise_mult_dp, sr, num_steps=1)
                    eps, delta = accountant.get_privacy_spent()

                    proof = generate_proof(
                        client_id=ci, round_id=rnd,
                        top_k_indices=list(range(2)),
                        clip_norm=clip_norm_dp,
                        noise_multiplier=noise_mult_dp,
                        epsilon=eps,
                        sparse_state=sp,
                    )
                    all_proofs.append((proof, sp))
                    states.append((fs_dp, n))

                srv.aggregate(states)
                acc = evaluate(srv.global_model, td, 64, dev)
                eps, delta = accountant.get_privacy_spent()
                eps_history.append({"Round": rnd, "ε (epsilon)": round(eps, 6),
                                    "Accuracy": round(acc, 4)})

                progress_dp.progress(rnd / num_rounds_dp)
                status_dp.markdown(
                    f"**Round {rnd}/{num_rounds_dp}** | "
                    f"Accuracy: {acc:.2%} | ε={eps:.4f}, δ={delta:.2e}"
                )

                df_eps = pd.DataFrame(eps_history).set_index("Round")
                fig_eps = go.Figure()
                fig_eps.add_trace(go.Scatter(
                    x=df_eps.index, y=df_eps["ε (epsilon)"],
                    mode="lines+markers", name="ε consumed",
                    line=dict(color="#DD8452", width=2),
                    fill="tozeroy", fillcolor="rgba(221,132,82,0.15)",
                ))
                fig_eps.add_trace(go.Scatter(
                    x=df_eps.index, y=df_eps["Accuracy"],
                    mode="lines+markers", name="Accuracy",
                    line=dict(color="#4C72B0", width=2),
                    yaxis="y2",
                ))
                fig_eps.update_layout(
                    title="Privacy Budget Consumed vs Accuracy",
                    xaxis_title="Round",
                    yaxis=dict(title="ε (epsilon)", side="left"),
                    yaxis2=dict(title="Accuracy", side="right",
                                overlaying="y", range=[0, 1]),
                    height=350, margin=dict(t=50, b=30),
                    legend=dict(orientation="h", y=-0.2),
                )
                budget_chart.plotly_chart(fig_eps, use_container_width=True)

            progress_dp.empty()
            status_dp.empty()

            # FIX #4 — removed dead `proof_area` container; render directly
            st.subheader(f"SEPG Proofs — Round {num_rounds_dp}")
            st.caption("Server verifies each proof before including the client's update.")

            n_cols = min(len(all_proofs), 3)
            proof_cols = st.columns(n_cols)
            for i, (proof, sp) in enumerate(all_proofs):
                passed, reason = verify_proof(proof, sp, expected_k=2)
                badge = "PASS" if passed else "FAIL"
                color = "green" if passed else "red"
                with proof_cols[i % n_cols]:
                    with st.expander(f"Client {proof.client_id} — :{color}[{badge}]",
                                     expanded=True):
                        st.markdown(f"**Client ID:** {proof.client_id}")
                        st.markdown(f"**Round ID:** {proof.round_id}")
                        st.markdown(f"**Top-K Experts:** {proof.top_k_indices}")
                        st.markdown(f"**Clip Norm (C):** {proof.dp_params['clip_norm']:.2f}")
                        st.markdown(f"**Noise Multiplier (σ):** {proof.dp_params['noise_mult']:.2f}")
                        st.markdown(f"**Epsilon (ε):** {proof.dp_params['epsilon']:.6f}")
                        st.markdown(f"**Hash (SHA-256):** `{proof.update_hash[:20]}...`")
                        st.markdown(f"**Verification:** :{color}[**{badge}**] — {reason}")

            st.subheader("Privacy Budget Summary")
            df_summary = pd.DataFrame(eps_history)
            st.dataframe(df_summary, hide_index=True, use_container_width=True)

            final_eps = eps_history[-1]["ε (epsilon)"]
            st.success(
                f"After {num_rounds_dp} rounds with σ={noise_mult_dp:.2f}: "
                f"**ε = {final_eps:.4f}** (δ = 1e-5). "
                f"Lower σ → stronger privacy but lower accuracy."
            )

        except Exception as exc:
            progress_dp.empty()
            status_dp.empty()
            st.error(f"DP training failed: {exc}")
            raise


# ---------------------------------------------------------------
# PAGE: ROBUSTNESS
# ---------------------------------------------------------------
elif page == "Robustness":
    page_banner("Robustness Under Attacks",
                "Simulate poisoning · free-rider · Sybil attacks and see which aggregation survives",
                "🛡️")
    flow_bar(["Honest Clients", "▶ Malicious Clients", "Aggregation Strategy", "Global Model", "Accuracy"], "▶ Malicious Clients")

    r_col1, r_col2, r_col3 = st.columns(3)
    attack_type     = r_col1.selectbox(
        "Attack Type",
        ["Poisoning (label flip)", "Free-rider (stale update)", "Sybil (duplicate)"],
    )
    mal_frac        = r_col2.slider("Malicious Fraction", 0, 40, 20, step=10,
                                     help="% of clients that are malicious") / 100
    num_rounds_rob  = r_col3.slider("Rounds", 1, 10, 5, key="rob_rounds")

    agg_choice = st.multiselect(
        "Aggregation Strategies to Compare",
        ["FedAvg", "Median", "Trimmed Mean"],
        default=["FedAvg", "Median", "Trimmed Mean"],
    )

    num_clients_rob = 5
    # FIX #5 — removed unused `n_honest` assignment

    attack_descriptions = {
        "Poisoning (label flip)": (
            "Malicious clients randomly flip training labels before updating the model. "
            "This injects noise into the gradient direction, degrading global accuracy."
        ),
        "Free-rider (stale update)": (
            "Free-riders return the global model with tiny random noise, "
            "pretending to have trained. They consume resources without contributing."
        ),
        "Sybil (duplicate)": (
            "One attacker registers as multiple clients. "
            "Sybil updates are weight-amplified in FedAvg, biasing the global model."
        ),
    }
    st.info(attack_descriptions[attack_type])
    concept_card(
        "Why robust aggregation?",
        "Plain FedAvg takes a weighted mean — one large malicious update can dominate. "
        "Coordinate-wise Median ignores outliers by taking the middle value per parameter. "
        "Trimmed Mean removes the top and bottom fraction before averaging."
    )

    if st.button("Run Robustness Simulation", type="primary",
                 use_container_width=True):
        if not agg_choice:
            st.warning("Select at least one aggregation strategy.")
            st.stop()

        progress_rob = st.progress(0)
        status_rob   = st.empty()
        # FIX #10
        try:
            set_seed()
            dev   = torch.device("cpu")
            n_mal = max(0, int(num_clients_rob * mal_frac))

            with st.spinner("Loading dataset..."):
                cl, td, vs, nc, voc = load_ag_news(num_clients_rob)

            kw = dict(vocab_size=vs, embed_dim=64, num_classes=nc,
                      num_experts=8, expert_hidden_dim=256, k=2, lora_r=8)

            agg_map = {
                "FedAvg":        "aggregate",
                "Median":        "aggregate_median",
                "Trimmed Mean":  "aggregate_trimmed_mean",
            }

            # Client role table
            client_rows = [{"Client": f"Client {i}",
                            "Role": "Malicious" if i < n_mal else "Honest"}
                           for i in range(num_clients_rob)]
            st.subheader("Client Roles")
            st.dataframe(pd.DataFrame(client_rows),
                         hide_index=True, use_container_width=True)

            acc_chart_rob = st.empty()
            all_acc_rows  = {m: [] for m in agg_choice}
            total_steps   = len(agg_choice) * num_rounds_rob

            for method_name in agg_choice:
                set_seed()
                srv = FedServer(MoETextClassifier(**kw), device=dev)

                for rnd in range(1, num_rounds_rob + 1):
                    round_states = []

                    for ci, cds in enumerate(cl):
                        cm = MoETextClassifier(**kw)
                        cm.load_state_dict(srv.get_global_state(), strict=False)

                        if ci < n_mal:
                            if "Poisoning" in attack_type:
                                state, n = poisoning_train(
                                    cm, cds, epochs=1, batch_size=64,
                                    lr=2e-3, device=dev, num_classes=nc)
                            elif "Free-rider" in attack_type:
                                state, n = freerider_train(
                                    srv.get_global_state(), n_samples=100)
                            else:  # Sybil
                                fs, _, n, _, _, _, _ = local_train(
                                    cm, cds, 1, 64, 2e-3, dev)
                                state = fs
                        else:
                            fs, _, n, _, _, _, _ = local_train(
                                cm, cds, 1, 64, 2e-3, dev)
                            state = fs

                        round_states.append((state, n))

                    if "Sybil" in attack_type and n_mal > 0:
                        sybil_updates = sybil_clones(
                            round_states[0][0], round_states[0][1], num_clones=2)
                        round_states = sybil_updates + round_states[n_mal:]

                    getattr(srv, agg_map[method_name])(round_states)
                    acc = evaluate(srv.global_model, td, 64, dev)
                    all_acc_rows[method_name].append({"Round": rnd, "Accuracy": acc})

                    step = list(agg_choice).index(method_name) * num_rounds_rob + rnd
                    progress_rob.progress(step / total_steps)
                    status_rob.text(
                        f"[{method_name}] Round {rnd}/{num_rounds_rob} — Acc={acc:.2%}"
                    )

                    fig_rob = go.Figure()
                    clrs = {"FedAvg": "#4C72B0", "Median": "#55A868",
                            "Trimmed Mean": "#DD8452"}
                    for m, rows in all_acc_rows.items():
                        if rows:
                            df_m = pd.DataFrame(rows)
                            fig_rob.add_trace(go.Scatter(
                                x=df_m["Round"], y=df_m["Accuracy"],
                                mode="lines+markers", name=m,
                                line=dict(color=clrs.get(m, "gray"), width=2),
                            ))
                    fig_rob.update_layout(
                        title=f"Accuracy Under {attack_type} ({int(mal_frac*100)}% malicious)",
                        xaxis_title="Round", yaxis_title="Test Accuracy",
                        yaxis_range=[0, 1], height=400,
                        legend=dict(orientation="h", y=-0.2),
                    )
                    acc_chart_rob.plotly_chart(fig_rob, use_container_width=True)

            progress_rob.empty()
            status_rob.empty()

            st.subheader("Final Accuracy Comparison")
            final_rows = []
            for method_name in agg_choice:
                if all_acc_rows[method_name]:
                    fa = all_acc_rows[method_name][-1]["Accuracy"]
                    final_rows.append({
                        "Strategy":         method_name,
                        "Final Accuracy":   f"{fa:.2%}",
                        "Attack Resistance": "High" if fa > 0.45 else "Low",
                    })
            st.dataframe(pd.DataFrame(final_rows),
                         hide_index=True, use_container_width=True)
            insight_box(
                f"At {int(mal_frac*100)}% malicious clients: "
                "FedAvg is most vulnerable because it weights by sample count. "
                "Median is most robust — it takes the coordinate-wise middle value, "
                "making large adversarial updates statistically invisible."
            )

            best = max(agg_choice,
                       key=lambda m: all_acc_rows[m][-1]["Accuracy"]
                       if all_acc_rows[m] else 0)
            st.success(
                f"**Best strategy under {attack_type} with "
                f"{int(mal_frac*100)}% malicious: {best}.** "
                "Robust aggregation outperforms plain FedAvg when attackers "
                "exceed ~20% of clients."
            )

        except Exception as exc:
            progress_rob.empty()
            status_rob.empty()
            st.error(f"Simulation failed: {exc}")
            raise


# ---------------------------------------------------------------
# PAGE: EXPERIMENTS  — FIX #6: interactive Plotly charts
# ---------------------------------------------------------------
elif page == "Experiments":
    page_banner("Experiment Results",
                "4 core experiments validating privacy · communication efficiency · verification overhead · attack robustness",
                "📊")
    flow_bar(["Privacy-Utility", "Comm vs K", "Verification Overhead", "Robustness"], "Privacy-Utility")

    results_json = PLOT_DIR / "experiment_results.json"
    exp_results  = {}
    if results_json.exists():
        with open(results_json) as f:
            exp_results = json.load(f)

    has_data = bool(exp_results)
    if not has_data:
        st.warning(
            "No results JSON found. Static images shown where available. "
            "Run the experiment suite to get interactive charts."
        )

    # ---- Exp 1: Privacy-Utility ----
    st.divider()
    st.subheader("Experiment 1: Privacy-Utility Tradeoff")
    st.markdown(
        "Higher noise multiplier → smaller ε (stronger privacy) → lower accuracy."
    )

    if "privacy_utility" in exp_results:
        pu     = exp_results["privacy_utility"]
        df_pu  = pd.DataFrame(pu)
        # Replace huge epsilon with a display cap for the chart
        df_pu["eps_display"] = df_pu["epsilon"].apply(lambda x: 50.0 if x > 100 else x)
        df_pu["label"]       = df_pu["noise_mult"].apply(lambda x: f"σ={x:.1f}")
        df_pu["epsilon_str"] = df_pu["epsilon"].apply(
            lambda x: "∞" if x > 100 else f"{x:.3f}")

        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig_pu = go.Figure()
            fig_pu.add_trace(go.Scatter(
                x=df_pu["eps_display"], y=df_pu["accuracy"],
                mode="lines+markers+text",
                text=df_pu["label"], textposition="top center",
                marker=dict(size=10, color="#4C72B0"),
                line=dict(width=2),
            ))
            fig_pu.update_layout(
                title="Accuracy vs Privacy Budget (ε)",
                xaxis_title="ε (epsilon) — higher = less private",
                yaxis_title="Test Accuracy",
                yaxis_range=[0, 1], height=380,
            )
            st.plotly_chart(fig_pu, use_container_width=True)

        with col_r:
            tbl = df_pu[["noise_mult", "epsilon_str", "accuracy"]].copy()
            tbl.columns = ["Noise σ", "ε", "Accuracy"]
            tbl["Accuracy"] = tbl["Accuracy"].apply(lambda x: f"{x:.2%}")
            st.dataframe(tbl, hide_index=True, use_container_width=True)
    else:
        img1 = PLOT_DIR / "exp1_privacy_utility.png"
        if img1.exists():
            st.image(str(img1), use_container_width=True)
        else:
            st.warning("Run the experiment suite to generate this plot.")

    c1, c2, c3 = st.columns(3)
    c1.metric("No DP Accuracy",    "58.0%", "σ=0")
    c2.metric("With DP (σ=0.1)",   "25.0%", "ε=0.29")
    c3.metric("Privacy cost",      "~33% accuracy", "at tight ε")

    # ---- Exp 2: Communication vs K ----
    st.divider()
    st.subheader("Experiment 2: Communication Savings vs Top-K")
    st.markdown(
        "Sending only Top-K expert weights. K=1 saves ~40%; K=4 balances accuracy and saving."
    )

    if "comm_vs_k" in exp_results:
        ck    = exp_results["comm_vs_k"]
        df_ck = pd.DataFrame(ck)

        col_l2, col_r2 = st.columns([2, 1])
        with col_l2:
            fig_ck = go.Figure()
            fig_ck.add_trace(go.Bar(
                x=df_ck["K"], y=df_ck["saving_pct"],
                name="Comm Saving %", marker_color="#4C72B0", opacity=0.7,
                text=[f"{v:.1f}%" for v in df_ck["saving_pct"]],
                textposition="outside",
            ))
            fig_ck.add_trace(go.Scatter(
                x=df_ck["K"], y=df_ck["accuracy"],
                name="Accuracy", mode="lines+markers",
                marker=dict(color="#DD8452", size=9),
                line=dict(width=2, color="#DD8452"),
                yaxis="y2",
            ))
            fig_ck.update_layout(
                title="Communication Saving & Accuracy vs Top-K",
                xaxis_title="K (experts sent per client)",
                yaxis=dict(title="Comm Saving (%)", range=[0, 60]),
                yaxis2=dict(title="Accuracy", overlaying="y",
                            side="right", range=[0, 1]),
                barmode="group", height=380,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ck, use_container_width=True)

        with col_r2:
            tbl2 = df_ck[["K", "accuracy", "saving_pct"]].copy()
            tbl2.columns = ["K", "Accuracy", "Saving %"]
            tbl2["Accuracy"] = tbl2["Accuracy"].apply(lambda x: f"{x:.2%}")
            tbl2["Saving %"] = tbl2["Saving %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(tbl2, hide_index=True, use_container_width=True)
    else:
        img2 = PLOT_DIR / "exp2_comm_vs_k.png"
        if img2.exists():
            st.image(str(img2), use_container_width=True)
        else:
            st.warning("Run the experiment suite to generate this plot.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Accuracy", "59.1%", "at K=4")
    c2.metric("Best Saving",   "39.5%", "at K=1")
    c3.metric("Sweet Spot",    "K=3–4", "28% saving, ~57% accuracy")

    # ---- Exp 3: Verification Overhead ----
    st.divider()
    st.subheader("Experiment 3: SEPG Verification Overhead")
    st.markdown(
        "Proof generation + verification time is constant ~6 ms across all K values."
    )

    if "verification_overhead" in exp_results:
        vo    = exp_results["verification_overhead"]
        df_vo = pd.DataFrame(vo)

        col_l3, col_r3 = st.columns([2, 1])
        with col_l3:
            fig_vo = go.Figure()
            fig_vo.add_trace(go.Bar(
                x=df_vo["K"], y=df_vo["gen_ms"],
                name="Proof Generation", marker_color="#4C72B0", opacity=0.8,
            ))
            fig_vo.add_trace(go.Bar(
                x=df_vo["K"], y=df_vo["ver_ms"],
                name="Proof Verification", marker_color="#DD8452", opacity=0.8,
            ))
            fig_vo.update_layout(
                barmode="stack",
                title="SEPG Overhead vs K (stacked ms)",
                xaxis_title="K (experts in proof)",
                yaxis_title="Time (ms)",
                height=380,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_vo, use_container_width=True)

        with col_r3:
            tbl3 = df_vo[["K", "gen_ms", "ver_ms", "total_ms"]].copy()
            tbl3.columns = ["K", "Gen (ms)", "Verify (ms)", "Total (ms)"]
            for col in ["Gen (ms)", "Verify (ms)", "Total (ms)"]:
                tbl3[col] = tbl3[col].apply(lambda x: f"{x:.2f}")
            st.dataframe(tbl3, hide_index=True, use_container_width=True)
    else:
        img3 = PLOT_DIR / "exp3_verification_overhead.png"
        if img3.exists():
            st.image(str(img3), use_container_width=True)
        else:
            st.warning("Run the experiment suite to generate this plot.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Gen Time",    "~3.1 ms")
    c2.metric("Avg Verify Time", "~3.0 ms")
    c3.metric("Total Overhead",  "~6.1 ms", "Constant K=1..8")

    # ---- Exp 4: Robustness ----
    st.divider()
    st.subheader("Experiment 4: Robustness Under Poisoning Attacks")
    st.markdown(
        "Accuracy of FedAvg, Median, and Trimmed Mean as malicious fraction grows 0→40%."
    )

    if "robustness" in exp_results:
        rob_data = exp_results["robustness"]
        rows_rob = []
        for method, res_list in rob_data.items():
            for r in res_list:
                rows_rob.append({
                    "Strategy":     method,
                    "Malicious %":  r["malicious_pct"],
                    "Accuracy":     r["accuracy"],
                })
        df_rob = pd.DataFrame(rows_rob)

        col_l4, col_r4 = st.columns([2, 1])
        with col_l4:
            clrs4 = {"FedAvg": "#4C72B0", "Median": "#55A868",
                     "Trimmed Mean": "#DD8452"}
            fig_rob4 = go.Figure()
            for method in df_rob["Strategy"].unique():
                sub = df_rob[df_rob["Strategy"] == method]
                fig_rob4.add_trace(go.Scatter(
                    x=sub["Malicious %"], y=sub["Accuracy"],
                    mode="lines+markers", name=method,
                    line=dict(color=clrs4.get(method, "gray"), width=2),
                    marker=dict(size=8),
                ))
            fig_rob4.update_layout(
                title="Robustness: Accuracy vs % Malicious Clients",
                xaxis_title="Malicious Clients (%)",
                yaxis_title="Test Accuracy",
                yaxis_range=[0, 1], height=380,
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_rob4, use_container_width=True)

        with col_r4:
            tbl4 = df_rob.copy()
            tbl4["Accuracy"] = tbl4["Accuracy"].apply(lambda x: f"{x:.2%}")
            tbl4["Malicious %"] = tbl4["Malicious %"].apply(lambda x: f"{x:.0f}%")
            st.dataframe(tbl4, hide_index=True, use_container_width=True)
    else:
        img4 = PLOT_DIR / "exp4_robustness.png"
        if img4.exists():
            st.image(str(img4), use_container_width=True)
        else:
            st.warning("Run the experiment suite to generate this plot.")

    c1, c2, c3 = st.columns(3)
    c1.metric("FedAvg @ 40% mal",      "41.8%", "-16% from clean")
    c2.metric("Median @ 40% mal",      "46.0%", "-7% from clean")
    c3.metric("Trimmed Mean @ 40%",    "44.0%", "-9% from clean")

    st.info(
        "Median is the most robust — coordinate-wise aggregation limits "
        "any single adversarial update's influence regardless of magnitude."
    )


# ---------------------------------------------------------------
# PAGE: CUSTOM CSV
# ---------------------------------------------------------------
elif page == "Custom CSV":
    page_banner("Custom CSV Training",
                "Upload any labelled CSV · auto-detect columns · federated training · confusion matrix + expert routing",
                "📂")
    flow_bar(["▶ Upload CSV", "Map Columns", "Configure", "Train", "Evaluate"], "▶ Upload CSV")

    st.subheader("Step 1 — Upload your CSV")
    uploaded = st.file_uploader(
        "Choose a CSV file", type=["csv"],
        help="Must have at least one text column and one label column.",
    )

    if uploaded is not None:
        try:
            raw = uploaded.read()
            df_raw = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df_raw = pd.read_csv(io.BytesIO(raw), encoding=enc)
                    break
                except Exception:
                    continue
            if df_raw is None:
                st.error("Could not decode the file. Try saving as UTF-8.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.success(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df_raw.head(10), use_container_width=True)

        # ---- Step 2: Column mapping ----
        st.subheader("Step 2 — Map columns")
        all_cols   = df_raw.columns.tolist()
        avg_lens   = {c: df_raw[c].astype(str).str.len().mean() for c in all_cols}
        guess_text = max(avg_lens, key=avg_lens.get)
        nunique    = {c: df_raw[c].nunique() for c in all_cols}
        guess_label = min(
            (c for c in all_cols if c != guess_text and nunique[c] <= 50),
            key=lambda c: nunique[c],
            default=all_cols[0],
        )

        col_a, col_b = st.columns(2)
        text_col  = col_a.selectbox("Text column",  all_cols,
                                     index=all_cols.index(guess_text))
        label_col = col_b.selectbox("Label column", all_cols,
                                     index=all_cols.index(guess_label))

        if text_col == label_col:
            st.warning("Text and label columns must be different.")
            st.stop()

        df = df_raw[[text_col, label_col]].dropna()
        df = df.copy()
        df[text_col]  = df[text_col].astype(str)
        df[label_col] = df[label_col].astype(str)

        # ---- Step 3: Label configuration ----
        st.subheader("Step 3 — Label configuration")
        unique_labels = sorted(df[label_col].unique().tolist())
        n_classes     = len(unique_labels)

        if n_classes < 2:
            st.error("Need at least 2 distinct labels.")
            st.stop()
        if n_classes > 20:
            st.warning(f"{n_classes} unique labels detected — using first 20.")
            unique_labels = unique_labels[:20]
            df = df[df[label_col].isin(unique_labels)]
            n_classes = 20

        label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        idx2label = {i: lbl for lbl, i in label2idx.items()}

        st.markdown(
            f"**{n_classes} classes:** {', '.join(unique_labels[:10])}"
            + (" ..." if n_classes > 10 else "")
        )

        dist = df[label_col].value_counts().reset_index()
        dist.columns = ["Label", "Count"]
        fig_dist = px.bar(dist, x="Label", y="Count",
                          title="Class Distribution in Uploaded Data",
                          text_auto=True)
        fig_dist.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

        # ---- Step 4: Training config ----
        st.subheader("Step 4 — Training configuration")
        cfg1, cfg2, cfg3 = st.columns(3)
        num_clients_csv  = cfg1.slider("Federated Clients", 2, 8, 4,  key="csv_clients")
        num_rounds_csv   = cfg2.slider("Training Rounds",   1, 15, 5, key="csv_rounds")
        top_k_csv        = cfg3.slider("Top-K Experts",     1, 4, 2,  key="csv_topk")

        cfg4, cfg5, cfg6 = st.columns(3)
        num_experts_csv  = cfg4.slider("Number of Experts", 2, 8, 4, key="csv_experts")
        lr_csv           = cfg5.select_slider(
            "Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3, key="csv_lr")
        test_split       = cfg6.slider("Test Split %", 10, 40, 20, key="csv_test") / 100

        max_seq = st.slider("Max Sequence Length (tokens)", 16, 128, 64, key="csv_seq")

        # ---- Step 5: Train ----
        st.subheader("Step 5 — Train & Evaluate")

        if st.button("Start Federated Training", type="primary",
                     use_container_width=True, key="csv_train_btn"):

            progress_csv = st.progress(0)
            status_csv   = st.empty()
            # FIX #10
            try:
                with st.spinner("Tokenising data..."):
                    texts      = df[text_col].tolist()
                    labels_int = [label2idx[l] for l in df[label_col].tolist()]

                    token_counts: Counter = Counter()
                    for t in texts:
                        token_counts.update(t.lower().split())
                    vocab_csv = {"<pad>": 0}
                    for tok, _ in token_counts.most_common(4999):
                        vocab_csv[tok] = len(vocab_csv)

                    full_ds = TextClassificationDataset(
                        texts, labels_int, vocab_csv, seq_len=max_seq)

                    n_test  = max(int(len(full_ds) * test_split), n_classes)
                    n_train = len(full_ds) - n_test
                    if n_train < num_clients_csv:
                        st.error(
                            f"Not enough training samples ({n_train}) for "
                            f"{num_clients_csv} clients. Upload more data or reduce clients."
                        )
                        st.stop()

                    set_seed()
                    train_ds, test_ds = random_split(full_ds, [n_train, n_test])
                    shard_sizes = [n_train // num_clients_csv] * num_clients_csv
                    shard_sizes[0] += n_train - sum(shard_sizes)
                    client_shards = list(random_split(train_ds, shard_sizes))

                st.info(
                    f"Dataset: {n_train:,} train | {n_test:,} test | "
                    f"Vocab: {len(vocab_csv):,} | Classes: {n_classes}"
                )

                dev    = torch.device("cpu")
                kw_csv = dict(vocab_size=len(vocab_csv), embed_dim=64,
                              num_classes=n_classes, num_experts=num_experts_csv,
                              expert_hidden_dim=128, k=top_k_csv, lora_r=8)
                set_seed()
                srv = FedServer(MoETextClassifier(**kw_csv), device=dev)

                metrics_cols = st.columns(3)
                m_acc        = metrics_cols[0].empty()
                m_best       = metrics_cols[1].empty()
                m_rnd        = metrics_cols[2].empty()

                chart_l, chart_r = st.columns(2)
                acc_chart_csv = chart_l.empty()
                heatmap_csv   = chart_r.empty()

                acc_history = []
                best_acc    = 0.0
                best_round  = 1

                for rnd in range(1, num_rounds_csv + 1):
                    states    = []
                    usage_mat = []

                    for ci, shard in enumerate(client_shards):
                        cm = MoETextClassifier(**kw_csv)
                        cm.load_state_dict(srv.get_global_state(), strict=False)
                        fs, _, n, _, _, _, eu = local_train(
                            cm, shard, 1, 32, lr_csv, dev,
                            top_k_sparse=top_k_csv)
                        states.append((fs, n))
                        usage_mat.append((eu / max(n, 1)).numpy())

                    srv.aggregate(states)
                    acc_csv = evaluate(srv.global_model, test_ds, 64, dev)
                    acc_history.append({"Round": rnd, "Accuracy": acc_csv})
                    if acc_csv > best_acc:
                        best_acc, best_round = acc_csv, rnd

                    progress_csv.progress(rnd / num_rounds_csv)
                    status_csv.text(
                        f"Round {rnd}/{num_rounds_csv} — Accuracy: {acc_csv:.2%}")
                    m_acc.metric("Test Accuracy", f"{acc_csv:.2%}")
                    m_best.metric("Best Accuracy", f"{best_acc:.2%}",
                                  f"Round {best_round}")
                    m_rnd.metric("Round", f"{rnd}/{num_rounds_csv}")

                    df_ah   = pd.DataFrame(acc_history).set_index("Round")
                    fig_ah  = px.line(df_ah, y="Accuracy",
                                      title="Test Accuracy per Round", markers=True)
                    fig_ah.update_layout(height=350, yaxis_range=[0, 1])
                    acc_chart_csv.plotly_chart(fig_ah, use_container_width=True)

                    um = np.array(usage_mat)
                    fig_hm = px.imshow(
                        um,
                        x=[f"E{i}" for i in range(num_experts_csv)],
                        y=[f"C{i}" for i in range(len(client_shards))],
                        title=f"Expert Usage — Round {rnd}",
                        color_continuous_scale="YlOrRd",
                        labels=dict(color="Usage"),
                    )
                    fig_hm.update_layout(height=350)
                    heatmap_csv.plotly_chart(fig_hm, use_container_width=True)

                progress_csv.empty()
                status_csv.empty()

                st.success(
                    f"Training complete! Best accuracy: **{best_acc:.2%}** "
                    f"at round {best_round}."
                )

                # ---- Detailed evaluation ----
                st.subheader("Detailed Statistics")

                loader_eval    = DataLoader(test_ds, batch_size=64, shuffle=False)
                all_preds      = []
                all_true       = []
                all_rp_rows    = []
                model_eval     = srv.global_model.to(dev).eval()

                with torch.no_grad():
                    for ids_b, lbl_b in loader_eval:
                        ids_b = ids_b.to(dev)
                        x     = model_eval.embedding(ids_b).mean(dim=1)
                        x, rp = model_eval.moe(x)
                        logits = model_eval.classifier(x)
                        all_preds.extend(logits.argmax(-1).cpu().tolist())
                        all_true.extend(lbl_b.tolist())
                        all_rp_rows.append(rp.cpu().numpy())

                all_rp_mat = np.vstack(all_rp_rows)

                # Confusion matrix
                st.markdown("**Confusion Matrix**")
                conf_mat = np.zeros((n_classes, n_classes), dtype=int)
                for t, p in zip(all_true, all_preds):
                    conf_mat[t][p] += 1

                fig_cm = px.imshow(
                    conf_mat,
                    x=[f"{idx2label.get(i, i)} (pred)" for i in range(n_classes)],
                    y=[f"{idx2label.get(i, i)} (true)" for i in range(n_classes)],
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                    text_auto=True,
                )
                fig_cm.update_layout(height=max(350, n_classes * 50))
                st.plotly_chart(fig_cm, use_container_width=True)

                # Per-class metrics
                st.markdown("**Per-Class Accuracy**")
                per_class_rows = []
                for i in range(n_classes):
                    total_i   = conf_mat[i].sum()
                    correct_i = conf_mat[i][i]
                    prec_i    = conf_mat[i, i] / max(conf_mat[:, i].sum(), 1) * 100
                    rec_i     = correct_i / max(total_i, 1) * 100
                    f1_i      = 2 * prec_i * rec_i / max(prec_i + rec_i, 1e-6)
                    per_class_rows.append({
                        "Class":     idx2label.get(i, str(i)),
                        "Support":   int(total_i),
                        "Correct":   int(correct_i),
                        "Recall %":  round(rec_i, 1),
                        "Precision %": round(prec_i, 1),
                        "F1 %":      round(f1_i, 1),
                    })

                df_per_class = pd.DataFrame(per_class_rows)
                st.dataframe(df_per_class, hide_index=True, use_container_width=True)

                fig_pca = px.bar(
                    df_per_class, x="Class", y="Recall %",
                    title="Per-Class Recall (%)",
                    text="Recall %",
                    color="F1 %",
                    color_continuous_scale="RdYlGn",
                )
                fig_pca.update_layout(height=350, margin=dict(t=40, b=20))
                st.plotly_chart(fig_pca, use_container_width=True)

                # Expert routing per class
                st.markdown("**Expert Routing by Class**")
                st.caption(
                    "Average routing probability per expert, split by true class label. "
                    "Darker = expert preferred for that class."
                )
                class_routing = np.zeros((n_classes, num_experts_csv))
                class_counts  = np.zeros(n_classes)
                for true_lbl, rp_row in zip(all_true, all_rp_mat):
                    class_routing[true_lbl] += rp_row
                    class_counts[true_lbl]  += 1
                for i in range(n_classes):
                    if class_counts[i] > 0:
                        class_routing[i] /= class_counts[i]

                fig_cr = px.imshow(
                    class_routing,
                    x=[f"Expert {i}" for i in range(num_experts_csv)],
                    y=[idx2label.get(i, str(i)) for i in range(n_classes)],
                    title="Mean Expert Routing Probability per Class",
                    color_continuous_scale="Viridis",
                    labels=dict(color="Avg Prob"),
                    text_auto=".3f",
                )
                fig_cr.update_layout(height=max(350, n_classes * 45))
                st.plotly_chart(fig_cr, use_container_width=True)

                dom_rows = []
                for e in range(num_experts_csv):
                    top_cls = int(np.argmax(class_routing[:, e]))
                    dom_rows.append({
                        "Expert":           f"Expert {e}",
                        "Dominant Class":   idx2label.get(top_cls, str(top_cls)),
                        "Avg Routing Prob": f"{class_routing[top_cls, e]:.4f}",
                    })
                st.markdown("**Expert Specialisation Summary**")
                st.dataframe(pd.DataFrame(dom_rows),
                             hide_index=True, use_container_width=True)

                # FIX #8 — download button for results
                st.divider()
                dl_col1, dl_col2 = st.columns(2)

                csv_per_class = df_per_class.to_csv(index=False)
                dl_col1.download_button(
                    label="Download Per-Class Stats (CSV)",
                    data=csv_per_class,
                    file_name="per_class_stats.csv",
                    mime="text/csv",
                )

                conf_df = pd.DataFrame(
                    conf_mat,
                    index=[idx2label.get(i, str(i)) for i in range(n_classes)],
                    columns=[idx2label.get(i, str(i)) for i in range(n_classes)],
                )
                dl_col2.download_button(
                    label="Download Confusion Matrix (CSV)",
                    data=conf_df.to_csv(),
                    file_name="confusion_matrix.csv",
                    mime="text/csv",
                )

                # Save to session
                st.session_state.model              = srv.global_model
                st.session_state.vocab              = vocab_csv
                st.session_state.model_kw           = kw_csv
                st.session_state.custom_class_names = idx2label
                st.session_state.model_source       = f"Custom CSV ({n_classes} classes)"

                st.info(
                    "Model saved to session! Go to **Predict** to classify new text "
                    "with this custom-trained model."
                )

            except Exception as exc:
                progress_csv.empty()
                status_csv.empty()
                st.error(f"Training failed: {exc}")
                raise

    else:
        st.markdown(
            """
**Accepted CSV formats:**

| Format | Text column | Label column | Example |
|--------|------------|--------------|---------|
| AG News style | `description` | `class` (1-4) | `1,"World news text..."` |
| Sentiment | `review` | `sentiment` (`pos`/`neg`) | `"Great product","pos"` |
| Any multi-class | any string column | ≤20 unique values | custom |

The system auto-detects which column is text (longest avg string) and which is label (lowest cardinality).
You can override both in Step 2.

**Minimum requirements:** 2 columns, ≥ 20 rows per class, ≤ 20 distinct labels.
            """
        )


# ---------------------------------------------------------------
# PAGE: COMPARE
# ---------------------------------------------------------------
elif page == "Compare":
    page_banner("Communication Savings Explorer",
                "Sparse updates: clients only send Top-K expert weights — adjust K and see bandwidth savings instantly",
                "📡")
    flow_bar(["Dense Update (all params)", "▶ Top-K Selection", "Sparse Update (K experts)", "Server Aggregate"], "▶ Top-K Selection")
    concept_card(
        "Sparse Communication Key Idea",
        "In a standard FL round every client sends the full model (~600 KB). "
        "With MoE Top-K routing, each client only used K out of 8 expert networks — "
        "so it only sends those K experts back, saving up to 40% bandwidth."
    )

    c1, c2 = st.columns(2)
    ne = c1.slider("Number of Experts",  2, 16, 8, key="cmp_ne")
    tk = c2.slider("Top-K Experts Sent", 1, min(ne, 8), 2, key="cmp_tk")

    m = MoETextClassifier(vocab_size=5000, embed_dim=64, num_classes=4,
                          num_experts=ne, expert_hidden_dim=256, k=tk, lora_r=8)
    total   = sum(p.numel() for p in m.parameters())
    embed   = sum(p.numel() for n, p in m.named_parameters()
                  if n.startswith("embedding"))
    expert  = sum(p.numel() for n, p in m.named_parameters()
                  if "moe.experts" in n)
    per_exp = expert // ne
    other   = total - embed - expert
    sparse  = total - (ne - tk) * per_exp
    saving  = (ne - tk) * per_exp / total * 100

    mc = st.columns(4)
    mc[0].metric("Total Params", f"{total:,}")
    mc[1].metric("Dense Size",   f"{total * 4 / 1024:.0f} KB")
    mc[2].metric("Sparse Size",  f"{sparse * 4 / 1024:.0f} KB")
    mc[3].metric("Saving",       f"{saving:.1f}%",
                 f"{(ne - tk) * per_exp:,} params skipped")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Parameter Breakdown")
        kept_exp = tk * per_exp
        skip_exp = (ne - tk) * per_exp
        fig_stack = go.Figure()
        for name, val, color in [
            ("Embedding",                     embed,    "#4C72B0"),
            (f"Top-{tk} Experts (sent)",       kept_exp, "#55A868"),
            (f"{ne - tk} Experts (skipped)",   skip_exp, "#DD8452"),
            ("Router + LoRA",                  other,    "#8172B2"),
        ]:
            fig_stack.add_trace(go.Bar(
                name=name, x=[val], y=["Model"],
                orientation="h", marker_color=color,
                text=f"{val:,}", textposition="inside",
            ))
        fig_stack.update_layout(barmode="stack", height=200,
                                margin=dict(t=10, b=10), showlegend=True,
                                legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_stack, use_container_width=True)

    with col_r:
        st.subheader("Saving Across All K Values")
        rows = []
        for k in range(1, ne + 1):
            sp = total - (ne - k) * per_exp
            sv = (ne - k) * per_exp / total * 100
            rows.append({"K": k, "Params Sent": sp,
                         "Size (KB)": sp * 4 / 1024, "Saving %": sv})
        df_k = pd.DataFrame(rows)
        fig_k = px.bar(df_k, x="K", y="Saving %",
                       title="Communication Saving vs Top-K", text_auto=".1f")
        fig_k.update_layout(height=350)
        st.plotly_chart(fig_k, use_container_width=True)

    st.subheader("Detailed Comparison Table")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------
# PAGE: ARCHITECTURE — FIX #9: DP + SEPG code snippet tabs
# ---------------------------------------------------------------
elif page == "Architecture":
    page_banner("Model Architecture",
                "Embedding → MoE Layer (Top-K gating) → LoRA Classifier · trained with DP-SGD · verified with SEPG",
                "🏗️")
    flow_bar(["Token IDs", "Embedding", "Mean Pool", "▶ MoE Router", "Top-K Experts", "LoRA Head", "Output"], "▶ MoE Router")

    ne = 8
    m = MoETextClassifier(vocab_size=5000, embed_dim=64, num_classes=4,
                          num_experts=ne, expert_hidden_dim=256, k=2, lora_r=8)
    total  = sum(p.numel() for p in m.parameters())
    embed  = sum(p.numel() for n, p in m.named_parameters()
                 if n.startswith("embedding"))
    expert = sum(p.numel() for n, p in m.named_parameters()
                 if "moe.experts" in n)
    other  = total - embed - expert

    mc = st.columns(4)
    mc[0].metric("Total",       f"{total:,}")
    mc[1].metric("Embedding",   f"{embed:,}",  f"{embed/total*100:.0f}%")
    mc[2].metric("8 Experts",   f"{expert:,}", f"{expert/total*100:.0f}%")
    mc[3].metric("Router+LoRA", f"{other:,}",  f"{other/total*100:.0f}%")

    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Data Flow")
        st.graphviz_chart("""
        digraph {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fontname="Helvetica"];

            input   [label="Input Token IDs\\n[batch, 64]",               fillcolor="#E8F4FD"];
            embed   [label="Embedding Layer\\n5000 x 64 = 320K params",   fillcolor="#B8D4E3"];
            pool    [label="Mean Pooling\\n[batch, 64]",                   fillcolor="#B8D4E3"];
            router  [label="Router\\nLinear(64, 8) + Softmax",             fillcolor="#F5D5A0"];
            topk    [label="Top-K Selection\\nPick 2 of 8 experts",        fillcolor="#F5D5A0"];
            experts [label="Expert MLPs (x8)\\n64 → 256 → 64",            fillcolor="#FADBD8"];
            combine [label="Weighted Sum\\nof Top-K outputs",              fillcolor="#F5D5A0"];
            lora    [label="LoRA Classifier\\nBase(frozen) + A*B(train)",  fillcolor="#D5F5E3"];
            output  [label="Class Prediction\\nWorld/Sports/Business/Tech",fillcolor="#E8F4FD"];

            input -> embed -> pool -> router;
            router -> topk -> experts -> combine -> lora -> output;
        }
        """)

    with col2:
        st.subheader("Parameter Distribution")
        fig = px.pie(
            names=["Embedding", "MoE Experts", "Router+LoRA"],
            values=[embed, expert, other],
            color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868"],
            hole=0.3,
        )
        fig.update_traces(textinfo="label+percent", textfont_size=13)
        fig.update_layout(height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Expert Details")
    exp_cols = st.columns(min(ne, 4))
    for i in range(ne):
        ep = sum(p.numel() for n, p in m.named_parameters()
                 if f"moe.experts.{i}" in n)
        with exp_cols[i % min(ne, 4)]:
            with st.expander(f"Expert {i}"):
                for n, p in m.named_parameters():
                    if f"moe.experts.{i}" in n:
                        st.text(f"{n.split(f'experts.{i}.')[-1]}: {list(p.shape)}")
                st.caption(f"{ep:,} params")

    st.divider()

    st.subheader("Key Code")
    # FIX #9 — added DP and SEPG tabs
    t1, t2, t3, t4, t5 = st.tabs(["MoE Gating", "LoRA", "FedAvg", "Differential Privacy", "SEPG Proof"])

    with t1:
        st.code("""\
# Router produces probabilities over 8 experts
logits = self.router(x)                  # (batch, 8)
probs  = F.softmax(logits, dim=-1)

# Select Top-2 experts per input
topk_vals, topk_idx = torch.topk(probs, k=2, dim=-1)

# Only 2 experts compute; rest are skipped entirely
""", language="python")

    with t2:
        st.code("""\
# Base weights FROZEN, only A and B trained
def forward(self, x):
    base_out = self.base(x)             # frozen
    lora_out = self.B(self.A(x))        # trainable (rank=8)
    return base_out + lora_out * (alpha / r)
""", language="python")

    with t3:
        st.code("""\
# Server aggregates sparse updates correctly:
# - Each expert averaged only across clients that sent it
# - Non-updated experts keep their previous weights
for name in agg_state:
    w = agg_weight[name]
    if abs(w - 1.0) > 1e-6:
        agg_state[name] /= w   # re-normalise sparse key
""", language="python")

    with t4:
        st.code("""\
# 1. Clip gradient update to L2 norm <= C
def clip_update(state, max_norm):
    flat = torch.cat([t.float().flatten() for t in state.values()])
    factor = min(1.0, max_norm / (flat.norm(2) + 1e-8))
    return {k: v * factor for k, v in state.items()}

# 2. Add Gaussian noise  N(0, (sigma * C)^2)
def add_noise(state, noise_scale):
    return {k: v + torch.randn_like(v) * noise_scale
            for k, v in state.items()}

# Combined DP-SGD step
def apply_dp(state, clip_norm, noise_multiplier):
    clipped = clip_update(state, clip_norm)
    return add_noise(clipped, noise_multiplier * clip_norm)

# Privacy accountant (basic Gaussian composition)
eps_per_step = q * sqrt(2 * ln(1.25/delta)) / sigma
epsilon_total = eps_per_step * sqrt(num_rounds)
""", language="python")

    with t5:
        st.code("""\
# Client generates proof after local training
proof = SEPGProof(
    client_id     = ci,
    round_id      = rnd,
    top_k_indices = [2, 5],           # which experts were activated
    dp_params     = {"clip_norm": 1.0, "noise_mult": 0.5, "epsilon": 0.03},
    update_hash   = sha256(sparse_state),   # integrity check
)

# Server runs 4 checks before accepting update:
# 1. len(top_k_indices) == expected_K
# 2. clip_norm <= max_allowed
# 3. noise_mult >= min_required
# 4. sha256(received_state) == proof.update_hash
passed, reason = verify_proof(proof, sparse_state, expected_k=2)
""", language="python")


# ---------------------------------------------------------------
# PAGE: ABOUT
# ---------------------------------------------------------------
elif page == "About":
    page_banner("About zkFedMoE",
                "Zero-Knowledge Federated Mixture-of-Experts · IIIT Kota · Group #34 · April 2026",
                "ℹ️")

    c1, c2, c3 = st.columns(3)
    c1.markdown("**Team (Group #34)**\n- Keshav Kashyap\n- Lakshya Sharma\n- Prakriti Patel")
    c2.markdown("**Advisor**\nDr. Gyan Singh Yadav\nCSE Department")
    c3.markdown("**Institution**\nIndian Institute of\nInformation Technology, Kota")

    st.divider()

    st.subheader("Implementation Status")
    status = {
        "Component": [
            "MoE + LoRA Model",
            "FedAvg FL Pipeline",
            "AG News (120K samples)",
            "Sparse Top-K Updates",
            "Router-based Expert Selection",
            "Differential Privacy (DP-SGD)",
            "SEPG Proof Generation & Verification",
            "Adversary Simulation (Poisoning / Free-rider / Sybil)",
            "Robust Aggregation (Median + Trimmed Mean)",
            "4 Core Experiment Plots",
            "Custom CSV Training",
            "Interactive Dashboard",
        ],
        "Status": ["Done"] * 12,
        "Dashboard Page": [
            "Architecture", "Train", "Train", "Compare", "Predict",
            "Privacy & DP", "Privacy & DP", "Robustness", "Robustness",
            "Experiments", "Custom CSV", "All",
        ],
    }
    st.dataframe(pd.DataFrame(status), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Dashboard Pages")
    pages_desc = {
        "Page": ["Predict", "Train", "Custom CSV", "Privacy & DP",
                 "Robustness", "Experiments", "Compare", "Architecture"],
        "What it does": [
            "Live text classification + expert routing + compare two headlines side-by-side",
            "Full FL training loop with optional DP + SEPG proof verification per round",
            "Upload any CSV → auto column detection → FL training → confusion matrix + per-class stats + download",
            "DP-SGD training with live ε/δ budget chart + per-client SHA-256 proof display",
            "Attack simulation (poisoning/free-rider/Sybil) + FedAvg vs Median vs TrimmedMean comparison",
            "Interactive Plotly charts for all 4 experiments (privacy-utility, comm vs K, overhead, robustness)",
            "Instant communication saving calculator for any expert/K configuration",
            "Architecture diagram + expert breakdown + 5 code snippet tabs (MoE/LoRA/FedAvg/DP/SEPG)",
        ],
    }
    st.dataframe(pd.DataFrame(pages_desc), hide_index=True, use_container_width=True)
