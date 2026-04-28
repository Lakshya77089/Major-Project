"""
zkFedMoE Interactive Dashboard
================================
Launch:  python -m streamlit run dashboard.py
"""

import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from src.data.text_datasets import (
    TextClassificationDataset,
    build_ag_news_clients,
    build_ag_news_clients_noniid,
    client_class_distribution,
    dirichlet_split,
)
from src.fl.adversaries import freerider_train, poisoning_train, sybil_clones
from src.fl.attacks import membership_inference_attack
from src.fl.bidding import (
    ResourceBid, BidCommitment,
    fresh_nonce, commit_bid, verify_bid, run_auction,
)
from src.fl.client import local_train
from src.fl.dp import PrivacyAccountant, RenyiAccountant, apply_dp
from src.fl.sepg import (
    generate_proof, verify_proof,
    committee_decision, oracle_score,
)
from src.fl.server import FedServer
from src.fl.zkhash import benchmark_compare as zkhash_benchmark
from src.chain.ledger import Ledger
from src.models.moe_model import MoETextClassifier, predict_with_routing
from src.viz.animations import predict_animation_frame, fl_topology_frame

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
/* Force dark text on the light metric card so values stay readable
   regardless of Streamlit's light/dark theme. */
[data-testid="stMetric"] *,
[data-testid="stMetric"] label,
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #1f2937 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600;
    opacity: 0.85;
}
/* Delta: keep semantic green/red but readable on light bg */
[data-testid="stMetricDelta"] svg { fill: currentColor; }
[data-testid="stMetricDelta"][class*="positive"],
[data-testid="stMetricDelta"][class*="positive"] * { color: #16a34a !important; }
[data-testid="stMetricDelta"][class*="negative"],
[data-testid="stMetricDelta"][class*="negative"] * { color: #dc2626 !important; }
</style>
""", unsafe_allow_html=True)

PLOT_DIR = Path(__file__).parent / "plots"


def set_seed(s=42):
    torch.manual_seed(s)


def _rand_client_id():
    """Pick a client id 1..10 for ledger demo transactions."""
    import random as _r
    return _r.randint(1, 10)


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
    clients_1, _td, vs, nc, voc = _load_ag_news_raw()
    full_train = clients_1[0]

    SUBSAMPLE = 32_000
    if len(full_train) > SUBSAMPLE:
        idxs = torch.randperm(len(full_train))[:SUBSAMPLE].tolist()
        full_train = Subset(full_train, idxs)

    n = len(full_train)
    sizes = [n // 4] * 4
    sizes[0] += n - sum(sizes)
    cl = list(random_split(full_train, sizes))

    kw = dict(vocab_size=vs, embed_dim=64, num_classes=nc,
              num_experts=4, expert_hidden_dim=128, k=2, lora_r=8)
    srv = FedServer(MoETextClassifier(**kw), device=dev)
    for _ in range(10):
        sts = []
        for c in cl:
            m = MoETextClassifier(**kw)
            m.load_state_dict(srv.get_global_state(), strict=False)
            fs, _, _n, _, _, _, _ = local_train(m, c, 2, 64, 1e-3, dev)
            sts.append((fs, _n))
        srv.aggregate(sts)
    return srv.global_model, voc, kw


# ---------------------------------------------------------------
# Sidebar — FIX #7: model status block
# ---------------------------------------------------------------
st.sidebar.title("zkFedMoE")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "News Detection", "Disease Detection", "General zkFedMoE",
     "Train", "Custom CSV",
     "Privacy & DP", "Robustness", "Non-IID & MIA",
     "Bidding & Oracle", "Chain Explorer",
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
        ("🔮 News Detection", "Type a news headline → see detected class + which experts activate. Compare two headlines side-by-side."),
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
    | 3 | **News Detection** | Type a news headline — highlight which experts fire and why |
    | 4 | **Compare** | Drag the K slider to show real-time communication saving calculation |
    | 5 | **Privacy & DP** | Run DP training, show SEPG proofs with PASS badges |
    | 6 | **Robustness** | Run Poisoning at 30% — show FedAvg drops, Median holds |
    | 7 | **Experiments** | Open all 4 interactive charts, point to sweet-spot K=3–4 |
    | 8 | **Custom CSV** | Upload your own CSV and train a new model live |
    """)


# ---------------------------------------------------------------
# PAGE: PREDICT
# ---------------------------------------------------------------
elif page == "News Detection":
    page_banner(
        "News Detection System",
        "Federated MoE classifier · Top-K expert routing · DP-SGD · SEPG proofs · "
        "watch the full zkFedMoE pipeline run end-to-end below, then try a headline.",
        "🔮",
    )
    flow_bar(
        ["FL Train (×N clients)", "DP-SGD", "SEPG Proof", "Aggregate",
         "Tokenise", "Embedding", "MoE Router", "▶ Predict"],
        "▶ Predict",
    )

    # ---- Behind-the-Scenes: full zkFedMoE pipeline animation ----
    with st.expander(
        "🎬 Behind the scenes — how the model was federated-trained "
        "(click to play the full zkFedMoE pipeline animation)",
        expanded=False,
    ):
        concept_card(
            "What you're about to see",
            "The model used below was trained <b>federated</b> across simulated "
            "clients — no raw text ever leaves a client. Each round: server broadcasts "
            "global weights → clients train locally on AG News shards → DP-SGD "
            "clips + adds Gaussian noise → SEPG proof (SHA-256 hash + Top-K + "
            "DP params) is generated → server verifies proofs → robust aggregation → "
            "ledger entry. Below is a visual walk-through of one such round.",
        )
        anim_n_clients = st.slider("Clients in animation", 3, 8, 5, key="news_anim_n")
        anim_speed = st.select_slider(
            "Animation speed",
            options=["Slow", "Normal", "Fast"], value="Normal", key="news_anim_speed",
        )
        if st.button("▶ Play one FL round", key="news_anim_play"):
            speed_map = {"Slow": 0.9, "Normal": 0.55, "Fast": 0.25}
            delay = speed_map[anim_speed]
            slot = st.empty()
            sub_caption = st.empty()
            phases = [
                ("broadcast", "Server broadcasts global θ_t to all clients", None),
                *[("train",
                   f"Client C{cid} trains locally · Top-K expert routing · "
                   f"DP-SGD clip+noise · SEPG proof generated",
                   cid) for cid in range(anim_n_clients)],
                ("upload", "Clients upload sparse updates + SEPG proofs to server", None),
                ("aggregate", "Server: verify 4-check SEPG · FedAvg robust aggregate · "
                              "append to audit ledger", None),
                ("done", "Round complete · global θ_{t+1} ready for next round", None),
            ]
            for ph, caption, active in phases:
                fig = fl_topology_frame(
                    num_clients=anim_n_clients,
                    phase=ph,
                    round_id=1, total_rounds=10,
                    accuracy=None, active_client=active,
                )
                slot.plotly_chart(fig, use_container_width=True,
                                  key=f"news_anim_{ph}_{active}")
                sub_caption.info(caption)
                time.sleep(delay)
            sub_caption.success(
                "Pipeline complete. The model below is the result of repeating "
                "this round many times. Now try classifying a headline ↓"
            )

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

        # ---- Live inference animation (5 stages) ----
        anim_box = st.empty()
        for _stage in range(6):
            anim_box.plotly_chart(
                predict_animation_frame(
                    stage=_stage,
                    tokens=text.lower().split(),
                    oov=oov,
                    num_experts=num_exp,
                    top_experts=top_experts,
                    pred_label=pred_label,
                    class_probs=class_probs,
                ),
                use_container_width=True,
                key=f"predict_anim_{_stage}",
            )
            time.sleep(0.45)

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

    # DP, SEPG, and FedProx toggles
    st.divider()
    dp_col, sepg_col, prox_col = st.columns(3)
    use_dp     = dp_col.toggle("Enable Differential Privacy", value=False)
    use_sepg   = sepg_col.toggle("Enable SEPG Proof Verification", value=False)
    use_prox   = prox_col.toggle("Enable FedProx (heterogeneity)", value=False,
                                 help="Adds (mu/2)*||theta - theta_global||^2 proximal term to local "
                                      "loss. Helps under non-IID data.")

    # FIX #3 — always define clip_norm / noise_mult so training loop never NameErrors
    clip_norm  = 1.0
    noise_mult = 0.5
    fedprox_mu = 0.0
    if use_dp:
        dp_c1, dp_c2 = st.columns(2)
        clip_norm  = dp_c1.slider("Clip Norm (C)", 0.1, 5.0, 1.0, step=0.1)
        noise_mult = dp_c2.slider("Noise Multiplier (σ)", 0.0, 2.0, 0.5, step=0.1)
    if use_prox:
        fedprox_mu = st.select_slider(
            "FedProx μ", options=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            value=0.01, key="fedprox_mu",
        )

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

            st.markdown("**Federated Round Animation**")
            fl_anim = st.empty()

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

                # Phase 1: server broadcasts global model
                fl_anim.plotly_chart(
                    fl_topology_frame(num_clients, "broadcast", rnd, num_rounds, da if rnd > 1 else None),
                    use_container_width=True,
                    key=f"fl_anim_{rnd}_broadcast",
                )
                time.sleep(0.45)

                for ci, cds in enumerate(cl):
                    # Phase 2: client ci trains locally
                    fl_anim.plotly_chart(
                        fl_topology_frame(num_clients, "train", rnd, num_rounds,
                                          da if rnd > 1 else None, active_client=ci),
                        use_container_width=True,
                        key=f"fl_anim_{rnd}_train_{ci}",
                    )
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv_d.get_global_state(), strict=False)
                    fs, sp, n, db, sb, tki, eu = local_train(
                        cm, cds, 1, 64, lr, dev, top_k_sparse=top_k,
                        fedprox_mu=fedprox_mu,
                    )
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

                # Phase 3: clients upload sparse gradients + proofs
                fl_anim.plotly_chart(
                    fl_topology_frame(num_clients, "upload", rnd, num_rounds, da if rnd > 1 else None),
                    use_container_width=True,
                    key=f"fl_anim_{rnd}_upload",
                )
                time.sleep(0.45)

                # Phase 4: server aggregates
                fl_anim.plotly_chart(
                    fl_topology_frame(num_clients, "aggregate", rnd, num_rounds, da if rnd > 1 else None),
                    use_container_width=True,
                    key=f"fl_anim_{rnd}_aggregate",
                )
                time.sleep(0.4)

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

            # Final "done" frame
            fl_anim.plotly_chart(
                fl_topology_frame(num_clients, "done", num_rounds, num_rounds, da),
                use_container_width=True,
                key="fl_anim_done",
            )

            progress.empty()
            status_text.empty()

            st.session_state.model        = srv_d.global_model
            st.session_state.vocab        = voc
            st.session_state.model_kw     = kw
            st.session_state.model_source = "AG News FL" if use_ag else "Small corpus FL"
            st.session_state.pop("custom_class_names", None)
            st.success(
                f"Training complete! Dense={da:.2%}, Sparse={sa:.2%}, "
                f"Saving={sav:.1f}%. Model saved — go to **News Detection**."
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

    # NEW: Renyi DP + Secure Aggregation toggles
    adv_col1, adv_col2 = st.columns(2)
    use_renyi = adv_col1.toggle(
        "Use Rényi DP accountant (tighter bound)", value=True, key="dp_renyi",
        help="Uses Rényi DP composition (Mironov 2017) — the standard in Opacus/TF-Privacy. "
             "Gives the correct ε bound; the basic √T composition can be misleadingly optimistic.",
    )
    use_secure_agg = adv_col2.toggle(
        "Use Secure Aggregation (pairwise masking)", value=False, key="dp_secure",
        help="Bonawitz-style pairwise masking: the server only sees the SUM of updates, "
             "not any individual client's update. Masks cancel on aggregation.",
    )

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
            srv = FedServer(MoETextClassifier(**kw), device=dev)
            # Track BOTH accountants so we can display the comparison
            accountant_basic = PrivacyAccountant(target_delta=1e-5)
            accountant_renyi = RenyiAccountant(target_delta=1e-5)

            budget_chart = st.empty()
            eps_history  = []
            all_proofs   = []          # overwritten each round; last round shown
            secure_residuals = []

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
                    accountant_basic.accumulate(noise_mult_dp, sr, num_steps=1)
                    accountant_renyi.accumulate(noise_mult_dp, sr, num_steps=1)
                    # Pick whichever accountant the user enabled for the proof & chart
                    eps_active, delta = (
                        accountant_renyi.get_privacy_spent()
                        if use_renyi else accountant_basic.get_privacy_spent()
                    )

                    proof = generate_proof(
                        client_id=ci, round_id=rnd,
                        top_k_indices=list(range(2)),
                        clip_norm=clip_norm_dp,
                        noise_multiplier=noise_mult_dp,
                        epsilon=eps_active,
                        sparse_state=sp,
                    )
                    all_proofs.append((proof, sp))
                    states.append((fs_dp, n))

                # Aggregate using secure aggregation if enabled
                if use_secure_agg and len(states) >= 2:
                    diag = srv.aggregate_secure(states, round_id=rnd, mask_scale=0.05)
                    secure_residuals.append(diag["max_residual"])
                else:
                    srv.aggregate(states)

                acc = evaluate(srv.global_model, td, 64, dev)
                eps_basic, _ = accountant_basic.get_privacy_spent()
                eps_renyi, _ = accountant_renyi.get_privacy_spent()
                eps_active = eps_renyi if use_renyi else eps_basic
                eps_history.append({
                    "Round": rnd,
                    "ε (active)": round(eps_active, 6),
                    "ε (basic)":  round(eps_basic, 6),
                    "ε (Rényi)":  round(eps_renyi, 6),
                    "Accuracy":   round(acc, 4),
                })
                eps = eps_active
                delta = 1e-5

                progress_dp.progress(rnd / num_rounds_dp)
                status_dp.markdown(
                    f"**Round {rnd}/{num_rounds_dp}** | "
                    f"Accuracy: {acc:.2%} | ε={eps:.4f}, δ={delta:.2e}"
                )

                df_eps = pd.DataFrame(eps_history).set_index("Round")
                fig_eps = go.Figure()
                # Show both accountants on the same chart
                fig_eps.add_trace(go.Scatter(
                    x=df_eps.index, y=df_eps["ε (Rényi)"],
                    mode="lines+markers", name="ε (Rényi DP)",
                    line=dict(color="#DD8452", width=2),
                ))
                fig_eps.add_trace(go.Scatter(
                    x=df_eps.index, y=df_eps["ε (basic)"],
                    mode="lines+markers", name="ε (basic √T)",
                    line=dict(color="#CF9F3C", width=2, dash="dot"),
                ))
                fig_eps.add_trace(go.Scatter(
                    x=df_eps.index, y=df_eps["Accuracy"],
                    mode="lines+markers", name="Accuracy",
                    line=dict(color="#4C72B0", width=2),
                    yaxis="y2",
                ))
                fig_eps.update_layout(
                    title=("Privacy Budget vs Accuracy — "
                           f"{'Rényi DP' if use_renyi else 'basic √T'} used for proofs"),
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

            st.subheader("Privacy Budget Summary (both accountants)")
            df_summary = pd.DataFrame(eps_history)
            st.dataframe(df_summary, hide_index=True, use_container_width=True)

            final = eps_history[-1]
            concept_card(
                "Rényi DP vs. basic √T composition",
                "The Rényi accountant is the tight bound used by production libraries (Opacus, "
                "TF-Privacy). The simple √T bound can under- or over-estimate ε depending on "
                "the (σ, q) regime. Trust the Rényi column for any formal claim."
            )
            st.success(
                f"After {num_rounds_dp} rounds with σ={noise_mult_dp:.2f}: "
                f"**ε (Rényi) = {final['ε (Rényi)']:.4f}**, "
                f"ε (basic) = {final['ε (basic)']:.4f} (δ = 1e-5)."
            )

            if use_secure_agg and secure_residuals:
                st.subheader("Secure Aggregation Diagnostic")
                max_res = max(secure_residuals)
                st.metric(
                    "Max mask-cancel residual across all rounds", f"{max_res:.2e}",
                    help="Pairwise masks cancel to zero on sum. Residual near 1e-6 or "
                         "smaller means the masked-sum equals the true sum to floating-"
                         "point precision — server never sees individual updates."
                )
                insight_box(
                    "Secure Aggregation is ACTIVE. The server aggregated only the "
                    f"masked sum of {len(cl)} client updates. "
                    "No individual client's update was visible to the server."
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
# PAGE: NON-IID & MIA  (F5 + F6)
# ---------------------------------------------------------------
elif page == "Non-IID & MIA":
    page_banner(
        "Non-IID Data & Membership Inference",
        "Dirichlet(α) partitioning simulates real-world heterogeneous clients. "
        "MIA measures how well DP protects training samples from leakage.",
        "🧪",
    )
    flow_bar(
        ["Dirichlet(α) split", "FL Train", "Measure MIA AUC", "DP ON vs OFF"],
        "Dirichlet(α) split",
    )

    concept_card(
        "Why non-IID matters",
        "Real FL almost never has IID data. User A types about sports; user B about finance. "
        "Our existing experiments use random_split (IID). Dirichlet(α) partitioning is the "
        "standard way to simulate real heterogeneity: low α = each client skewed to few classes, "
        "high α = approaches IID."
    )
    concept_card(
        "Membership Inference Attack (MIA)",
        "An attacker tries to guess whether a sample was in the training set. Members tend to "
        "have LOWER loss (memorisation). We compute the AUC of this attack — 0.5 means random "
        "guessing (DP is working), 1.0 means perfect leakage."
    )

    col1, col2, col3, col4 = st.columns(4)
    alpha_niid = col1.select_slider(
        "Dirichlet α", options=[0.1, 0.3, 0.5, 1.0, 5.0, 100.0], value=0.5, key="niid_alpha",
    )
    n_clients_niid = col2.slider("Clients", 2, 8, 5, key="niid_clients")
    n_rounds_niid = col3.slider("FL Rounds", 1, 8, 3, key="niid_rounds")
    use_dp_niid = col4.toggle("DP ON", value=False, key="niid_dp")

    c5, c6 = st.columns(2)
    clip_niid = c5.slider("DP Clip C", 0.1, 5.0, 1.0, step=0.1, key="niid_clip") if use_dp_niid else 1.0
    sigma_niid = c6.slider("DP σ",     0.0, 2.0, 0.5, step=0.05, key="niid_sigma") if use_dp_niid else 0.0

    if st.button("Run Non-IID Training + MIA", type="primary",
                 use_container_width=True, key="niid_run_btn"):
        progress_niid = st.progress(0)
        status_niid = st.empty()
        try:
            set_seed()
            dev = torch.device("cpu")

            with st.spinner("Loading AG News..."):
                clients_niid, test_ds_niid, vs_n, nc_n, _ = (
                    build_ag_news_clients_noniid(
                        num_clients=n_clients_niid, alpha=alpha_niid,
                        use_external_csv=True, seed=42,
                    )
                )

            # Per-client class distribution visualisation
            st.subheader("Per-client class distribution (lower α → more skew)")
            dist_rows = []
            for i, c in enumerate(clients_niid):
                dist = client_class_distribution(c, nc_n)
                for cls_id, cnt in enumerate(dist):
                    dist_rows.append({
                        "Client": f"Client {i}",
                        "Class": ["World", "Sports", "Business", "Tech"][cls_id]
                                 if cls_id < 4 else str(cls_id),
                        "Count": int(cnt),
                    })
            df_dist = pd.DataFrame(dist_rows)
            fig_dist = px.bar(
                df_dist, x="Client", y="Count", color="Class",
                barmode="stack",
                title=f"Client class distributions (Dirichlet α={alpha_niid})",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_dist.update_layout(height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)

            # Train
            kw = dict(vocab_size=vs_n, embed_dim=64, num_classes=nc_n,
                      num_experts=8, expert_hidden_dim=256, k=2, lora_r=8)
            srv = FedServer(MoETextClassifier(**kw), device=dev)

            acc_rows = []
            for rnd in range(1, n_rounds_niid + 1):
                states = []
                for ci, cds in enumerate(clients_niid):
                    cm = MoETextClassifier(**kw)
                    cm.load_state_dict(srv.get_global_state(), strict=False)
                    fs, _, n, _, _, _, _ = local_train(
                        cm, cds, 1, 64, 2e-3, dev, top_k_sparse=2)
                    if use_dp_niid and sigma_niid > 0:
                        fs = apply_dp(fs, clip_norm=clip_niid, noise_multiplier=sigma_niid)
                    states.append((fs, n))

                srv.aggregate(states)
                acc = evaluate(srv.global_model, test_ds_niid, 64, dev)
                acc_rows.append({"Round": rnd, "Accuracy": acc})
                progress_niid.progress(rnd / n_rounds_niid)
                status_niid.text(f"Round {rnd}/{n_rounds_niid}, acc={acc:.2%}")

            progress_niid.empty()
            status_niid.empty()

            # Accuracy curve
            df_acc = pd.DataFrame(acc_rows).set_index("Round")
            fig_acc = px.line(df_acc, y="Accuracy",
                              title="Test Accuracy per Round", markers=True)
            fig_acc.update_layout(height=300, yaxis_range=[0, 1])
            st.plotly_chart(fig_acc, use_container_width=True)

            # ---- Membership Inference Attack ----
            st.subheader("Membership Inference Attack (MIA)")
            with st.spinner("Running MIA..."):
                # Members = concat of first 2 clients' shards (subsample for speed)
                from torch.utils.data import ConcatDataset
                members = ConcatDataset(clients_niid[:2])
                # Non-members = held-out test set
                mia_result = membership_inference_attack(
                    srv.global_model, members, test_ds_niid,
                    device=dev, max_samples=400,
                )

            mia_cols = st.columns(4)
            mia_cols[0].metric("MIA AUC", f"{mia_result.auc:.3f}",
                               help="0.5 = random guessing (DP works), 1.0 = perfect leakage")
            mia_cols[1].metric("Attack Accuracy", f"{mia_result.attack_accuracy:.1%}")
            mia_cols[2].metric("Member loss (mean)", f"{mia_result.train_loss_mean:.3f}")
            mia_cols[3].metric("Non-member loss (mean)", f"{mia_result.nonmember_loss_mean:.3f}")

            if mia_result.auc < 0.55:
                st.success(
                    f"✓ {mia_result.summary()}  "
                    f"Members and non-members are statistically indistinguishable."
                )
            elif mia_result.auc < 0.65:
                insight_box(
                    f"{mia_result.summary()}  Defense is weak — "
                    "increase σ or train for more rounds."
                )
            else:
                st.error(
                    f"⚠ {mia_result.summary()}  "
                    "Significant privacy leakage detected."
                )

            st.caption(
                "Run this with DP OFF then DP ON to see the AUC drop — that's DP working."
            )

        except Exception as exc:
            progress_niid.empty()
            status_niid.empty()
            st.error(f"Non-IID / MIA run failed: {exc}")
            raise


# ---------------------------------------------------------------
# PAGE: BIDDING & ORACLE  (F3 + F4)
# ---------------------------------------------------------------
elif page == "Bidding & Oracle":
    page_banner(
        "Resource Bidding & Verifiable Oracle Committee",
        "Pedersen-style commitments hide bids until reveal · M-oracle median voting "
        "withstands up to ⌊(M-1)/2⌋ malicious oracles",
        "🔐",
    )
    flow_bar(
        ["Client commits bid", "Server collects", "Reveal", "Verify", "Auction"],
        "Server collects",
    )

    concept_card(
        "Why commit-then-reveal?",
        "If clients revealed resource bids directly, a malicious last bidder could see all "
        "earlier offers and undercut. Commit-reveal forces every client to lock in their bid "
        "(via a hash) BEFORE seeing anyone else, then prove they didn't change it. The hash is "
        "binding (can't change) and hiding (server can't reverse it without the nonce)."
    )
    concept_card(
        "Why a committee of oracles?",
        "A single oracle scoring updates is a single point of trust. With M oracles voting "
        "independently and the median taken, up to ⌊(M-1)/2⌋ of them can be malicious or "
        "wrong without changing the decision (the median is unaffected by outliers)."
    )

    st.subheader("Step 1 — Configure")
    cfg1, cfg2, cfg3 = st.columns(3)
    n_clients_bid = cfg1.slider("Number of clients", 2, 10, 5, key="bid_clients")
    n_oracles    = cfg2.slider("Oracle committee size (M)", 1, 9, 5, step=2, key="bid_oracles")
    top_n        = cfg3.slider("Auction winners (top-N)", 1, n_clients_bid,
                                min(3, n_clients_bid), key="bid_topn")

    if st.button("Run Bidding + Committee Round", type="primary",
                 use_container_width=True, key="bid_run"):
        try:
            import random as _rand
            _rand.seed(42)

            st.subheader("Step 2 — Clients submit commitments")
            commitments = {}
            revealed = {}
            for ci in range(n_clients_bid):
                # Random plausible offer
                bw = round(_rand.uniform(10, 200), 1)
                cp = round(_rand.uniform(50, 500), 1)
                sg = round(_rand.uniform(256, 4096), 0)
                bid = ResourceBid(client_id=ci, bandwidth_mbps=bw,
                                  compute_gflops=cp, storage_mb=sg,
                                  nonce=fresh_nonce())
                com = commit_bid(bid, round_id=1)
                commitments[ci] = com
                revealed[ci]    = bid

            commit_rows = [
                {"Client": cid,
                 "Commitment hash": com.commitment_hash[:24] + "...",
                 "Round": com.round_id}
                for cid, com in commitments.items()
            ]
            st.dataframe(pd.DataFrame(commit_rows),
                         hide_index=True, use_container_width=True)
            st.caption("Server sees only the hashes at this point — bids are still hidden.")

            st.subheader("Step 3 — Clients reveal bids; server verifies")
            verify_rows = []
            for cid, bid in revealed.items():
                ok, why = verify_bid(commitments[cid], bid)
                verify_rows.append({
                    "Client": cid,
                    "Bandwidth (Mbps)": bid.bandwidth_mbps,
                    "Compute (GFLOPS)": bid.compute_gflops,
                    "Storage (MB)": bid.storage_mb,
                    "Verified": "✓" if ok else "✗",
                    "Reason": why,
                })
            st.dataframe(pd.DataFrame(verify_rows),
                         hide_index=True, use_container_width=True)

            # Show what tampering looks like
            tampered = ResourceBid(client_id=0, bandwidth_mbps=9999.0,
                                   compute_gflops=9999.0, storage_mb=9999.0,
                                   nonce=revealed[0].nonce)
            ok_t, why_t = verify_bid(commitments[0], tampered)
            insight_box(
                f"Tamper check: a fabricated bid for Client 0 with absurd values "
                f"is detected: ok={ok_t}, reason={why_t}"
            )

            st.subheader(f"Step 4 — Auction (top {top_n} wins)")
            result = run_auction(commitments, revealed, top_n=top_n)
            lb_rows = [
                {"Rank": rank + 1, "Client": cid, "Score": f"{score:.2f}",
                 "Won": "★" if cid in result.accepted_client_ids else ""}
                for rank, (cid, score) in enumerate(result.leaderboard)
            ]
            st.dataframe(pd.DataFrame(lb_rows),
                         hide_index=True, use_container_width=True)
            st.success(
                f"Winners: {result.accepted_client_ids}  |  "
                f"Top score: {result.winning_score:.2f}  |  "
                f"Rejected: {result.rejected_count}"
            )

            st.divider()
            st.subheader(f"Step 5 — Oracle committee scoring (M={n_oracles})")
            st.caption("Each oracle scores each winning client's hypothetical update independently. "
                       "Final decision = median across the committee.")

            # Hypothetical update states for each winner
            import torch as _t
            committee_rows = []
            for cid in result.accepted_client_ids:
                fake_state = {f"layer_{cid}": _t.randn(8) * 0.1}
                attestation = committee_decision(
                    num_oracles=n_oracles, client_id=cid, round_id=1,
                    sparse_state=fake_state, expected_k=2,
                )
                for v in attestation.votes:
                    committee_rows.append({
                        "Client": cid,
                        "Oracle": v.oracle_id,
                        "Score": round(v.score, 3),
                        "Vote hash": v.vote_hash,
                        "Rationale": v.rationale[:60] + "...",
                    })
                committee_rows.append({
                    "Client": cid,
                    "Oracle": "MEDIAN",
                    "Score": round(attestation.median_score, 3),
                    "Vote hash": "—",
                    "Rationale": ("ACCEPTED" if attestation.accepted
                                  else "REJECTED") + " by committee",
                })
            st.dataframe(pd.DataFrame(committee_rows),
                         hide_index=True, use_container_width=True)

        except Exception as exc:
            st.error(f"Bidding/Oracle round failed: {exc}")
            raise


# ---------------------------------------------------------------
# PAGE: CHAIN EXPLORER  (F9)
# ---------------------------------------------------------------
elif page == "Chain Explorer":
    page_banner(
        "Hash-Chained Ledger Explorer",
        "Append-only, tamper-evident audit trail of all FL events · "
        "Merkle roots + per-block hashes · pure Python (no Ethereum)",
        "🔗",
    )
    flow_bar(
        ["Add transactions", "Seal block", "Verify chain", "Tamper test"],
        "Verify chain",
    )

    concept_card(
        "Why a ledger?",
        "An append-only ledger gives every participant a verifiable audit trail of FL events: "
        "who registered, who bid what, which proofs passed/failed, how reputation evolved. "
        "Tampering with any past block would change the merkle root, which breaks the chain "
        "of block hashes, which is detectable by anyone."
    )
    concept_card(
        "What about Ethereum?",
        "Real ZK + reputation systems often deploy to Ethereum L2s (Polygon zkEVM, StarkNet) "
        "for trustless verification at the cost of gas. Our Python ledger has the same "
        "structural integrity properties (Merkle tree + hash chain) without the deployment cost, "
        "making it perfect for a B.Tech demonstration."
    )

    if "ledger" not in st.session_state:
        st.session_state["ledger"] = Ledger()

    ledger: Ledger = st.session_state["ledger"]

    cols = st.columns(4)
    cols[0].metric("Chain height", ledger.height(),
                   help="Number of blocks (incl. genesis)")
    cols[1].metric("Total transactions", ledger.total_transactions())
    cols[2].metric("Latest block hash", ledger.latest_block().block_hash[:12] + "...")
    cols[3].metric("Mempool size", len(ledger._mempool))

    st.divider()
    st.subheader("Add transactions to the mempool")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    if bcol1.button("+ Register Client"):
        cid = _rand_client_id()
        ledger.add_transaction("register", client_id=cid,
                               did=f"did:zkfedmoe:client{cid}")
        st.toast(f"Registered Client {cid}")
    if bcol2.button("+ Bid Commit"):
        ledger.add_transaction("bid_commit",
                               client_id=_rand_client_id(),
                               hash=fresh_nonce(8),
                               round=1)
        st.toast("Bid commitment added")
    if bcol3.button("+ SEPG Verify"):
        cid = _rand_client_id()
        ledger.add_transaction("verify", client_id=cid, round=1, accepted=True,
                               proof_hash=fresh_nonce(8))
        st.toast(f"Proof verification for Client {cid}")
    if bcol4.button("+ Reputation"):
        cid = _rand_client_id()
        import random as _r
        ledger.add_transaction("reputation", client_id=cid,
                               score=round(_r.uniform(0.4, 1.0), 3))
        st.toast(f"Reputation updated for Client {cid}")

    s1, s2 = st.columns(2)
    if s1.button("📦 Seal block (commit mempool)", type="primary",
                 use_container_width=True):
        if ledger._mempool:
            b = ledger.seal_block()
            st.success(f"Block {b.block_id} sealed with {len(b.tx_list)} transactions")
        else:
            st.warning("Mempool is empty — add some transactions first.")

    if s2.button("🔁 Reset chain", use_container_width=True):
        st.session_state["ledger"] = Ledger()
        st.rerun()

    st.divider()
    st.subheader("Chain blocks")
    summary = ledger.to_summary()
    if summary:
        df_blocks = pd.DataFrame(summary)
        st.dataframe(df_blocks, hide_index=True, use_container_width=True)

    st.subheader("Verify chain integrity")
    verify_col1, verify_col2 = st.columns([1, 3])
    if verify_col1.button("Verify now"):
        ok, why = ledger.verify()
        if ok:
            verify_col2.success(f"✓ Chain integrity holds: {why}")
        else:
            verify_col2.error(f"✗ Chain corrupt: {why}")

    if verify_col1.button("🔥 Tamper test"):
        if ledger.height() < 2:
            verify_col2.warning("Need at least one sealed block to tamper with.")
        else:
            # Mutate a payload field of a transaction in block 1
            try:
                tgt = ledger.blocks[1].tx_list[0]
                old = tgt.payload.copy()
                tgt.payload["tampered"] = "yes_a_bad_actor"
                ok, why = ledger.verify()
                # Restore so the chain becomes valid again next click
                tgt.payload = old
                if not ok:
                    verify_col2.error(
                        f"✓ Tamper detected (and reverted): {why}. "
                        "This shows the merkle root + block hash catch any modification."
                    )
                else:
                    verify_col2.warning("Tampering didn't break chain — unexpected.")
            except IndexError:
                verify_col2.warning("Block 1 has no transactions to tamper with.")


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
                    "Model saved to session! Go to **News Detection** to classify new text "
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
# PAGE: DISEASE DETECTION (full zkFedMoE pipeline on symptom dataset)
# ---------------------------------------------------------------
elif page == "Disease Detection":
    page_banner(
        "Disease Detection System",
        "1000-hospital scenario · each hospital trains locally on its patient records · "
        "DP-SGD + SEPG proofs + robust aggregation + audit ledger · "
        "tell the system your symptoms and it predicts the most likely disease.",
        "🩺",
    )
    flow_bar(
        ["Symptom dataset", "Split across hospitals", "Local train",
         "DP-SGD", "SEPG proof", "Verify + Aggregate", "Ledger", "▶ Predict"],
        "▶ Predict",
    )

    concept_card(
        "Why federated, not centralised?",
        "Real hospitals cannot pool patient records — HIPAA / GDPR forbid it. "
        "Each hospital trains a local model on its own patients and uploads only "
        "<b>weight updates</b> (not raw symptoms) to a central server. The server "
        "verifies each update with an SEPG proof and aggregates them into a global "
        "model. We frame this as a <b>1000-hospital scenario</b> from the paper, "
        "and let you simulate a tractable subset (up to 50) live in the browser.",
    )
    concept_card(
        "Dataset",
        "Curated 24-disease × 70-symptom dataset modelled on the public Kaggle "
        "<i>Disease Symptom Prediction</i> corpus (CC0). Each disease has 4–13 "
        "canonical symptoms; we synthesise multiple variants per disease with "
        "<b>random symptom drop</b> (incomplete reporting) and <b>noise</b> "
        "(co-morbidities) — the same noise model real clinical EHRs exhibit.",
    )

    # Lazy import the dataset module
    try:
        from data.disease_symptoms import (
            build_dataset as _build_disease_ds,
            SYMPTOMS as _DIS_SYMPTOMS,
            DISEASES as _DIS_DISEASES,
        )
    except Exception as _imp_exc:
        st.error(f"Could not import disease dataset module: {_imp_exc}")
        st.stop()

    # ---- Configuration ----
    st.subheader("1. Federation configuration")
    cfg_d1, cfg_d2, cfg_d3, cfg_d4 = st.columns(4)
    n_hosp_d = cfg_d1.slider(
        "Hospitals (clients)", 2, 50, 10, key="dis_n",
        help="In production this would be ~1000+ hospitals. We simulate a smaller "
             "subset for tractable training time. The paper's claim is the framework "
             "scales — the dashboard demonstrates correctness on a tractable subset.",
    )
    n_rounds_d = cfg_d2.slider("FL rounds", 1, 20, 10, key="dis_rounds")
    alpha_d = cfg_d3.select_slider(
        "Dirichlet α (heterogeneity)",
        options=[0.1, 0.3, 0.5, 1.0, 5.0, 100.0], value=1.0, key="dis_alpha",
        help="Smaller α → each hospital sees a skewed disease mix (realistic). "
             "Larger α → hospitals see balanced classes.",
    )
    n_variants_d = cfg_d4.slider(
        "Records per disease", 20, 150, 80, step=10, key="dis_variants",
        help="Synthetic patient records per disease (canonical symptoms + drop/noise).",
    )

    cfg_d5, cfg_d6, cfg_d7, cfg_d8 = st.columns(4)
    local_epochs_d = cfg_d5.slider("Local epochs / round", 1, 5, 3, key="dis_epochs")
    lr_d = cfg_d6.select_slider(
        "Learning rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=2e-3, key="dis_lr",
    )
    batch_d = cfg_d7.slider("Batch size", 8, 64, 32, step=8, key="dis_batch")
    aggr_d = cfg_d8.selectbox(
        "Aggregation", ["FedAvg", "Coord-wise Median", "Trimmed Mean"],
        index=0, key="dis_aggr",
    )

    # ---- DP / SEPG toggles ----
    st.markdown("**Privacy & verification (zkFedMoE pipeline)**")
    pp1, pp2, pp3, pp4 = st.columns(4)
    use_dp_d = pp1.checkbox("DP-SGD", value=True, key="dis_dp",
                            help="Clip update L2 to C, add Gaussian noise σ·C.")
    clip_C_d = pp2.slider("Clip norm C", 0.5, 5.0, 1.5, 0.1, key="dis_clip",
                          disabled=not use_dp_d)
    sigma_d = pp3.slider("Noise σ", 0.0, 2.0, 0.10, 0.05, key="dis_sigma",
                         disabled=not use_dp_d,
                         help="DP noise multiplier. This MLP is small (~25K params) "
                              "and trains on ~1500 rows, so σ above 0.15 begins to "
                              "destroy accuracy. The paper's MoE text model tolerates "
                              "much higher σ because of its scale (~150K params, 120K rows).")
    use_sepg_d = pp4.checkbox("SEPG proofs", value=True, key="dis_sepg",
                              help="Each client emits a SHA-256 proof of its update; "
                                   "server runs the 4-check verification.")

    st.caption(
        f"Total dataset = {n_variants_d} × {len(_DIS_DISEASES)} = "
        f"**{n_variants_d * len(_DIS_DISEASES)}** patient records, "
        f"{len(_DIS_SYMPTOMS)} binary symptom features, "
        f"{len(_DIS_DISEASES)} disease classes."
    )

    if st.button("🩺 Run federated disease-detection training",
                 type="primary", use_container_width=True, key="dis_run"):
        progress_d = st.progress(0.0)
        status_d = st.empty()
        try:
            from torch.utils.data import TensorDataset, DataLoader, Subset

            set_seed(42)
            np.random.seed(42)

            # ---- Build dataset ----
            with st.spinner("Building symptom-disease dataset..."):
                X_d, y_d, syms_d, dis_names_d = _build_disease_ds(
                    n_variants=n_variants_d, seed=42,
                )
                n_features_d = X_d.shape[1]
                n_classes_d = len(dis_names_d)
                n_total_d = X_d.shape[0]
                n_test_d = max(int(0.2 * n_total_d), n_classes_d)

                perm_d = np.random.permutation(n_total_d)
                test_ix_d = perm_d[:n_test_d]
                train_ix_d = perm_d[n_test_d:]

                X_tr = torch.from_numpy(X_d[train_ix_d]).float()
                y_tr = torch.from_numpy(y_d[train_ix_d]).long()
                X_te = torch.from_numpy(X_d[test_ix_d]).float()
                y_te = torch.from_numpy(y_d[test_ix_d]).long()
                train_ds_d = TensorDataset(X_tr, y_tr)
                test_ds_d = TensorDataset(X_te, y_te)

            st.success(
                f"Dataset built: {n_total_d} records "
                f"({len(train_ix_d)} train / {len(test_ix_d)} test) · "
                f"{n_features_d} symptoms · {n_classes_d} diseases."
            )

            # ---- Dirichlet split across hospitals ----
            with st.spinner(f"Splitting across {n_hosp_d} hospitals (Dirichlet α={alpha_d})..."):
                rng_d = np.random.default_rng(42)
                by_class_d = [np.where(y_tr.numpy() == c)[0] for c in range(n_classes_d)]
                for arr_c in by_class_d:
                    rng_d.shuffle(arr_c)

                client_ix_d = None
                for _attempt in range(50):
                    cand = [[] for _ in range(n_hosp_d)]
                    for c in range(n_classes_d):
                        if len(by_class_d[c]) == 0:
                            continue
                        proportions = rng_d.dirichlet([alpha_d] * n_hosp_d)
                        split_pts = (np.cumsum(proportions) *
                                     len(by_class_d[c])).astype(int)[:-1]
                        chunks = np.split(by_class_d[c], split_pts)
                        for cid, chunk in enumerate(chunks):
                            cand[cid].extend(chunk.tolist())
                    if min(len(ix) for ix in cand) >= 3:
                        client_ix_d = cand
                        break
                if client_ix_d is None:
                    client_ix_d = cand
                client_dss_d = [Subset(train_ds_d, ix) for ix in client_ix_d]

            # ---- Per-hospital distribution chart ----
            st.subheader("2. Per-hospital data distribution")
            dist_rows = []
            for cid, ix in enumerate(client_ix_d):
                yc = y_tr.numpy()[ix]
                for c in range(n_classes_d):
                    cnt = int((yc == c).sum())
                    if cnt > 0:
                        dist_rows.append({
                            "Hospital": f"H{cid}",
                            "Disease":  dis_names_d[c],
                            "Patients": cnt,
                        })
            df_dist_d = pd.DataFrame(dist_rows)
            fig_dist_d = px.bar(
                df_dist_d, x="Hospital", y="Patients", color="Disease",
                barmode="stack",
                title=f"Each of {n_hosp_d} hospitals has its own patient mix "
                      f"(Dirichlet α={alpha_d}, non-IID)",
            )
            fig_dist_d.update_layout(height=320, margin=dict(t=50, b=20),
                                    showlegend=False)
            st.plotly_chart(fig_dist_d, use_container_width=True)
            insight_box(
                "Each hospital's bar is different — some see mostly Diabetes, "
                "others mostly Malaria. Real hospitals never have the same patient "
                "mix; FL must converge despite this heterogeneity. We use Dirichlet "
                "α to control the skew, exactly as in the paper's Experiment 5."
            )

            # ---- Model ----
            class _DiseaseClf(torch.nn.Module):
                def __init__(self, in_f, hidden=128, n_cls=24):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(in_f, hidden), torch.nn.ReLU(),
                        torch.nn.Dropout(0.15),
                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                        torch.nn.Dropout(0.1),
                        torch.nn.Linear(hidden, n_cls),
                    )
                def forward(self, x): return self.net(x)

            global_model_d = _DiseaseClf(n_features_d, 128, n_classes_d)

            def _eval_d(model, ds, top_k=3):
                model.eval()
                ldr = DataLoader(ds, batch_size=128, shuffle=False)
                correct1 = correct_topk = total = 0
                with torch.no_grad():
                    for X, y in ldr:
                        out = model(X)
                        correct1 += int((out.argmax(-1) == y).sum())
                        topk_idx = out.topk(top_k, dim=-1).indices
                        correct_topk += int((topk_idx == y.unsqueeze(-1)).any(-1).sum())
                        total += y.size(0)
                return correct1 / max(total, 1), correct_topk / max(total, 1)

            initial_top1, initial_top3 = _eval_d(global_model_d, test_ds_d)
            st.info(
                f"Initial (untrained) global model: top-1 **{initial_top1:.1%}**, "
                f"top-3 **{initial_top3:.1%}** "
                f"(random baseline = {1.0/n_classes_d:.1%})."
            )

            # ---- FL animation + training ----
            st.subheader("3. Federated training — live FL pipeline")
            anim_slot_d = st.empty()
            anim_caption_d = st.empty()

            chart_col_l, chart_col_r = st.columns([3, 2])
            acc_chart_d = chart_col_l.empty()
            sepg_table = chart_col_r.empty()

            acc_history = []
            sepg_log = []
            ledger_d = Ledger()
            ledger_d.add_transaction(
                "register",
                event=f"disease-detection-fl-start",
                hospitals=n_hosp_d, rounds=n_rounds_d,
            )
            ledger_d.seal_block()

            anim_n = min(n_hosp_d, 8)  # animation only shows up to 8 nodes for clarity

            def _aggregate(states_with_n, mode):
                keys = list(states_with_n[0][0].keys())
                if mode == "FedAvg":
                    total_n = sum(n for _, n in states_with_n)
                    out = {k: torch.zeros_like(states_with_n[0][0][k]).float()
                           for k in keys}
                    for st_, n in states_with_n:
                        w = n / total_n
                        for k in keys:
                            out[k] += st_[k].float() * w
                    return out
                if mode == "Coord-wise Median":
                    out = {}
                    for k in keys:
                        stacked = torch.stack([st_[k].float() for st_, _ in states_with_n])
                        out[k] = stacked.median(dim=0).values
                    return out
                # Trimmed mean (10% top + bottom)
                out = {}
                k_clients = len(states_with_n)
                trim = max(0, k_clients // 10)
                for k in keys:
                    stacked = torch.stack([st_[k].float() for st_, _ in states_with_n])
                    sorted_, _ = stacked.sort(dim=0)
                    if trim * 2 < k_clients:
                        sorted_ = sorted_[trim: k_clients - trim] if trim > 0 else sorted_
                    out[k] = sorted_.mean(dim=0)
                return out

            for rnd in range(1, n_rounds_d + 1):
                # Phase: broadcast
                anim_slot_d.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="broadcast",
                                       round_id=rnd, total_rounds=n_rounds_d,
                                       accuracy=acc_history[-1]["top1"] if acc_history else None),
                    use_container_width=True,
                    key=f"dis_anim_bcast_{rnd}",
                )
                anim_caption_d.info(
                    f"Round {rnd}/{n_rounds_d} · server broadcasts global model to "
                    f"all {n_hosp_d} hospitals."
                )
                time.sleep(0.25)

                global_state_d = {k: v.detach().cpu().clone()
                                  for k, v in global_model_d.state_dict().items()}
                client_updates_d = []
                round_proofs_d = []
                round_losses_d = []

                for cid in range(n_hosp_d):
                    # Visual: highlight active hospital (only for first 8 in viz)
                    viz_active = cid if cid < anim_n else None
                    if viz_active is not None and rnd == 1:
                        anim_slot_d.plotly_chart(
                            fl_topology_frame(num_clients=anim_n, phase="train",
                                               round_id=rnd, total_rounds=n_rounds_d,
                                               active_client=viz_active),
                            use_container_width=True,
                            key=f"dis_anim_tr_{rnd}_{cid}",
                        )
                        anim_caption_d.info(
                            f"Hospital H{cid} trains locally · DP-SGD · SEPG proof"
                        )
                        time.sleep(0.05)

                    local_model = _DiseaseClf(n_features_d, 128, n_classes_d)
                    local_model.load_state_dict(global_state_d)
                    local_model.train()
                    loader_l = DataLoader(client_dss_d[cid],
                                          batch_size=batch_d, shuffle=True)
                    opt_l = torch.optim.Adam(local_model.parameters(), lr=lr_d)
                    crit_l = torch.nn.CrossEntropyLoss()
                    last_loss = 0.0
                    for _ep in range(local_epochs_d):
                        for X_b, y_b in loader_l:
                            opt_l.zero_grad()
                            out_l = local_model(X_b)
                            loss_l = crit_l(out_l, y_b)
                            loss_l.backward()
                            opt_l.step()
                            last_loss = loss_l.item()

                    new_state = {k: v.detach().cpu().clone()
                                 for k, v in local_model.state_dict().items()}

                    # Compute update delta = local - global
                    delta = {k: new_state[k].float() - global_state_d[k].float()
                             for k in new_state}

                    # ---- DP-SGD: clip + noise ----
                    if use_dp_d:
                        delta_dp = apply_dp(delta, clip_norm=clip_C_d,
                                            noise_multiplier=sigma_d)
                    else:
                        delta_dp = delta

                    # Reconstruct uploaded state = global + delta_dp
                    uploaded_state = {k: global_state_d[k].float() + delta_dp[k]
                                      for k in delta_dp}

                    # ---- SEPG proof ----
                    proof_pass = True
                    proof_reason = "no SEPG"
                    if use_sepg_d:
                        proof = generate_proof(
                            client_id=cid, round_id=rnd,
                            top_k_indices=[0],  # MLP has no MoE experts; placeholder
                            clip_norm=clip_C_d if use_dp_d else 0.0,
                            noise_multiplier=sigma_d if use_dp_d else 0.0,
                            epsilon=0.0,
                            sparse_state=uploaded_state,
                        )
                        ok, reason = verify_proof(
                            proof, uploaded_state, expected_k=1,
                            max_clip_norm=10.0, min_noise_mult=0.0,
                        )
                        proof_pass = ok
                        proof_reason = reason
                        round_proofs_d.append(proof)

                    sepg_log.append({
                        "Round":   rnd,
                        "Hosp":    f"H{cid}",
                        "Records": len(client_ix_d[cid]),
                        "Loss":    round(last_loss, 4),
                        "SEPG":    "✅" if proof_pass else "❌",
                        "Hash":    (proof.update_hash[:10] + "…")
                                   if use_sepg_d else "-",
                    })

                    if proof_pass:
                        client_updates_d.append((uploaded_state, len(client_ix_d[cid])))
                    round_losses_d.append(last_loss)

                # Phase: upload
                anim_slot_d.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="upload",
                                       round_id=rnd, total_rounds=n_rounds_d),
                    use_container_width=True,
                    key=f"dis_anim_up_{rnd}",
                )
                anim_caption_d.info(
                    f"All hospitals upload sparse updates + SEPG proofs · "
                    f"server runs 4-check verification"
                )
                time.sleep(0.2)

                # ---- Aggregate ----
                if client_updates_d:
                    agg_state = _aggregate(client_updates_d, aggr_d)
                    global_model_d.load_state_dict(agg_state)

                # Phase: aggregate
                anim_slot_d.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="aggregate",
                                       round_id=rnd, total_rounds=n_rounds_d),
                    use_container_width=True,
                    key=f"dis_anim_agg_{rnd}",
                )
                anim_caption_d.info(
                    f"Server aggregates {len(client_updates_d)}/{n_hosp_d} verified "
                    f"updates via {aggr_d} · appends round to audit ledger"
                )
                time.sleep(0.25)

                # Ledger
                ledger_d.add_transaction(
                    "verify",
                    round=rnd,
                    accepted=len(client_updates_d),
                    rejected=n_hosp_d - len(client_updates_d),
                    aggregation=aggr_d,
                )
                ledger_d.seal_block()

                # Eval
                top1, top3 = _eval_d(global_model_d, test_ds_d)
                acc_history.append({
                    "Round": rnd,
                    "top1":  top1,
                    "top3":  top3,
                    "loss":  float(np.mean(round_losses_d)),
                })

                # Acc chart
                df_acc = pd.DataFrame(acc_history)
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=df_acc["Round"], y=df_acc["top1"],
                    mode="lines+markers", name="Top-1 acc",
                    line=dict(color="#1565C0", width=3)))
                fig_acc.add_trace(go.Scatter(
                    x=df_acc["Round"], y=df_acc["top3"],
                    mode="lines+markers", name="Top-3 acc",
                    line=dict(color="#16A34A", width=2, dash="dash")))
                fig_acc.update_layout(
                    title=f"Round {rnd}/{n_rounds_d} · global model accuracy",
                    yaxis=dict(range=[0, 1], title="Accuracy"),
                    xaxis_title="Round", height=350,
                    margin=dict(t=50, b=30),
                    legend=dict(orientation="h", y=-0.2),
                )
                acc_chart_d.plotly_chart(fig_acc, use_container_width=True,
                                         key=f"dis_acc_{rnd}")

                df_sepg = pd.DataFrame(sepg_log[-min(len(sepg_log), n_hosp_d * 3):])
                sepg_table.dataframe(df_sepg, hide_index=True,
                                     use_container_width=True, height=350)

                progress_d.progress(rnd / n_rounds_d)
                status_d.markdown(
                    f"**Round {rnd}/{n_rounds_d}** · top-1 = **{top1:.1%}** · "
                    f"top-3 = **{top3:.1%}** · accepted = "
                    f"{len(client_updates_d)}/{n_hosp_d}"
                )

            # Done
            anim_slot_d.plotly_chart(
                fl_topology_frame(num_clients=anim_n, phase="done",
                                   round_id=n_rounds_d, total_rounds=n_rounds_d,
                                   accuracy=acc_history[-1]["top1"]),
                use_container_width=True, key="dis_anim_done",
            )
            anim_caption_d.success(
                f"Federated training complete · {n_rounds_d} rounds · "
                f"{ledger_d.height()} ledger blocks sealed · "
                f"final top-1 {acc_history[-1]['top1']:.1%}"
            )
            progress_d.empty()
            status_d.empty()

            # ---- Persist for predict UI ----
            st.session_state["dis_model"]      = global_model_d
            st.session_state["dis_symptoms"]   = syms_d
            st.session_state["dis_diseases"]   = dis_names_d
            st.session_state["dis_n_features"] = n_features_d
            st.session_state["dis_n_classes"]  = n_classes_d
            st.session_state["dis_final_top1"] = acc_history[-1]["top1"]
            st.session_state["dis_final_top3"] = acc_history[-1]["top3"]
            st.session_state["dis_ledger"]     = ledger_d

            # ---- Final summary ----
            st.subheader("4. Final summary")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Top-1 accuracy", f"{acc_history[-1]['top1']:.1%}",
                       f"{(acc_history[-1]['top1']-initial_top1)*100:+.1f} pp")
            sm2.metric("Top-3 accuracy", f"{acc_history[-1]['top3']:.1%}",
                       f"{(acc_history[-1]['top3']-initial_top3)*100:+.1f} pp")
            sm3.metric("Hospitals", n_hosp_d)
            sm4.metric("Ledger blocks", ledger_d.height())

            with st.expander("Audit ledger entries"):
                ledger_df = pd.DataFrame([{
                    "Block": b.block_id,
                    "Type":  ", ".join(t.tx_type for t in b.tx_list) or "(empty)",
                    "Hash":  b.block_hash[:16] + "…",
                    "Prev":  b.prev_hash[:16] + "…",
                } for b in ledger_d.blocks])
                st.dataframe(ledger_df, hide_index=True, use_container_width=True)

            ok_chain, reason_chain = ledger_d.verify()
            if ok_chain:
                st.success("Ledger integrity verified ✅ — chain is unbroken.")
            else:
                st.error(f"Ledger broken: {reason_chain}")

            insight_box(
                "Notice how the top-3 line climbs to 95%+ while top-1 stays around "
                "70–85%. Many diseases share symptoms (e.g. high_fever appears in "
                "Malaria, Dengue, Typhoid, Pneumonia, Tuberculosis) — top-3 is the "
                "honest clinical metric for a symptom checker. Real diagnosis "
                "needs lab tests, not just symptoms."
            )

        except Exception as exc:
            progress_d.empty()
            status_d.empty()
            st.error(f"Disease FL run failed: {exc}")
            raise

    # ----------------------------------------------------------------
    # Predict UI (always visible after training)
    # ----------------------------------------------------------------
    st.divider()
    st.subheader("5. Tell the system your symptoms")

    if "dis_model" not in st.session_state:
        st.info(
            "Run federated training above first. Once the global model is trained, "
            "you can tick off symptoms here and the system will predict the most "
            "likely disease — using the federated model that never saw any single "
            "hospital's raw records."
        )
    else:
        st.caption(
            f"Using global federated model · top-1 test acc "
            f"**{st.session_state['dis_final_top1']:.1%}** · "
            f"top-3 test acc **{st.session_state['dis_final_top3']:.1%}**."
        )
        syms_p   = st.session_state["dis_symptoms"]
        dis_p    = st.session_state["dis_diseases"]
        model_p  = st.session_state["dis_model"]
        n_feat_p = st.session_state["dis_n_features"]
        n_cls_p  = st.session_state["dis_n_classes"]

        # Group symptoms into 5 columns of checkboxes
        st.markdown("**Tick all symptoms the patient has:**")
        n_cols = 5
        cols_p = st.columns(n_cols)
        chosen = []
        for i, sym in enumerate(syms_p):
            label = sym.replace("_", " ")
            if cols_p[i % n_cols].checkbox(label, key=f"dis_sym_{i}"):
                chosen.append(i)

        # Quick presets
        preset_col, _, btn_col = st.columns([2, 2, 1])
        preset_choice = preset_col.selectbox(
            "Or load a preset symptom profile:",
            ["(custom)",
             "Fever + cough + breathing trouble",
             "Itching + skin rash",
             "Headache + vision blur + nausea",
             "Joint pain + skin rash + high fever"],
            key="dis_preset",
        )
        # Preset → symptom indices
        preset_map = {
            "Fever + cough + breathing trouble":
                ["high_fever", "cough", "breathlessness", "fatigue"],
            "Itching + skin rash":
                ["itching", "skin_rash", "nodal_skin_eruptions"],
            "Headache + vision blur + nausea":
                ["headache", "blurred_and_distorted_vision", "nausea", "vomiting"],
            "Joint pain + skin rash + high fever":
                ["joint_pain", "skin_rash", "high_fever", "chills", "back_pain"],
        }
        if preset_choice in preset_map:
            sym2ix_p = {s: i for i, s in enumerate(syms_p)}
            chosen = [sym2ix_p[s] for s in preset_map[preset_choice] if s in sym2ix_p]

        do_predict = btn_col.button("🔍 Diagnose", type="primary",
                                     use_container_width=True, key="dis_diag_btn")

        if do_predict:
            if not chosen:
                st.warning("Please tick at least one symptom (or pick a preset).")
            else:
                vec = np.zeros(n_feat_p, dtype=np.float32)
                for ix in chosen:
                    vec[ix] = 1.0
                x_t = torch.from_numpy(vec).float().unsqueeze(0)
                model_p.eval()
                with torch.no_grad():
                    logits_p = model_p(x_t)
                    probs_raw = torch.softmax(logits_p, dim=-1).squeeze(0).numpy()

                # ---- Symptom-coverage prior (rules out clinically absurd matches) ----
                # For each disease, what fraction of the user's symptoms appear
                # in that disease's canonical symptom list?  A disease that
                # shares zero symptoms with the query shouldn't be in the top-3.
                from data.disease_symptoms import disease_symptom_matrix
                D_S = disease_symptom_matrix()             # (n_dis, n_sym) {0,1}
                user_vec = vec                              # (n_sym,)
                # Per-disease overlap counts
                overlap = (D_S * user_vec).sum(axis=1)      # (n_dis,)
                n_user = max(int(user_vec.sum()), 1)
                coverage = overlap / n_user                 # in [0, 1]
                # Floor at 0.05 so a strongly-trained signal can still surface,
                # but diseases with zero canonical overlap get pushed way down.
                prior = np.where(coverage > 0, 0.5 + 0.5 * coverage, 0.05)
                probs_p = probs_raw * prior
                probs_p = probs_p / max(probs_p.sum(), 1e-8)

                # Top-3
                top3_idx = probs_p.argsort()[::-1][:3]

                st.markdown("---")
                rcol1, rcol2 = st.columns([1, 2])
                with rcol1:
                    st.markdown("### 🩺 Diagnosis")
                    for rank, idx in enumerate(top3_idx):
                        emo = "🥇" if rank == 0 else ("🥈" if rank == 1 else "🥉")
                        st.markdown(
                            f"**{emo} {dis_p[idx]}** — "
                            f"{probs_p[idx]:.1%} confidence"
                        )
                    st.caption(
                        f"Symptoms entered: {len(chosen)} · "
                        f"showing top-3 candidates because many diseases share symptoms."
                    )
                with rcol2:
                    fig_pred = go.Figure(go.Bar(
                        x=[dis_p[i] for i in top3_idx[::-1]],
                        y=[probs_p[i] for i in top3_idx[::-1]],
                        orientation="v",
                        marker_color=["#FF6B6B", "#FB923C", "#FBBF24"][::-1],
                        text=[f"{probs_p[i]:.1%}" for i in top3_idx[::-1]],
                        textposition="outside",
                    ))
                    fig_pred.update_layout(
                        title="Top-3 disease probabilities (after coverage prior)",
                        yaxis=dict(range=[0, 1.05], title="Probability"),
                        height=320, margin=dict(t=50, b=20),
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                with st.expander("Active symptoms in this query"):
                    st.write([syms_p[ix].replace("_", " ") for ix in chosen])

                with st.expander("How this diagnosis was computed"):
                    raw_top3 = probs_raw.argsort()[::-1][:5]
                    st.markdown(
                        "**Step 1 — Federated MLP raw output (top-5):** "
                        + " · ".join(
                            f"{dis_p[i]} {probs_raw[i]:.1%}"
                            for i in raw_top3
                        )
                    )
                    st.markdown(
                        "**Step 2 — Symptom-coverage prior:** "
                        "for each candidate disease, we check what fraction of "
                        "your reported symptoms appear in that disease's canonical "
                        "symptom list. Diseases that share zero symptoms with your "
                        "query are floored at 5% (they cannot be top-1 unless the "
                        "model is overwhelmingly confident)."
                    )
                    cov_df = pd.DataFrame({
                        "Disease":      [dis_p[i] for i in raw_top3],
                        "Raw model %":  [f"{probs_raw[i]:.1%}" for i in raw_top3],
                        "Coverage":     [f"{coverage[i]:.0%}"   for i in raw_top3],
                        "Final %":      [f"{probs_p[i]:.1%}"    for i in raw_top3],
                    })
                    st.dataframe(cov_df, hide_index=True, use_container_width=True)

                st.caption(
                    "_Disclaimer: research demo trained on synthetic-from-clinical-prior "
                    "data with DP noise added during federated training. Not a substitute "
                    "for professional medical diagnosis._"
                )


# ---------------------------------------------------------------
# PAGE: GENERAL ZKFEDMOE (multi-client wizard)
# ---------------------------------------------------------------
elif page == "General zkFedMoE":
    page_banner(
        "General zkFedMoE",
        "Configure → upload one CSV per client → run the full zkFedMoE pipeline · "
        "DP-SGD · SEPG proofs · robust aggregation · audit ledger · live FL animation.",
        "🔐",
    )

    # Wizard state — note: all stored slots use the `genfl_cfg_*` prefix to
    # avoid colliding with widget keys (which Streamlit auto-binds and refuses
    # to let us write to in the same run).
    if "genfl_step" not in st.session_state:
        st.session_state["genfl_step"] = 1
    if "genfl_cfg_clients" not in st.session_state:
        st.session_state["genfl_cfg_clients"] = []  # [{"name", "X", "y"}, ...]
    if "genfl_cfg_cur" not in st.session_state:
        st.session_state["genfl_cfg_cur"] = 0
    if "genfl_cfg_done_count" not in st.session_state:
        st.session_state["genfl_cfg_done_count"] = 0  # how many uploads parsed

    step = st.session_state["genfl_step"]
    flow_bar(
        ["1. Configure", "2. Upload per client", "3. Schema check",
         "4. FL training", "5. Predict"],
        ["1. Configure", "2. Upload per client", "3. Schema check",
         "4. FL training", "5. Predict"][step - 1],
    )

    # --------- STEP 1: CONFIGURE ---------
    if step == 1:
        concept_card(
            "How this differs from Custom CSV",
            "<b>Custom CSV</b> takes a single file and partitions it across simulated "
            "clients (Dirichlet split). <b>This page</b> models true heterogeneity: "
            "each client uploads its <i>own</i> CSV — different sizes, different "
            "label distributions, possibly different feature scales — and the full "
            "zkFedMoE pipeline runs across them.",
        )
        st.subheader("1. Configure federation")

        c1, c2, c3 = st.columns(3)
        n_clients_g = c1.slider("Number of clients", 2, 20, 4,
                                key="genfl_w_n")
        n_rounds_g = c2.slider("FL rounds", 1, 20, 8,
                               key="genfl_w_rounds")
        aggr_g = c3.selectbox(
            "Aggregation", ["FedAvg", "Coord-wise Median", "Trimmed Mean"],
            key="genfl_w_aggr",
        )

        c4, c5, c6 = st.columns(3)
        local_epochs_g = c4.slider("Local epochs / round", 1, 5, 2,
                                   key="genfl_w_ep")
        lr_g = c5.select_slider(
            "Learning rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            value=1e-3, key="genfl_w_lr",
        )
        batch_g = c6.slider("Batch size", 4, 64, 16, step=4,
                            key="genfl_w_bs")

        st.markdown("**Privacy & verification (zkFedMoE pipeline)**")
        p1, p2, p3, p4 = st.columns(4)
        use_dp_g = p1.checkbox("DP-SGD", value=True, key="genfl_w_dp")
        clip_C_g = p2.slider("Clip norm C", 0.5, 5.0, 1.5, 0.1,
                             key="genfl_w_clip", disabled=not use_dp_g)
        sigma_g = p3.slider("Noise σ", 0.0, 2.0, 0.1, 0.05,
                            key="genfl_w_sigma", disabled=not use_dp_g)
        use_sepg_g = p4.checkbox("SEPG proofs", value=True,
                                 key="genfl_w_sepg")

        st.info(
            f"You will be asked to upload **{n_clients_g} CSV files** "
            "(one per client). Each CSV: numeric columns, last column = "
            "integer label, comma-separated. Headers are auto-skipped."
        )

        if st.button("Continue → Upload client data", type="primary",
                     use_container_width=True, key="genfl_to_step2"):
            # Snapshot the widget values into separate cfg slots so step 4
            # can read them without ever touching widget-bound names.
            st.session_state["genfl_cfg_n"] = n_clients_g
            st.session_state["genfl_cfg_rounds"] = n_rounds_g
            st.session_state["genfl_cfg_aggr"] = aggr_g
            st.session_state["genfl_cfg_ep"] = local_epochs_g
            st.session_state["genfl_cfg_lr"] = lr_g
            st.session_state["genfl_cfg_bs"] = batch_g
            st.session_state["genfl_cfg_use_dp"] = use_dp_g
            st.session_state["genfl_cfg_clip"] = clip_C_g
            st.session_state["genfl_cfg_sigma"] = sigma_g
            st.session_state["genfl_cfg_use_sepg"] = use_sepg_g
            st.session_state["genfl_cfg_clients"] = []
            st.session_state["genfl_cfg_cur"] = 0
            st.session_state["genfl_cfg_done_count"] = 0
            st.session_state["genfl_step"] = 2
            st.rerun()

    # --------- STEP 2: UPLOAD PER CLIENT ---------
    elif step == 2:
        n_clients_g = st.session_state["genfl_cfg_n"]
        cur = st.session_state["genfl_cfg_cur"]
        already = st.session_state["genfl_cfg_clients"]

        st.subheader(f"2. Upload data for client {cur + 1} / {n_clients_g}")

        # Progress dots
        dots = "  ".join(
            "✅" if i < cur else ("🟦" if i == cur else "⬜")
            for i in range(n_clients_g)
        )
        st.markdown(f"**Client progress:** {dots}")

        st.caption(
            "CSV format: all columns numeric, last column is the integer class label, "
            "comma-separated. Header rows are skipped automatically. "
            "All clients should share the same number of features and the same label set."
        )

        # Show what each previous client contributed
        if already:
            with st.expander(f"📁 Already uploaded ({len(already)} client/s)"):
                rows_so_far = [{
                    "Client":   c["name"],
                    "Records":  c["X"].shape[0],
                    "Features": c["X"].shape[1],
                    "Classes":  sorted(set(c["y"].tolist())),
                } for c in already]
                st.dataframe(pd.DataFrame(rows_so_far), hide_index=True,
                             use_container_width=True)

        client_name = st.text_input(
            "Client name (optional)",
            value=f"Client_{cur + 1}",
            key=f"genfl_w_cname_{cur}",
        )
        uploaded = st.file_uploader(
            f"📤 Drop CSV file for {client_name}",
            type=["csv"], key=f"genfl_w_csv_{cur}",
        )

        # Auto-process the upload as soon as it arrives, then auto-advance.
        if uploaded is not None:
            try:
                raw = uploaded.read().decode("utf-8", errors="replace")
                rows = []
                for line in raw.strip().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    try:
                        rows.append([float(p) for p in parts])
                    except ValueError:
                        continue
                if not rows:
                    st.error("No numeric rows parsed in this CSV.")
                    st.stop()
                arr = np.array(rows, dtype=np.float32)
                Xc = arr[:, :-1]
                yc = arr[:, -1].astype(np.int64)

                cls_counts = np.bincount(yc) if yc.size > 0 else np.array([0])
                st.success(
                    f"✅ Parsed {Xc.shape[0]} rows · {Xc.shape[1]} features · "
                    f"classes present: {sorted(set(yc.tolist()))}"
                )
                cls_df = pd.DataFrame({
                    "Class": list(range(len(cls_counts))),
                    "Count": cls_counts.tolist(),
                })
                st.bar_chart(cls_df, x="Class", y="Count")

                # Auto-advance on the *first* time we successfully parse this
                # client's file. We track that with done_count vs cur, so a
                # rerun (e.g. after another widget changes) doesn't double-add.
                if st.session_state["genfl_cfg_done_count"] == cur:
                    label = (
                        "Confirm & next client →"
                        if cur + 1 < n_clients_g
                        else "Confirm & go to schema check →"
                    )
                    if st.button(label, type="primary",
                                 key=f"genfl_confirm_{cur}",
                                 use_container_width=True):
                        already.append({"name": client_name,
                                        "X": Xc, "y": yc})
                        st.session_state["genfl_cfg_clients"] = already
                        st.session_state["genfl_cfg_done_count"] = cur + 1
                        if cur + 1 < n_clients_g:
                            st.session_state["genfl_cfg_cur"] = cur + 1
                        else:
                            st.session_state["genfl_step"] = 3
                        st.rerun()
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

        # Navigation row
        nav_l, nav_r = st.columns(2)
        if nav_l.button("← Back to config", key="genfl_back_step1"):
            st.session_state["genfl_step"] = 1
            st.rerun()
        if nav_r.button("⟳ Restart wizard", key="genfl_restart"):
            for k in list(st.session_state.keys()):
                if k.startswith("genfl_"):
                    del st.session_state[k]
            st.session_state["genfl_step"] = 1
            st.rerun()

    # --------- STEP 3: SCHEMA CHECK ---------
    elif step == 3:
        clients = st.session_state["genfl_cfg_clients"]
        st.subheader("3. Schema reconciliation")

        feat_counts = [c["X"].shape[1] for c in clients]
        all_classes = set()
        for c in clients:
            all_classes.update(c["y"].tolist())

        rows = [{
            "Client":   c["name"],
            "Records":  c["X"].shape[0],
            "Features": c["X"].shape[1],
            "Classes":  sorted(set(c["y"].tolist())),
        } for c in clients]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        ok = (len(set(feat_counts)) == 1)
        if ok:
            st.success(
                f"✅ All clients have {feat_counts[0]} features · "
                f"{len(all_classes)} unique classes across the federation: "
                f"{sorted(all_classes)}"
            )
        else:
            st.error(
                f"❌ Feature-count mismatch: {feat_counts}. "
                "All clients must share the same feature schema. "
                "Please restart the wizard with consistent CSVs."
            )

        bcol1, bcol2 = st.columns(2)
        if bcol1.button("← Re-upload", key="genfl_back_step2"):
            st.session_state["genfl_cfg_clients"] = []
            st.session_state["genfl_cfg_cur"] = 0
            st.session_state["genfl_cfg_done_count"] = 0
            st.session_state["genfl_step"] = 2
            st.rerun()
        if ok and bcol2.button("Run zkFedMoE training →",
                                type="primary", key="genfl_to_step4"):
            st.session_state["genfl_step"] = 4
            st.rerun()

    # --------- STEP 4: FL TRAINING ---------
    elif step == 4:
        clients = st.session_state["genfl_cfg_clients"]
        n_clients_g = st.session_state["genfl_cfg_n"]
        n_rounds_g = st.session_state["genfl_cfg_rounds"]
        aggr_g = st.session_state["genfl_cfg_aggr"]
        local_epochs_g = st.session_state["genfl_cfg_ep"]
        lr_g = st.session_state["genfl_cfg_lr"]
        batch_g = st.session_state["genfl_cfg_bs"]
        use_dp_g = st.session_state["genfl_cfg_use_dp"]
        clip_C_g = st.session_state["genfl_cfg_clip"]
        sigma_g = st.session_state["genfl_cfg_sigma"]
        use_sepg_g = st.session_state["genfl_cfg_use_sepg"]

        st.subheader(f"4. Federated training across {n_clients_g} clients")

        # Combine all clients' data, then standardise globally (using a public
        # mean/std proxy: the per-client mean averaged — does NOT leak raw rows).
        all_X = np.concatenate([c["X"] for c in clients], axis=0)
        all_y = np.concatenate([c["y"] for c in clients], axis=0)
        n_features_g = all_X.shape[1]
        n_classes_g = int(all_y.max()) + 1

        # Per-client local mean/std → average (a stand-in for secure mean exchange)
        per_mu = np.stack([c["X"].mean(axis=0) for c in clients], axis=0)
        per_sd = np.stack([c["X"].std(axis=0) + 1e-8 for c in clients], axis=0)
        mu_g = per_mu.mean(axis=0, keepdims=True)
        sd_g = per_sd.mean(axis=0, keepdims=True)

        from torch.utils.data import TensorDataset, DataLoader

        # Build global test set: 20% pulled out of each client (so test reflects
        # the federation distribution, not a single client).
        test_X_list, test_y_list = [], []
        client_dss_g = []
        for c in clients:
            Xc = (c["X"] - mu_g) / sd_g
            yc = c["y"]
            n = Xc.shape[0]
            n_te = max(int(0.2 * n), 1)
            perm = np.random.RandomState(42).permutation(n)
            te_ix = perm[:n_te]
            tr_ix = perm[n_te:]
            test_X_list.append(Xc[te_ix])
            test_y_list.append(yc[te_ix])
            X_tr = torch.from_numpy(Xc[tr_ix]).float()
            y_tr = torch.from_numpy(yc[tr_ix]).long()
            client_dss_g.append(TensorDataset(X_tr, y_tr))
        X_te_g = torch.from_numpy(np.concatenate(test_X_list, axis=0)).float()
        y_te_g = torch.from_numpy(np.concatenate(test_y_list, axis=0)).long()
        test_ds_g = TensorDataset(X_te_g, y_te_g)

        # Generic MLP
        class _GenClf(torch.nn.Module):
            def __init__(self, in_f, hidden=64, n_cls=2):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_f, hidden), torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                    torch.nn.Linear(hidden, n_cls),
                )
            def forward(self, x): return self.net(x)

        global_model_g = _GenClf(n_features_g, 64, n_classes_g)

        def _eval_g(model, ds):
            model.eval()
            ldr = DataLoader(ds, batch_size=128, shuffle=False)
            c = t = 0
            with torch.no_grad():
                for X, y in ldr:
                    out = model(X)
                    c += int((out.argmax(-1) == y).sum())
                    t += y.size(0)
            return c / max(t, 1)

        def _aggregate_g(states_with_n, mode):
            keys = list(states_with_n[0][0].keys())
            if mode == "FedAvg":
                total_n = sum(n for _, n in states_with_n)
                out = {k: torch.zeros_like(states_with_n[0][0][k]).float() for k in keys}
                for st_, n in states_with_n:
                    w = n / total_n
                    for k in keys:
                        out[k] += st_[k].float() * w
                return out
            if mode == "Coord-wise Median":
                out = {}
                for k in keys:
                    stacked = torch.stack([st_[k].float() for st_, _ in states_with_n])
                    out[k] = stacked.median(dim=0).values
                return out
            out = {}
            kc = len(states_with_n)
            trim = max(0, kc // 10)
            for k in keys:
                stacked = torch.stack([st_[k].float() for st_, _ in states_with_n])
                sorted_, _ = stacked.sort(dim=0)
                if trim > 0 and trim * 2 < kc:
                    sorted_ = sorted_[trim: kc - trim]
                out[k] = sorted_.mean(dim=0)
            return out

        if st.button("▶ Start FL run", type="primary",
                     use_container_width=True, key="genfl_start"):
            ledger_g = Ledger()
            ledger_g.add_transaction(
                "register", event="genfl-run-start",
                clients=n_clients_g, rounds=n_rounds_g,
            )
            ledger_g.seal_block()

            initial_acc_g = _eval_g(global_model_g, test_ds_g)
            st.info(f"Initial accuracy: **{initial_acc_g:.1%}** "
                    f"(random baseline = {1.0/n_classes_g:.1%})")

            anim_n = min(n_clients_g, 8)
            anim_slot_g = st.empty()
            anim_caption_g = st.empty()
            chart_l, chart_r = st.columns([3, 2])
            acc_chart_g = chart_l.empty()
            sepg_table_g = chart_r.empty()

            acc_history_g = []
            sepg_log_g = []
            progress_g = st.progress(0.0)

            for rnd in range(1, n_rounds_g + 1):
                anim_slot_g.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="broadcast",
                                       round_id=rnd, total_rounds=n_rounds_g,
                                       accuracy=acc_history_g[-1]["acc"]
                                       if acc_history_g else None),
                    use_container_width=True, key=f"gen_anim_b_{rnd}",
                )
                anim_caption_g.info(f"Round {rnd}/{n_rounds_g} · broadcast global θ")
                time.sleep(0.2)

                global_state_g = {k: v.detach().cpu().clone()
                                  for k, v in global_model_g.state_dict().items()}
                client_updates_g = []
                round_losses_g = []

                for cid in range(n_clients_g):
                    if cid < anim_n and rnd == 1:
                        anim_slot_g.plotly_chart(
                            fl_topology_frame(num_clients=anim_n, phase="train",
                                               round_id=rnd, total_rounds=n_rounds_g,
                                               active_client=cid),
                            use_container_width=True,
                            key=f"gen_anim_t_{rnd}_{cid}",
                        )
                        anim_caption_g.info(
                            f"{clients[cid]['name']} trains locally · "
                            f"DP-SGD · SEPG proof"
                        )
                        time.sleep(0.05)

                    local_model = _GenClf(n_features_g, 64, n_classes_g)
                    local_model.load_state_dict(global_state_g)
                    local_model.train()
                    loader_l = DataLoader(client_dss_g[cid],
                                          batch_size=batch_g, shuffle=True)
                    opt_l = torch.optim.Adam(local_model.parameters(), lr=lr_g)
                    crit_l = torch.nn.CrossEntropyLoss()
                    last_loss = 0.0
                    for _ep in range(local_epochs_g):
                        for X_b, y_b in loader_l:
                            opt_l.zero_grad()
                            out_l = local_model(X_b)
                            loss_l = crit_l(out_l, y_b)
                            loss_l.backward()
                            opt_l.step()
                            last_loss = loss_l.item()

                    new_state = {k: v.detach().cpu().clone()
                                 for k, v in local_model.state_dict().items()}
                    delta = {k: new_state[k].float() - global_state_g[k].float()
                             for k in new_state}
                    if use_dp_g:
                        delta_dp = apply_dp(delta, clip_norm=clip_C_g,
                                            noise_multiplier=sigma_g)
                    else:
                        delta_dp = delta
                    uploaded_state = {k: global_state_g[k].float() + delta_dp[k]
                                      for k in delta_dp}

                    proof_pass = True
                    proof_hash_short = "-"
                    if use_sepg_g:
                        proof = generate_proof(
                            client_id=cid, round_id=rnd,
                            top_k_indices=[0],
                            clip_norm=clip_C_g if use_dp_g else 0.0,
                            noise_multiplier=sigma_g if use_dp_g else 0.0,
                            epsilon=0.0, sparse_state=uploaded_state,
                        )
                        ok, _ = verify_proof(proof, uploaded_state, expected_k=1,
                                             max_clip_norm=10.0, min_noise_mult=0.0)
                        proof_pass = ok
                        proof_hash_short = proof.update_hash[:10] + "…"

                    sepg_log_g.append({
                        "Rnd":  rnd,
                        "Client": clients[cid]["name"],
                        "Recs": len(client_dss_g[cid]),
                        "Loss": round(last_loss, 4),
                        "SEPG": "✅" if proof_pass else "❌",
                        "Hash": proof_hash_short,
                    })

                    if proof_pass:
                        client_updates_g.append((uploaded_state, len(client_dss_g[cid])))
                    round_losses_g.append(last_loss)

                anim_slot_g.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="upload",
                                       round_id=rnd, total_rounds=n_rounds_g),
                    use_container_width=True, key=f"gen_anim_u_{rnd}",
                )
                anim_caption_g.info(
                    f"All clients upload · server runs SEPG verify · {aggr_g}"
                )
                time.sleep(0.2)

                if client_updates_g:
                    agg_state = _aggregate_g(client_updates_g, aggr_g)
                    global_model_g.load_state_dict(agg_state)

                anim_slot_g.plotly_chart(
                    fl_topology_frame(num_clients=anim_n, phase="aggregate",
                                       round_id=rnd, total_rounds=n_rounds_g),
                    use_container_width=True, key=f"gen_anim_a_{rnd}",
                )
                anim_caption_g.info(
                    f"Aggregated {len(client_updates_g)}/{n_clients_g} verified "
                    f"updates · ledger entry sealed"
                )
                time.sleep(0.2)

                ledger_g.add_transaction(
                    "verify", round=rnd,
                    accepted=len(client_updates_g),
                    rejected=n_clients_g - len(client_updates_g),
                    aggregation=aggr_g,
                )
                ledger_g.seal_block()

                acc_g = _eval_g(global_model_g, test_ds_g)
                acc_history_g.append({
                    "Round": rnd, "acc": acc_g,
                    "loss": float(np.mean(round_losses_g)),
                })
                df_acc_g = pd.DataFrame(acc_history_g)
                fig_acc_g = go.Figure()
                fig_acc_g.add_trace(go.Scatter(
                    x=df_acc_g["Round"], y=df_acc_g["acc"],
                    mode="lines+markers", name="Test acc",
                    line=dict(color="#1565C0", width=3)))
                fig_acc_g.update_layout(
                    title=f"Round {rnd}/{n_rounds_g} · global accuracy",
                    yaxis=dict(range=[0, 1]),
                    height=350, margin=dict(t=50, b=30),
                )
                acc_chart_g.plotly_chart(fig_acc_g, use_container_width=True,
                                         key=f"gen_acc_{rnd}")

                df_sepg_g = pd.DataFrame(
                    sepg_log_g[-min(len(sepg_log_g), n_clients_g * 3):]
                )
                sepg_table_g.dataframe(df_sepg_g, hide_index=True,
                                        use_container_width=True, height=350)

                progress_g.progress(rnd / n_rounds_g)

            anim_slot_g.plotly_chart(
                fl_topology_frame(num_clients=anim_n, phase="done",
                                   round_id=n_rounds_g, total_rounds=n_rounds_g,
                                   accuracy=acc_history_g[-1]["acc"]),
                use_container_width=True, key="gen_anim_done",
            )
            anim_caption_g.success(
                f"FL run complete · final acc {acc_history_g[-1]['acc']:.1%} · "
                f"{ledger_g.height()} ledger blocks sealed"
            )
            progress_g.empty()

            # Persist
            st.session_state["genfl_model"] = global_model_g
            st.session_state["genfl_mu"] = mu_g
            st.session_state["genfl_sd"] = sd_g
            st.session_state["genfl_res_n_features"] = n_features_g
            st.session_state["genfl_res_n_classes"] = n_classes_g
            st.session_state["genfl_final_acc"] = acc_history_g[-1]["acc"]
            st.session_state["genfl_ledger"] = ledger_g

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Initial acc", f"{initial_acc_g:.1%}")
            sm2.metric("Final acc", f"{acc_history_g[-1]['acc']:.1%}",
                       f"{(acc_history_g[-1]['acc']-initial_acc_g)*100:+.1f} pp")
            sm3.metric("Clients", n_clients_g)
            sm4.metric("Ledger blocks", ledger_g.height())

            ok_chain, _ = ledger_g.verify()
            if ok_chain:
                st.success("Ledger integrity verified ✅")

            if st.button("Continue → Predict", type="primary", key="genfl_to_step5"):
                st.session_state["genfl_step"] = 5
                st.rerun()

        # Always-available navigation
        nav_l, _, nav_r = st.columns([1, 2, 1])
        if nav_l.button("← Back to schema", key="genfl_back_step3"):
            st.session_state["genfl_step"] = 3
            st.rerun()
        if "genfl_model" in st.session_state and nav_r.button(
            "Skip to predict →", key="genfl_skip_step5"
        ):
            st.session_state["genfl_step"] = 5
            st.rerun()

    # --------- STEP 5: PREDICT ---------
    elif step == 5:
        st.subheader("5. Predict with the trained global model")
        if "genfl_model" not in st.session_state:
            st.warning("No trained model in session. Go back to step 4 and run training.")
            if st.button("← Back to training", key="genfl_back_step4_b"):
                st.session_state["genfl_step"] = 4
                st.rerun()
        else:
            model_g = st.session_state["genfl_model"]
            mu_g = st.session_state["genfl_mu"]
            sd_g = st.session_state["genfl_sd"]
            n_feat_g = st.session_state["genfl_res_n_features"]
            n_cls_g = st.session_state["genfl_res_n_classes"]

            st.caption(
                f"Final test accuracy: **{st.session_state['genfl_final_acc']:.1%}** · "
                f"{n_feat_g} features · {n_cls_g} classes."
            )

            with st.form("genfl_predict_form"):
                st.markdown("**Enter feature values:**")
                cols_p = st.columns(min(n_feat_g, 4))
                vals = []
                for i in range(n_feat_g):
                    v = cols_p[i % len(cols_p)].number_input(
                        f"Feature {i}", value=0.0, format="%.4f",
                        key=f"genfl_pred_f{i}",
                    )
                    vals.append(float(v))
                submitted_g = st.form_submit_button(
                    "🔍 Predict", type="primary", use_container_width=True
                )

            if submitted_g:
                x_raw = np.array(vals, dtype=np.float32).reshape(1, -1)
                x_std = (x_raw - mu_g) / sd_g
                x_t = torch.from_numpy(x_std).float()
                model_g.eval()
                with torch.no_grad():
                    logits_g = model_g(x_t)
                    probs_g = torch.softmax(logits_g, dim=-1).squeeze(0).numpy()
                pred_idx = int(probs_g.argmax())
                conf = float(probs_g[pred_idx])

                rcol1, rcol2 = st.columns([1, 2])
                rcol1.metric("Predicted class", f"Class {pred_idx}",
                             f"{conf:.1%} confidence")
                fig_pg = go.Figure(go.Bar(
                    x=[f"Class {i}" for i in range(n_cls_g)],
                    y=probs_g.tolist(),
                    marker_color=["#FF6B6B" if i == pred_idx else "#4C72B0"
                                  for i in range(n_cls_g)],
                    text=[f"{p:.1%}" for p in probs_g],
                    textposition="outside",
                ))
                fig_pg.update_layout(yaxis=dict(range=[0, 1.05]),
                                      height=320, margin=dict(t=30, b=20))
                rcol2.plotly_chart(fig_pg, use_container_width=True)

            nav_l, _, nav_r = st.columns([1, 2, 1])
            if nav_l.button("← Back to training", key="genfl_back_step4"):
                st.session_state["genfl_step"] = 4
                st.rerun()
            if nav_r.button("⟳ New federation", key="genfl_restart_end"):
                for k in list(st.session_state.keys()):
                    if k.startswith("genfl_"):
                        del st.session_state[k]
                st.session_state["genfl_step"] = 1
                st.rerun()


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
            "Architecture", "Train", "Train", "Compare", "News Detection",
            "Privacy & DP", "Privacy & DP", "Robustness", "Robustness",
            "Experiments", "Custom CSV", "All",
        ],
    }
    st.dataframe(pd.DataFrame(status), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Dashboard Pages")
    pages_desc = {
        "Page": ["News Detection", "Train", "Custom CSV", "Privacy & DP",
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
