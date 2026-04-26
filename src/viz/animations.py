"""Plotly-based live animations for the Predict and Train dashboard pages.

Two pure functions returning Plotly Figures:
  - predict_animation_frame: 5-stage data-flow strip for inference
  - fl_topology_frame:        N-client + central-server diagram for FL rounds

No model logic, no Streamlit imports here. The dashboard drives these by
rendering frames into st.empty() placeholders with short time.sleep() pauses
between calls.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence

import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Predict page: 5-stage data-flow animation
# ---------------------------------------------------------------------------

_PREDICT_STAGES = [
    ("Tokens",     "Text -> token IDs"),
    ("Embedding",  "IDs -> 64-dim vector"),
    ("MoE Router", "Score 4-8 experts"),
    ("Top-K",      "Pick K best experts"),
    ("Classifier", "LoRA -> class logits"),
    ("Output",     "Softmax -> argmax"),
]

_ACTIVE_COLOR = "#FF6B6B"
_PAST_COLOR   = "#7E9BC4"
_FUTURE_COLOR = "#D9D9D9"


def predict_animation_frame(
    stage: int,
    tokens: Sequence[str],
    oov: Sequence[str],
    num_experts: int,
    top_experts: Sequence[int],
    pred_label: str,
    class_probs: dict,
) -> go.Figure:
    """One frame of the inference data-flow animation.

    Args:
        stage: 0..5 (which stage to highlight).
        tokens: list of input tokens.
        oov: list of out-of-vocab tokens.
        num_experts: total experts in the MoE layer.
        top_experts: indices of the active (Top-K) experts.
        pred_label: predicted class label.
        class_probs: {label: prob} dict.
    """
    n = len(_PREDICT_STAGES)
    xs = list(range(n))

    colors = []
    for i in range(n):
        if i < stage:
            colors.append(_PAST_COLOR)
        elif i == stage:
            colors.append(_ACTIVE_COLOR)
        else:
            colors.append(_FUTURE_COLOR)

    labels = [s[0] for s in _PREDICT_STAGES]
    sublabels = [s[1] for s in _PREDICT_STAGES]

    fig = go.Figure()

    # Stage circles
    fig.add_trace(go.Scatter(
        x=xs, y=[0] * n,
        mode="markers+text",
        marker=dict(size=72, color=colors,
                    line=dict(color="#1a3d7c", width=2)),
        text=labels,
        textposition="middle center",
        textfont=dict(color="white", size=11, family="Arial Black"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Sub-labels under each stage
    fig.add_trace(go.Scatter(
        x=xs, y=[-0.55] * n,
        mode="text",
        text=sublabels,
        textfont=dict(size=10, color="#444"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Connector arrows
    for i in range(n - 1):
        col = _ACTIVE_COLOR if i < stage else "#BBBBBB"
        fig.add_annotation(
            x=i + 1, y=0, ax=i + 0.0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4,
            arrowwidth=2, arrowcolor=col,
            standoff=36, startstandoff=36,
        )

    # Per-stage detail above the active node
    detail = ""
    if stage == 0:
        sample = " ".join(tokens[:6]) if tokens else "(empty)"
        n_oov = len(oov)
        detail = f"Tokens: <b>{sample}</b>...  |  OOV: {n_oov}"
    elif stage == 1:
        detail = f"Mean-pooled into a <b>64-dim</b> dense vector"
    elif stage == 2:
        detail = f"Routing softmax over <b>{num_experts}</b> experts"
    elif stage == 3:
        detail = f"Active experts: <b>{top_experts}</b>"
    elif stage == 4:
        detail = "LoRA-adapted Linear: 64 -> n_classes logits"
    elif stage == 5:
        conf = max(class_probs.values()) if class_probs else 0
        detail = f"Predicted: <b>{pred_label}</b>  ({conf:.0%} confidence)"

    fig.add_annotation(
        x=stage, y=0.7,
        text=detail,
        showarrow=False,
        font=dict(size=12, color="#1a3d7c"),
        bgcolor="#FFF8D9",
        bordercolor="#FFB347",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[-0.6, n - 0.4]),
        yaxis=dict(visible=False, range=[-1.0, 1.2]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Train page: FL topology animation (N clients + 1 server)
# ---------------------------------------------------------------------------

_SERVER_COLOR     = "#1a3d7c"
_SERVER_ACTIVE    = "#FFB347"
_CLIENT_COLOR     = "#55A868"
_CLIENT_ACTIVE    = "#FF6B6B"
_BROADCAST_COLOR  = "#4C72B0"
_UPLOAD_COLOR     = "#DD8452"


def _client_positions(n: int) -> List[tuple]:
    """Evenly space clients along y=-1, x in [-1, 1]."""
    if n == 1:
        return [(0.0, -1.0)]
    return [(-1.0 + 2.0 * i / (n - 1), -1.0) for i in range(n)]


def fl_topology_frame(
    num_clients: int,
    phase: str,
    round_id: int,
    total_rounds: int,
    accuracy: Optional[float] = None,
    active_client: Optional[int] = None,
) -> go.Figure:
    """One frame of the FL round animation.

    Args:
        num_clients: number of clients to draw.
        phase: one of {"broadcast", "train", "upload", "aggregate", "done"}.
        round_id: current round number (1-indexed).
        total_rounds: total rounds.
        accuracy: optional latest test accuracy [0,1] to display.
        active_client: index of the client to highlight during the train phase.
    """
    server_xy = (0.0, 1.0)
    client_xys = _client_positions(num_clients)

    # ---- Colors per phase ----
    server_color = _SERVER_COLOR
    if phase == "aggregate":
        server_color = _SERVER_ACTIVE
    elif phase == "done":
        server_color = "#2E8B57"

    client_colors = []
    for i in range(num_clients):
        if phase == "train" and active_client == i:
            client_colors.append(_CLIENT_ACTIVE)
        else:
            client_colors.append(_CLIENT_COLOR)

    fig = go.Figure()

    # ---- Edges (arrows) ----
    if phase == "broadcast":
        for (cx, cy) in client_xys:
            fig.add_annotation(
                x=cx, y=cy, ax=server_xy[0], ay=server_xy[1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.3,
                arrowwidth=2.2, arrowcolor=_BROADCAST_COLOR,
                standoff=18, startstandoff=22,
            )
    elif phase == "upload":
        for (cx, cy) in client_xys:
            fig.add_annotation(
                x=server_xy[0], y=server_xy[1], ax=cx, ay=cy,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.3,
                arrowwidth=2.2, arrowcolor=_UPLOAD_COLOR,
                standoff=22, startstandoff=18,
            )

    # ---- Server node ----
    server_label = "Server"
    if phase == "aggregate":
        server_label = "Aggregating..."
    elif phase == "done":
        server_label = "Done"
    fig.add_trace(go.Scatter(
        x=[server_xy[0]], y=[server_xy[1]],
        mode="markers+text",
        marker=dict(size=110, color=server_color,
                    line=dict(color="#0a1f44", width=2)),
        text=[f"<b>{server_label}</b>"],
        textposition="middle center",
        textfont=dict(color="white", size=12),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ---- Client nodes ----
    cx_list = [p[0] for p in client_xys]
    cy_list = [p[1] for p in client_xys]
    client_labels = [f"<b>C{i}</b>" for i in range(num_clients)]
    sizes = [70] * num_clients
    if phase == "train" and active_client is not None:
        sizes[active_client] = 92
    fig.add_trace(go.Scatter(
        x=cx_list, y=cy_list,
        mode="markers+text",
        marker=dict(size=sizes, color=client_colors,
                    line=dict(color="#1a3d2c", width=2)),
        text=client_labels,
        textposition="middle center",
        textfont=dict(color="white", size=11),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ---- Phase banner ----
    phase_text = {
        "broadcast": "1. Server broadcasts global model -> all clients",
        "train":     f"2. Client {active_client} training locally on private data"
                     if active_client is not None
                     else "2. Clients training locally",
        "upload":    "3. Clients upload sparse gradients + SEPG proof -> server",
        "aggregate": "4. Server aggregates (FedAvg / Median / TrimMean)",
        "done":      "All rounds complete",
    }.get(phase, phase)

    acc_str = f" | Test acc: {accuracy:.1%}" if accuracy is not None else ""
    fig.add_annotation(
        x=0, y=1.85,
        text=f"<b>Round {round_id}/{total_rounds}</b> &nbsp;&nbsp; {phase_text}{acc_str}",
        showarrow=False,
        font=dict(size=13, color="#1a3d7c"),
        bgcolor="#F4F7FB",
        bordercolor="#1a3d7c",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[-1.35, 1.35]),
        yaxis=dict(visible=False, range=[-1.45, 2.15]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig
