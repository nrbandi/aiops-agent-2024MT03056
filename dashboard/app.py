"""
AIOps Agent — Viva Demo Dashboard
Streamlit-based live visualization of the Observe→Analyze→Decide→Act pipeline.
B. Nageshwar Rao | 2024MT03056 | BITS WILP M.Tech Cloud Computing
"""

import sys
import os
import time
import json
import yaml
import random
import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyze.zscore_filter import ZScoreFilter
from src.analyze.isolation_forest import IsolationForestScorer
from src.analyze.event_correlator import EventCorrelator
from src.decide.rule_engine import RuleEngine
from src.decide.recommendation_engine import RecommendationEngine
from src.act.action_log import ActionLog

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIOps Agent — 2024MT03056",
    page_icon="🤖",
    layout="wide",
)


# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("config/agent_config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_resource
def init_pipeline(cfg):
    return {
        "zscore": ZScoreFilter(cfg),
        "iforest": IsolationForestScorer(cfg),
        "correlator": EventCorrelator(cfg),
        "rule_eng": RuleEngine(cfg),
        "rec_eng": RecommendationEngine(cfg),
        "action_log": ActionLog(cfg),
    }


cfg = load_config()
pipeline = init_pipeline(cfg)

# ── Session state ─────────────────────────────────────────────────────────────
if "cycle" not in st.session_state:
    st.session_state.cycle = 0
if "metric_history" not in st.session_state:
    st.session_state.metric_history = []
if "anomaly_history" not in st.session_state:
    st.session_state.anomaly_history = []
if "events" not in st.session_state:
    st.session_state.events = []
if "running" not in st.session_state:
    st.session_state.running = False
if "last_event" not in st.session_state:
    st.session_state.last_event = None
if "last_recs" not in st.session_state:
    st.session_state.last_recs = []
if "last_window" not in st.session_state:
    st.session_state.last_window = []

METRICS = ["cpu_percent", "mem_percent", "disk_io_mbps", "net_mbps"]
METRIC_LABELS = {
    "cpu_percent": "CPU %",
    "mem_percent": "Memory %",
    "disk_io_mbps": "Disk I/O MB/s",
    "net_mbps": "Network MB/s",
}
SEVERITY_COLORS = {
    "CRITICAL": "#e74c3c",
    "HIGH": "#e67e22",
    "MEDIUM": "#f1c40f",
    "LOW": "#2ecc71",
}


def synthetic_window(cycle: int) -> list:
    rng = random.Random(cycle * 42)
    anomaly = (cycle % 5 == 0) and (cycle >= 10)
    window = []
    for _ in range(cfg["observe"]["window_size"]):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if anomaly:
            window.append(
                {
                    "timestamp": ts,
                    "cpu_percent": rng.uniform(85.0, 97.0),
                    "mem_percent": rng.uniform(82.0, 94.0),
                    "disk_io_mbps": rng.uniform(20.0, 60.0),
                    "net_mbps": rng.uniform(50.0, 200.0),
                }
            )
        else:
            window.append(
                {
                    "timestamp": ts,
                    "cpu_percent": rng.uniform(15.0, 40.0),
                    "mem_percent": rng.uniform(30.0, 50.0),
                    "disk_io_mbps": rng.uniform(5.0, 30.0),
                    "net_mbps": rng.uniform(10.0, 80.0),
                }
            )
    return window


def run_cycle():
    st.session_state.cycle += 1
    cycle = st.session_state.cycle
    window = synthetic_window(cycle)
    st.session_state.last_window = window

    # Record mean metric values for plotting
    means = {m: sum(s[m] for s in window) / len(window) for m in METRICS}
    means["cycle"] = cycle
    st.session_state.metric_history.append(means)

    # Pipeline
    zr = pipeline["zscore"].filter(window)
    ifr = pipeline["iforest"].score(window)
    ev = pipeline["correlator"].correlate(zr, ifr, window)

    score = ifr.get("anomaly_score", 0.0)
    st.session_state.anomaly_history.append(
        {
            "cycle": cycle,
            "score": score,
            "in_warmup": ifr.get("in_warmup", True),
        }
    )

    if ev:
        lw = st.session_state.last_window
        matches = pipeline["rule_eng"].match(ev, lw)
        recs = pipeline["rec_eng"].generate(
            matches,
            ev,
            environment=st.session_state.get("env", "production"),
            operator_role=st.session_state.get("role", "L2"),
        )
        wm = {
            "windows_seen": ifr.get("windows_seen"),
            "in_warmup": ifr.get("in_warmup"),
            "zscore_flagged": zr.get("flagged_metrics"),
            "if_score": score,
        }
        pipeline["action_log"].write(ev, recs, wm)
        st.session_state.last_event = ev
        st.session_state.last_recs = recs
        st.session_state.events.append(
            {
                "cycle": cycle,
                "severity": ev["severity_band"],
                "score": ev["severity_score"],
                "metrics": ", ".join(ev["contributing_metrics"]),
                "duration": ev["anomaly_duration_windows"],
                "action": recs[0]["action"][:80] + "..." if recs else "—",
            }
        )


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 AIOps Agent — Live Demo")
st.caption(
    "B. Nageshwar Rao | 2024MT03056 | BITS WILP M.Tech Cloud Computing | CCZG628T"
)
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    env = st.selectbox(
        "Environment", ["production", "staging", "development"], key="env"
    )
    role = st.selectbox("Operator Role", ["L1", "L2", "L3"], index=1, key="role")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Next Cycle", use_container_width=True):
            run_cycle()
    with col2:
        if st.button("⏩ Run 10", use_container_width=True):
            for _ in range(10):
                run_cycle()

    if st.button("🔄 Reset", use_container_width=True):
        for k in [
            "cycle",
            "metric_history",
            "anomaly_history",
            "events",
            "last_event",
            "last_recs",
            "last_window",
        ]:
            del st.session_state[k]
        (
            os.remove("data/logs/action_log.jsonl")
            if os.path.exists("data/logs/action_log.jsonl")
            else None
        )
        st.rerun()

    st.divider()
    st.metric("Cycles Run", st.session_state.cycle)
    st.metric("Anomalies Detected", len(st.session_state.events))
    if st.session_state.events:
        bands = [e["severity"] for e in st.session_state.events]
        st.metric("CRITICAL", bands.count("CRITICAL"))
        st.metric("HIGH", bands.count("HIGH"))
        st.metric("MEDIUM", bands.count("MEDIUM"))

    st.divider()
    st.caption("**Architecture**: Observe→Analyze→Decide→Act")
    st.caption("**Detection**: Z-score → Isolation Forest")
    st.caption("**Playbooks**: 15 (Section 5.5.1)")
    st.caption("**Scope**: Simulated Act layer (Section 2.4)")

# ── Main layout ───────────────────────────────────────────────────────────────
if not st.session_state.metric_history:
    st.info(
        "👈 Click **▶ Next Cycle** or **⏩ Run 10** in the sidebar to start the pipeline."
    )
    st.stop()

df_metrics = pd.DataFrame(st.session_state.metric_history)
df_anomaly = pd.DataFrame(st.session_state.anomaly_history)

# ── Row 1: Metric charts ──────────────────────────────────────────────────────
st.subheader("📡 Observe Layer — Live Telemetry")
c1, c2, c3, c4 = st.columns(4)

for col, metric, label in zip([c1, c2, c3, c4], METRICS, METRIC_LABELS.values()):
    with col:
        latest = df_metrics[metric].iloc[-1]
        delta = (
            df_metrics[metric].iloc[-1] - df_metrics[metric].iloc[-2]
            if len(df_metrics) > 1
            else 0
        )
        st.metric(label, f"{latest:.1f}", f"{delta:+.1f}")
        fig, ax = plt.subplots(figsize=(3, 1.2))
        ax.plot(df_metrics["cycle"], df_metrics[metric], color="#3498db", linewidth=1.5)
        ax.fill_between(
            df_metrics["cycle"], df_metrics[metric], alpha=0.15, color="#3498db"
        )
        ax.set_axis_off()
        fig.patch.set_alpha(0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()

# ── Row 2: Anomaly score + event panel ───────────────────────────────────────
st.subheader("🔍 Analyze Layer — Anomaly Score Timeline")

col_score, col_event = st.columns([2, 1])

with col_score:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    cycles = df_anomaly["cycle"].tolist()
    scores = df_anomaly["score"].tolist()

    ax.plot(cycles, scores, color="#95a5a6", linewidth=1, zorder=1)
    ax.axhline(
        0.50,
        color="#e67e22",
        linewidth=1,
        linestyle="--",
        label="Anomaly threshold (0.50)",
    )
    ax.fill_between(
        cycles,
        scores,
        0.50,
        where=[s > 0.50 for s in scores],
        alpha=0.3,
        color="#e74c3c",
        label="Anomalous zone",
    )

    # Mark detected events
    for ev in st.session_state.events:
        c = ev["cycle"]
        row = df_anomaly[df_anomaly["cycle"] == c]
        if not row.empty:
            color = SEVERITY_COLORS.get(ev["severity"], "#e74c3c")
            ax.scatter(c, row["score"].values[0], color=color, s=80, zorder=5)

    ax.set_xlabel("Pipeline Cycle", fontsize=9)
    ax.set_ylabel("IF Anomaly Score", fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.patch.set_alpha(0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with col_event:
    if st.session_state.last_event:
        ev = st.session_state.last_event
        band = ev["severity_band"]
        col = SEVERITY_COLORS.get(band, "#95a5a6")
        st.markdown(
            f"""
        <div style='border-left: 5px solid {col}; padding: 12px;
                    border-radius: 4px; background: #f8f9fa;'>
        <b style='color:{col}; font-size:18px;'>{band}</b><br>
        <b>Score:</b> {ev['severity_score']}<br>
        <b>Metrics:</b> {', '.join(ev['contributing_metrics'])}<br>
        <b>Duration:</b> {ev['anomaly_duration_windows']} windows<br>
        <b>Detected:</b> {ev['detection_timestamp'][:19]}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No anomaly detected yet.")

st.divider()

# ── Row 3: Decide layer — recommendations ────────────────────────────────────
st.subheader("🧠 Decide Layer — Recommendations")

if st.session_state.last_recs:
    for i, rec in enumerate(st.session_state.last_recs[:3]):
        band = rec["severity_band"]
        color = SEVERITY_COLORS.get(band, "#95a5a6")
        with st.expander(
            f"#{i+1} [{rec['playbook_id']}] {rec['playbook_name']} "
            f"| Match: {rec['match_score']} | Priority: {rec['priority_score']}",
            expanded=(i == 0),
        ):
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.markdown(f"**Severity:** `{band}`")
                st.markdown(f"**Role:** `{rec['operator_role']}`")
                st.markdown(f"**Environment:** `{rec['environment']}`")
                st.markdown(f"**Tags:** {', '.join(rec['tags'])}")
            with col_b:
                st.markdown("**Recommended Action:**")
                st.info(rec["action"])
else:
    st.info("Recommendations will appear once an anomaly is detected.")

st.divider()

# ── Row 4: Act layer — action log table ──────────────────────────────────────
st.subheader("📋 Act Layer — Simulated Action Log")

if st.session_state.events:
    df_events = pd.DataFrame(st.session_state.events)
    df_events.columns = [
        "Cycle",
        "Severity",
        "Score",
        "Metrics",
        "Duration(win)",
        "Top Action",
    ]

    def color_severity(val):
        colors = {
            "CRITICAL": "background-color: #fadbd8",
            "HIGH": "background-color: #fdebd0",
            "MEDIUM": "background-color: #fef9e7",
            "LOW": "background-color: #eafaf1",
        }
        return colors.get(val, "")

    st.dataframe(
        df_events.style.applymap(color_severity, subset=["Severity"]),
        use_container_width=True,
        height=300,
    )
    st.caption(
        "⚠️ Simulated Act layer — no live infrastructure modified (Section 2.4, Scope of Work)"
    )
else:
    st.info("Action log entries will appear once anomalies are detected.")
