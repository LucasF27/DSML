"""
Anomaly Detector Playground
============================
Interactive Streamlit app for teaching anomaly detection in time series.

Methods:  Absolute threshold · Rolling z-score
Data:    Synthetic heart-rate-like sensor trace (480 minutes) with injected spikes

Run with:
    streamlit run anomaly_app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Anomaly Detector Playground")
st.caption(
    "Time-series analysis is not only about forecasting — it's also about "
    "detecting unusual events.  Adjust the controls in the sidebar and watch "
    "how the algorithm catches (or misses) injected anomalies."
)

# ── Constants ─────────────────────────────────────────────────────────────────
N            = 480   # 8 hours at 1-minute resolution
N_ANOMALIES  = 12    # injected spikes
TOLERANCE    = 2     # ±2 minutes window when matching detections to ground truth

# ── Synthetic dataset ─────────────────────────────────────────────────────────
@st.cache_data
def make_signal():
    """Synthetic heart-rate sensor trace, 480 minutes (08:00–16:00).

    Construction:
        base 72 bpm  +  slow 160-min oscillation (exercise bout)
                     +  faster 40-min oscillation (breathing modulation)
                     +  Gaussian noise σ = 1.5 bpm
        N_ANOMALIES sudden positive/negative spikes injected at random positions
        (spaced ≥ 15 minutes apart to avoid overlap).
    """
    rng = np.random.default_rng(99)
    t   = np.arange(N)

    signal = (
        72
        + 8  * np.sin(2 * np.pi * t / 160)   # slow exercise cycle
        + 3  * np.sin(2 * np.pi * t / 40)    # faster oscillation
        + rng.normal(0, 1.5, N)              # measurement noise
    )

    # Inject anomalies: well-spaced, random sign, large magnitude
    anom_idx: list[int] = []
    candidates = list(range(20, N - 20))
    while len(anom_idx) < N_ANOMALIES:
        idx = int(rng.choice(candidates))
        if all(abs(idx - a) > 15 for a in anom_idx):
            anom_idx.append(idx)

    anom_idx.sort()
    signs      = rng.choice([-1, 1], size=N_ANOMALIES)
    magnitudes = signs * rng.uniform(18, 35, N_ANOMALIES)
    signal[anom_idx] += magnitudes

    ts = pd.Series(
        np.round(signal, 1),
        index=pd.date_range("2024-01-15 08:00", periods=N, freq="min"),
        name="Heart rate (bpm)",
    )
    return ts, anom_idx, magnitudes


series, true_anom_idx, true_magnitudes = make_signal()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    method: str = st.selectbox(
        "Detection method",
        options=["Absolute threshold", "Rolling z-score"],
    )

    window: int = st.slider(
        "Rolling window (minutes)",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
        help="Window used to compute the rolling mean and standard deviation.",
    )

    if method == "Absolute threshold":
        upper_thresh: float = st.slider(
            "Upper threshold (bpm)",
            min_value=80,
            max_value=130,
            value=100,
            step=1,
            help="Flag readings above this value.",
        )
        lower_thresh: float = st.slider(
            "Lower threshold (bpm)",
            min_value=30,
            max_value=70,
            value=50,
            step=1,
            help="Flag readings below this value.",
        )
    else:
        z_thresh: float = st.slider(
            "Z-score threshold (σ)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help=(
                "Flag points whose deviation from the rolling mean "
                "exceeds this many standard deviations."
            ),
        )

    show_truth: bool = st.toggle(
        "Show ground-truth anomalies",
        value=True,
        help="Toggle the green triangles that mark the injected spikes.",
    )

    st.markdown("---")
    st.info(
        f"**Dataset:** Synthetic heart-rate sensor trace (480 min).  "
        f"Base ≈ 72 bpm + slow oscillations + noise.  "
        f"**{N_ANOMALIES} anomalies injected** at random positions.",
        icon="💓",
    )

# ── Rolling statistics ────────────────────────────────────────────────────────
roll_mean = series.rolling(window=window, center=True, min_periods=1).mean()
roll_std  = (
    series.rolling(window=window, center=True, min_periods=1)
    .std()
    .bfill()
    .ffill()
    .clip(lower=0.1)          # prevent division-by-zero
)
z_scores  = (series.values - roll_mean.values) / roll_std.values

# ── Detection ─────────────────────────────────────────────────────────────────
if method == "Absolute threshold":
    detected_mask = (series.values > upper_thresh) | (series.values < lower_thresh)
else:
    detected_mask = np.abs(z_scores) > z_thresh

detected_idx: list[int] = list(np.where(detected_mask)[0])

# ── Ground-truth comparison (within ±TOLERANCE steps) ────────────────────────
true_set = set(true_anom_idx)


def near_true(i: int) -> bool:
    return any(abs(i - t) <= TOLERANCE for t in true_set)


tp = sum(1 for i in detected_idx if near_true(i))
fp = sum(1 for i in detected_idx if not near_true(i))
fn = sum(
    1 for t in true_anom_idx
    if not any(abs(d - t) <= TOLERANCE for d in detected_idx)
)
tn = N - tp - fp - fn

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

# ── Figure: signal + z-score ──────────────────────────────────────────────────
fig, (ax, ax_z) = plt.subplots(
    2, 1,
    figsize=(14, 7),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)

# — Main panel ——————————————————————————————————————————————————————————————
ax.plot(series.index, series.values,
        color="#2166ac", lw=1.2, alpha=0.85, label="Sensor signal")
ax.plot(series.index, roll_mean.values,
        color="#4dac26", lw=1.5, ls="--", alpha=0.75,
        label=f"Rolling mean (w = {window} min)")

if method == "Absolute threshold":
    ax.axhline(upper_thresh, color="#d73027", lw=1.6, ls="--",
               label=f"Upper threshold ({upper_thresh} bpm)")
    ax.axhline(lower_thresh, color="#d73027", lw=1.6, ls="-.",
               label=f"Lower threshold ({lower_thresh} bpm)")
else:
    upper_band = roll_mean.values + z_thresh * roll_std.values
    lower_band = roll_mean.values - z_thresh * roll_std.values
    ax.fill_between(series.index, lower_band, upper_band,
                    color="#fddbc7", alpha=0.50,
                    label=f"±{z_thresh:.1f}σ band")
    ax.plot(series.index, upper_band, color="#f4a582", lw=0.9, ls="--")
    ax.plot(series.index, lower_band, color="#f4a582", lw=0.9, ls="--")

# True anomalies (ground truth, hideable)
if show_truth:
    true_df = series.iloc[true_anom_idx]
    ax.scatter(
        true_df.index, true_df.values,
        marker="^", s=90, color="#1a9850", zorder=4, alpha=0.92,
        label=f"Injected anomaly ({N_ANOMALIES})",
    )

# Detected anomalies
if detected_idx:
    det_df = series.iloc[detected_idx]
    ax.scatter(
        det_df.index, det_df.values,
        marker="o", s=60, color="#d73027", zorder=5, alpha=0.88,
        label=f"Detected ({len(detected_idx)})",
    )

ax.set_ylabel("Heart rate (bpm)", fontsize=11)
ax.set_title(
    "Synthetic Heart-Rate Sensor — 8 Hours (08:00–16:00, Jan 2024)",
    fontsize=12,
)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(visible=True, alpha=0.25)

# — Z-score panel ————————————————————————————————————————————————————————————
ax_z.plot(series.index, np.abs(z_scores),
          color="#762a83", lw=1.2, label="|z-score|")
if method == "Rolling z-score":
    ax_z.axhline(z_thresh, color="#d73027", lw=1.5, ls="--",
                 label=f"Threshold = {z_thresh:.1f}σ")
ax_z.set_ylabel("|z-score|", fontsize=10)
ax_z.set_xlabel("Time", fontsize=10)
ax_z.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_z.grid(visible=True, alpha=0.25)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Metric strip ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Detected",            len(detected_idx))
c2.metric("True positives (TP)", tp)
c3.metric("False positives (FP)", fp,
          delta=None if fp == 0 else f"-{fp} alarms",
          delta_color="inverse")
c4.metric("False negatives (FN)", fn,
          delta=None if fn == 0 else f"-{fn} missed",
          delta_color="inverse")
c5.metric("Precision",  f"{precision:.2f}")
c6.metric("Recall",     f"{recall:.2f}")
c7.metric("F1 score",   f"{f1:.3f}")

# — Verdict banner ————————————————————————————————————————————————————————————
if f1 >= 0.85:
    st.success(
        f"✅ Strong detection:  F1 = {f1:.3f}  "
        f"(Precision {precision:.2f}, Recall {recall:.2f})"
    )
elif recall < 0.5:
    st.warning(
        f"⚠️ Too many **missed anomalies** (Recall = {recall:.2f}). "
        "Lower the threshold to catch more spikes."
    )
elif precision < 0.5:
    st.warning(
        f"⚠️ Too many **false alarms** (Precision = {precision:.2f}). "
        "Raise the threshold to reduce noise triggers."
    )
else:
    st.info(
        f"ℹ️ F1 = {f1:.3f}  — reasonable but improvable. "
        "Try tuning both window and threshold together."
    )

# ── Teaching notes ────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("📚 Teaching notes & suggested experiments", expanded=True):

    _notes = {
        "Absolute threshold": (
            "The **absolute threshold** detector flags any reading that crosses a "
            "fixed upper or lower bound.\n\n"
            "- **Strengths:** instant, zero computation, easy to explain to operators.\n"
            "- **Weaknesses:** the safe range must be set by hand; if the signal's "
            "normal level drifts (e.g. during exercise), legitimate peaks get flagged "
            "(**false positives**) and deep troughs injected in a low-baseline period "
            "may slip under the lower threshold (**false negatives**).\n\n"
            "This mirrors legacy industrial alarms, simple wearable cut-offs, and "
            "basic business-rule fraud filters."
        ),
        "Rolling z-score": (
            r"The **rolling z-score** method tracks a local mean $\bar{x}_{t,w}$ and "
            r"standard deviation $\sigma_{t,w}$ over a sliding window, then flags"
            r" $|z_t| > k$, where"
            "\n\n"
            r"$$z_t = \frac{x_t - \bar{x}_{t,w}}{\sigma_{t,w}}$$"
            "\n\n"
            "- **Adaptive:** the reference level shifts with the signal, so slow "
            "trends don't cause perpetual false alarms.\n"
            "- **Short window trap:** a very short window incorporates the spike "
            "itself into the local std — inflating it and _hiding_ the anomaly "
            "(watch this happen at w = 5).\n"
            "- **Long window trap:** a very long window is slow to adapt and "
            "misses events that look 'local' but are not extreme globally.\n"
            "- Rule of thumb: $k = 3$ captures 99.7 % of a Gaussian — but sensor "
            "noise is rarely perfectly Gaussian."
        ),
    }
    st.markdown(_notes[method])

    st.markdown("### 🔬 Suggested experiments")
    st.markdown(
        """
| Experiment | What to observe |
|---|---|
| Method = *Absolute*, move upper threshold from 90 → 120 | FP drops but FN rises — the precision–recall trade-off. |
| Method = *Absolute*, set both thresholds very wide | All anomalies missed (FN = 12). Recall = 0. |
| Method = *Z-score*, lower σ from 3.0 → 1.5 | More detections, but FP explodes (over-sensitivity). |
| Method = *Z-score*, window = 5, σ = 2.5 | Short window inflates local std → some spikes go undetected. |
| Method = *Z-score*, window = 60, σ = 2.5 | Over-smoothed baseline → wide band → spikes hidden. |
| Method = *Z-score*, window = 15, σ = 2.5 | Sweet spot for this signal — compare F1 to other settings. |
| Toggle **Show ground truth** OFF | Pretend you don't know where anomalies were injected. Can you guess? |
"""
    )

    st.markdown("### 💡 Key take-aways")
    st.markdown(
        """
- Every anomaly detector involves a **precision–recall trade-off**:  
  catching more anomalies always risks more false alarms, and vice-versa.
- The cost asymmetry between FP and FN is **domain-driven**, not algorithmic:
"""
    )
    st.markdown(
        """
| Domain | Anomaly | Cost of **missing** it (FN) | Cost of **false alarm** (FP) |
|---|---|---|---|
| Cardiac monitoring | Arrhythmia | Patient harm / death | Unnecessary intervention |
| Manufacturing | Vibration spike | Equipment failure | Unnecessary maintenance stop |
| Cybersecurity | Unusual login | Data breach | Blocked legitimate user |
| Finance | Fraudulent transaction | Financial loss | Customer friction / churn |
"""
    )
    st.markdown(
        """
- Adaptive methods (z-score, residual IQR) generalise better than fixed thresholds  
  when the signal is **non-stationary** (trend, seasonality, level shifts).
- Real-world systems combine multiple detectors, multiple sensors, and human review —  
  automated alerts are a *first filter*, not a final verdict.
"""
    )
