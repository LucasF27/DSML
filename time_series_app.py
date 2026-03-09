"""
Forecast Playground
===================
Interactive Streamlit app for teaching time-series forecasting.

Models:  Mean · Naïve · Seasonal naïve · Moving average · Linear regression (lags)
Data:    Synthetic monthly mean temperatures for Nottingham, Jan 2015 – Dec 2024

Run with:
    streamlit run time_series_app.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecast Playground",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Forecast Playground")
st.caption(
    "Compare simple forecasting models on the same seasonal series. "
    "Adjust the controls in the sidebar — MAE, RMSE, and the chart update instantly."
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEASON = 12  # monthly data → annual cycle

# ── Synthetic dataset ─────────────────────────────────────────────────────────
@st.cache_data
def make_series() -> pd.Series:
    """Synthetic monthly mean temperatures for Nottingham, 2015–2024.

    Construction:
        base 10.2 °C  +  gentle warming trend  +  annual sinusoidal cycle
        (peak ≈ July, trough ≈ January)         +  Gaussian noise σ = 0.75 °C
    """
    rng = np.random.default_rng(42)
    n   = 120                                           # 10 years × 12 months
    idx = pd.date_range("2015-01", periods=n, freq="MS")
    t   = np.arange(n)
    vals = (
        10.2
        + 0.018 * t                                    # +0.22 °C / decade trend
        + 7.5 * np.sin(2 * np.pi * (t - 3) / 12)     # annual cycle, peak July
        + rng.normal(0, 0.75, n)                       # measurement noise
    )
    return pd.Series(np.round(vals, 2), index=idx, name="Temp (°C)")


series = make_series()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    model_name: str = st.selectbox(
        "Model type",
        options=[
            "Mean",
            "Naïve",
            "Seasonal naïve",
            "Moving average",
            "Linear regression (with lags)",
            "Random Forest (with lags)",
        ],
    )

    split_pct: int = st.slider(
        "Train / test split (%)",
        min_value=50,
        max_value=90,
        value=80,
        step=5,
        help="Percentage of data used for training. The rest becomes the test set.",
    )

    window: int = st.slider(
        "Window size  (MA window / lag count)",
        min_value=1,
        max_value=24,
        value=3,
        help=(
            "Rolling window for the moving average, "
            "or the number of lag features used by linear regression."
        ),
    )

    horizon: int = st.slider(
        "Future forecast horizon (months)",
        min_value=1,
        max_value=24,
        value=6,
        help="How many months beyond the dataset end to project forward.",
    )

    st.markdown("---")
    st.info(
        "**Dataset:** Synthetic monthly mean temperatures for Nottingham "
        "(2015–2024).  "
        "Realistic UK seasonal pattern: ~2 °C (Jan) → ~18 °C (Jul), "
        "plus a slight warming trend.",
        icon="🌡️",
    )

# ── Train / test split ────────────────────────────────────────────────────────
n_train        = int(len(series) * split_pct / 100)
train: pd.Series = series.iloc[:n_train]
test:  pd.Series = series.iloc[n_train:]

# ── Model factory ─────────────────────────────────────────────────────────────
def make_predictor(name: str, w: int):
    """
    Return  fn(history: list[float]) -> float

    The predictor is constructed once; any fitting uses *train only*.
    Walk-forward evaluation calls fn repeatedly, passing a history list
    that is extended with *actual* test values after each step.
    """
    if name == "Mean":
        mu = float(train.mean())
        return lambda h: mu                                          # constant

    if name == "Naïve":
        return lambda h: float(h[-1])                               # last value

    if name == "Seasonal naïve":
        return lambda h: float(h[-SEASON]) if len(h) >= SEASON else float(h[-1])

    if name == "Moving average":
        return lambda h: float(np.mean(h[-w:]))

    # ── Linear regression with lag features ──────────────────────────────────
    vals = list(train.values)
    if name == "Linear regression (with lags)":
        if len(vals) > w:
            X_tr = np.array([vals[i : i + w] for i in range(len(vals) - w)])
            y_tr = np.array(vals[w:])
            lr   = LinearRegression().fit(X_tr, y_tr)
            return lambda h: float(                                      # noqa: E731
                lr.predict(np.array(h[-w:]).reshape(1, -1))[0]
            )
        return lambda h: float(np.mean(h[-w:]))

    # ── Random Forest with lag features ──────────────────────────────────────
    if len(vals) > w:
        X_tr = np.array([vals[i : i + w] for i in range(len(vals) - w)])
        y_tr = np.array(vals[w:])
        rf   = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_tr, y_tr)
        return lambda h: float(                                          # noqa: E731
            rf.predict(np.array(h[-w:]).reshape(1, -1))[0]
        )
    # Fallback: window ≥ training length — degrade to moving average
    return lambda h: float(np.mean(h[-w:]))


# ── Walk-forward 1-step evaluation ────────────────────────────────────────────
def walk_forward(pred_fn, train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    """One-step-ahead forecasts; history is updated with actual observations."""
    history = list(train_vals)
    preds   = []
    for actual in test_vals:
        preds.append(pred_fn(history))
        history.append(float(actual))
    return np.array(preds)


# ── Recursive multi-step projection ──────────────────────────────────────────
def project_future(pred_fn, base: pd.Series, h: int) -> pd.Series:
    """
    Forecast h steps beyond the end of `base` by recursively feeding
    each prediction back into the history buffer.
    """
    buf         = list(base.values)
    future_vals = []
    for _ in range(h):
        p = pred_fn(buf)
        future_vals.append(p)
        buf.append(p)
    future_idx = pd.date_range(base.index[-1], periods=h + 1, freq="MS")[1:]
    return pd.Series(np.round(future_vals, 3), index=future_idx)


# ── Compute forecasts ─────────────────────────────────────────────────────────
pred_fn    = make_predictor(model_name, window)
test_preds = walk_forward(pred_fn, train.values, test.values)

# Seasonal-naïve baseline — always computed for side-by-side comparison
sn_fn    = make_predictor("Seasonal naïve", SEASON)
sn_preds = walk_forward(sn_fn, train.values, test.values)

# Future projection — full series used as history, predictions feed back in
future_ser = project_future(pred_fn, series, horizon)

# ── Error metrics ─────────────────────────────────────────────────────────────
errors    = test.values - test_preds
mae       = float(np.mean(np.abs(errors)))
rmse      = math.sqrt(float(np.mean(errors ** 2)))

sn_errors = test.values - sn_preds
sn_mae    = float(np.mean(np.abs(sn_errors)))
sn_rmse   = math.sqrt(float(np.mean(sn_errors ** 2)))

delta_mae = mae - sn_mae

# ── Figure: time-series + residuals ──────────────────────────────────────────
fig, (ax, ax_r) = plt.subplots(
    2, 1,
    figsize=(13, 7),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=False,
)

# — Main panel ——————————————————————————————————————————————————————————————
ax.plot(train.index, train.values,
        color="#2166ac", lw=1.9, label="Train")
ax.plot(test.index, test.values,
        color="#1a9850", lw=1.9, label="Test (actual)")
ax.plot(test.index, test_preds,
        color="#d73027", lw=2.1, ls="--",
        label=f"{model_name}  [MAE {mae:.2f} °C]")

if model_name != "Seasonal naïve":
    ax.plot(test.index, sn_preds,
            color="#f46d43", lw=1.4, ls=":",
            label=f"Seasonal naïve  [MAE {sn_mae:.2f} °C]")

ax.plot(future_ser.index, future_ser.values,
        color="#762a83", lw=1.7, ls="-.", marker="o", ms=5,
        label=f"Future projection ({horizon} months)")

ax.axvline(train.index[-1], color="gray", lw=1.5, ls="--", alpha=0.8,
           label="Train / test split")
ax.axvspan(test.index[0], test.index[-1],
           alpha=0.07, color="#1a9850", label="Test region")

ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title(
    "Monthly Mean Temperature – Nottingham  (synthetic, 2015–2024)",
    fontsize=12,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(visible=True, alpha=0.25)

# — Residuals panel ————————————————————————————————————————————————————————
ax_r.bar(test.index, errors, width=20, color="#d73027", alpha=0.55)
ax_r.axhline(0, color="black", lw=0.9)
ax_r.set_ylabel("Residual (°C)", fontsize=10)
ax_r.set_xlabel("Date", fontsize=10)
ax_r.grid(visible=True, alpha=0.25)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Metric strip ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("MAE",                 f"{mae:.3f} °C")
c2.metric("RMSE",                f"{rmse:.3f} °C")
c3.metric("Seasonal-naïve MAE",  f"{sn_mae:.3f} °C")
c4.metric("Seasonal-naïve RMSE", f"{sn_rmse:.3f} °C")
c5.metric("Test size",           f"{len(test)} months")

# — Verdict banner ————————————————————————————————————————————————————————————
if model_name == "Seasonal naïve":
    st.info("📌 You are currently viewing the **seasonal-naïve** baseline itself.")
elif mae < sn_mae:
    st.success(
        f"✅ **{model_name}** beats the seasonal-naïve baseline "
        f"(MAE {mae:.3f} < {sn_mae:.3f} °C, Δ = {delta_mae:+.3f} °C)."
    )
else:
    st.warning(
        f"⚠️ **{model_name}** does **not** beat the seasonal-naïve baseline "
        f"(MAE {mae:.3f} ≥ {sn_mae:.3f} °C, Δ = {delta_mae:+.3f} °C). "
        "The simplest seasonal model wins!"
    )

# ── Teaching notes ────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("📚 Teaching notes & suggested experiments", expanded=True):

    _notes = {
        "Mean": (
            "The **mean model** predicts the constant training-set average for every "
            "future step.  It cannot represent trend or seasonality, so expect "
            "large errors near summer peaks and winter troughs.  "
            "Use it as an absolute performance floor — any reasonable model "
            "should beat it."
        ),
        "Naïve": (
            "The **naïve model** repeats the last observed value.  "
            "It is surprisingly competitive on random-walk series (e.g. daily stock prices) "
            "but struggles here because it completely ignores the annual cycle: "
            "after a warm July it predicts a warm August, missing the autumn drop."
        ),
        "Seasonal naïve": (
            "The **seasonal-naïve model** copies the value from exactly 12 months ago.  "
            "On strongly seasonal monthly data this is a formidable baseline — it "
            "captures the annual cycle with **zero parameter estimation**.  "
            "It is the standard benchmark in the M-competition literature.  "
            "Try every other model to see if you can beat it!"
        ),
        "Moving average": (
            f"The **moving-average model** (window = **{window}**) predicts the mean of "
            f"the most recent {window} observation(s).  \n\n"
            "Key insight: as *w* approaches 12, the model averages across full seasons, "
            "washing out the seasonal signal and pushing MAE higher.  "
            "Very short windows are noisy; very long windows over-smooth and lag behind peaks."
        ),
        "Linear regression (with lags)": (
            f"**Linear regression** fitted with **{window} lag feature(s)**.  "
            "The model is fitted *once* on the training set; test-set predictions "
            "use walk-forward 1-step evaluation with actual previous values as inputs.  \n\n"
            "Watch out for:  \n"
            "- **Too many lags, small training set** → overfitting → poor generalisation.  \n"
            "- **Lags not aligned to seasonality** (e.g. lags 1–3 only) → misses "
            "the annual cycle entirely.  \n"
            "Typical finding: it rarely beats seasonal-naïve without deliberate "
            "feature engineering (e.g. including lag 12)."
        ),
        "Random Forest (with lags)": (
            f"**Random Forest** (200 trees) fitted with **{window} lag feature(s)**.  "
            "The model is trained once on the training set and evaluated via "
            "walk-forward 1-step prediction using actual previous values.  \n\n"
            "Key insights:  \n"
            "- Random Forests can capture **non-linear** relationships between lags "
            "and the next value, unlike linear regression.  \n"
            "- They are relatively **robust to overfitting** thanks to averaging across "
            "many trees, but still benefit from feature engineering.  \n"
            "- Setting **lags = 12** aligns the feature window to the full annual cycle "
            "and typically produces the best results on this dataset.  \n"
            "- Despite their power, they cannot extrapolate beyond the training range, "
            "so future projections may plateau at typical historical values."
        ),
    }

    st.markdown(_notes[model_name])

    st.markdown("### 🔬 Suggested experiments")
    st.markdown(
        """
| Experiment | What to observe |
|---|---|
| Model = *Seasonal naïve*, vary split % | How stable is the baseline across different test sizes? |
| Model = *Moving average*, increase window from 3 → 12 | MAE climbs as the seasonal signal is averaged out (over-smoothing). |
| Model = *Moving average*, set window = 1 | Equivalent to the naïve model — verify that MAE matches. |
| Model = *Linear regression*, lags = 12, split = 80 % | Aligning one lag to the full season — does it beat seasonal naïve now? |
| Model = *Linear regression*, lags = 20, split = 50 % | Shrink training data, inflate features — watch for overfitting. |
| Compare *Mean* vs *Naïve*, split = 90 % | Which wins when the test period is very short (≈12 points)? |
| All models, split = 70 % | Rank every model — which consistently wins? |
"""
    )

    st.markdown("### 💡 Key take-aways")
    st.markdown(
        """
- A hard-to-beat baseline is the first thing to establish before trying anything "fancy".
- **Seasonal naïve is the baseline** for seasonal monthly data — not the mean, not naïve.
- Increasing model complexity ≠ lower error. More parameters need more data.
- Always evaluate on a *held-out* test set; in-sample fit is misleading for forecasting.
"""
    )
