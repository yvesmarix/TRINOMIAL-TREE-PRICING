# Streamlit UI pour TRINOMIAL-TREE-PRICING
# -------------------------------------------------------------
from __future__ import annotations

import datetime as dt
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# --- Imports directs depuis le package `pricing` ---
REPO_SRC = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from pricing.market import Market
from pricing.option import Option
from pricing.trinomial_tree import TrinomialTree
from pricing.blackscholes import BlackScholesPricer
from pricing import convergence

DAY_COUNT: float = 365.0  # ACT/365F


def yearfrac(start: dt.date, end: dt.date, basis: float = DAY_COUNT) -> float:
    """Fraction d'année ACT/365F (>=0)."""
    return max(0.0, (end - start).days / basis)


@dataclass
class AppInputs:
    pricing_date: dt.date
    maturity: dt.date
    S0: float
    K: float
    r: float
    sigma: float
    option_type: str  # 'call' | 'put'
    option_class: str  # 'european' | 'american'
    dividend: float
    dividend_date: Optional[dt.date]
    N: int
    pruning: bool
    epsilon: float
    conv_min: int
    conv_max: int
    conv_step: int
    curve_pts: int
    curve_span_pct: float


# ---------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def bs_price_cached(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    opt_type: str,
    dividend: float = 0.0,
    dividend_date: Optional[dt.date] = None,
) -> float:
    """Calcule le prix Black-Scholes avec cache."""
    bs = BlackScholesPricer(
        S,
        K,
        T,
        r,
        sigma,
        opt_type,
        dividend=dividend,
        dividend_date=dividend_date,
    )
    return float(bs.price())


def build_market_option(inputs: AppInputs) -> Tuple[Market, Option, float]:
    """Construit Market, Option et T à partir des inputs."""
    market = Market(
        inputs.S0,
        inputs.r,
        inputs.sigma,
        dividend=inputs.dividend,
        dividend_date=inputs.dividend_date,
    )
    option = Option(
        inputs.K,
        inputs.maturity,
        inputs.option_type,
        inputs.option_class,
    )
    T: float = yearfrac(inputs.pricing_date, inputs.maturity)
    return market, option, T


def tree_price(
    market: Market,
    N: int,
    pruning: bool,
    epsilon: float,
    pricing_date: dt.date,
    option: Option,
) -> Tuple[float, float]:
    """Calcule le prix via l'arbre trinomial et mesure le temps."""
    t0: float = time.perf_counter()
    tree = TrinomialTree(
        market,
        N=N,
        pruning=pruning,
        epsilon=epsilon,
        pricing_date=pricing_date,
    )
    price_val: float = float(tree.price(option))
    elapsed: float = time.perf_counter() - t0
    return price_val, elapsed


def _build_tree_for_greeks(
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> TrinomialTree:
    tree = TrinomialTree(
        market,
        N=inputs.N,
        pruning=inputs.pruning,
        epsilon=inputs.epsilon,
        pricing_date=inputs.pricing_date,
    )
    _ = tree.price(option, compute_greeks=True)
    return tree


def _greeks_from_tree(
    tree: TrinomialTree,
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> Dict[str, float]:
    delta: float = float(tree.delta())
    gamma: float = float(tree.gamma())
    vega: float = float(tree.vega(option))
    rho: float = float(tree.rho(option))
    theta_day: float = compute_theta(tree, market, option, inputs)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta_day": theta_day,
        "vega": vega,
        "rho": rho,
    }


def compute_tree_and_greeks(
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> Tuple[TrinomialTree, Dict[str, float]]:
    """Construit l'arbre et renvoie aussi les grecs."""
    tree_greeks = _build_tree_for_greeks(market, option, inputs)
    greeks = _greeks_from_tree(tree_greeks, market, option, inputs)
    return tree_greeks, greeks


def compute_theta(
    tree_greeks: TrinomialTree,
    market: Market,
    option: Option,
    inputs: AppInputs,
) -> float:
    """Theta (par jour) via différence finie sur +1 jour."""
    tree_next = TrinomialTree(
        market,
        N=inputs.N,
        pruning=inputs.pruning,
        epsilon=inputs.epsilon,
        pricing_date=inputs.pricing_date + dt.timedelta(days=1),
    )
    v0: float = float(tree_greeks.root.option_value)  # type: ignore[arg-type]
    v1: float = float(tree_next.price(option))
    return v1 - v0


def _strike_grid(inputs: AppInputs) -> np.ndarray:
    span: float = inputs.curve_span_pct / 100.0
    return np.linspace(
        inputs.K * (1.0 - span),
        inputs.K * (1.0 + span),
        inputs.curve_pts,
    )


def price_vs_strike(inputs: AppInputs) -> pd.DataFrame:
    """Calcule les prix (arbre et BS) pour une gamme de strikes."""
    strikes: np.ndarray = _strike_grid(inputs)
    market, _, T = build_market_option(inputs)
    rows: List[Tuple[float, float, float]] = []

    for k in strikes:
        opt_k = Option(k, inputs.maturity, inputs.option_type, inputs.option_class)
        tree_val, _ = tree_price(
            market,
            inputs.N,
            inputs.pruning,
            inputs.epsilon,
            inputs.pricing_date,
            opt_k,
        )
        bs_val: float
        if inputs.option_class.lower() == "european":
            bs_val = compute_bs_price(inputs, float(k), T)
        else:
            bs_val = float("nan")
        rows.append((float(k), tree_val, bs_val))

    return (
        pd.DataFrame(rows, columns=["Strike", "Tree", "BlackScholes"])
        .set_index("Strike")
    )


def compute_bs_price(inputs: AppInputs, strike: float, T: float) -> float:
    """Calcule le prix Black-Scholes pour un strike donné."""
    return bs_price_cached(
        inputs.S0,
        strike,
        T,
        inputs.r,
        inputs.sigma,
        inputs.option_type,
        dividend=inputs.dividend,
        dividend_date=inputs.dividend_date,
    )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Trinomial Tree Pricer", layout="wide")
st.title("📈 Trinomial Tree & Black–Scholes Pricer")
st.caption("Paramétrez l’arbre, comparez à Black–Scholes, et explorez graphiques/convergences.")

with st.sidebar:
    st.header("Paramètres")

    today: dt.date = dt.date.today()
    pricing_date: dt.date = st.date_input("Pricing date", value=today)
    maturity: dt.date = st.date_input("Maturity", value=today + dt.timedelta(days=180))

    S0: float = st.number_input("Spot S0", min_value=0.0, value=100.0)
    K: float = st.number_input("Strike K", min_value=0.0, value=100.0)
    r: float = st.number_input("Rate r", value=0.02, step=0.002, format="%0.6f")
    sigma: float = st.number_input("Volatility σ", min_value=0.0001, value=0.20, step=0.01, format="%0.6f")

    option_type: str = st.selectbox("Option type", ["call", "put"], index=0)
    option_class: str = st.selectbox("Style", ["european", "american"], index=0)

    has_div: bool = st.toggle("Dividende discret ?", value=False)
    dividend: float = 0.0
    dividend_date: Optional[dt.date] = None
    if has_div:
        dividend = st.number_input("Montant dividende", min_value=0.0, value=0.0)
        div_date: dt.date = st.date_input("Date dividende", value=today + dt.timedelta(days=90))
        dividend_date = div_date

    st.subheader("Arbre trinomial")
    N: int = st.number_input("Nombre d’étapes N", min_value=1, max_value=2000, value=100, step=10)
    pruning: bool = st.toggle("Pruning", value=True)
    epsilon: float = st.number_input(
        "Epsilon (tolérance)",
        min_value=1e-12,
        value=1e-7,
        step=1e-7,
        format="%0.1e",
    )

    st.subheader("Convergence & Courbe")
    conv_min: int = st.number_input("Convergence: N min", min_value=1, max_value=5000, value=50, step=5)
    conv_max: int = st.number_input("Convergence: N max", min_value=1, max_value=5000, value=400, step=10)
    conv_step: int = st.number_input("Convergence: pas", min_value=1, max_value=1000, value=25, step=1)

    curve_pts: int = st.slider("Points pour courbe Strike", min_value=5, max_value=101, value=41, step=2)
    curve_span_pct: float = st.slider("Écart autour de K (%)", min_value=5, max_value=100, value=30, step=5)

    st.subheader("Affichage de l’arbre")
    max_depth: int = st.number_input("Profondeur max (None=tout)", min_value=1, max_value=5000, value=min(50, N), step=1)
    proba_min: float = st.number_input("Proba min pour afficher un nœud", min_value=0.0, value=1e-6, format="%0.1e")
    percentile_clip: int = st.slider("Clip vertical (percentile)", min_value=0, max_value=20, value=0, step=1)
    edge_alpha: float = st.slider("Transparence des arêtes", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    linewidth: float = st.slider("Épaisseur des arêtes", min_value=0.1, max_value=2.0, value=0.4, step=0.1)

    run: bool = st.button("▶️ Calculer / Mettre à jour", type="primary")
    plot_btn: bool = st.button("🌳 Tracer l’arbre")

inputs = AppInputs(
    pricing_date=pricing_date,
    maturity=maturity,
    S0=float(S0),
    K=float(K),
    r=float(r),
    sigma=float(sigma),
    option_type=option_type,
    option_class=option_class,
    dividend=float(dividend),
    dividend_date=dividend_date,
    N=int(N),
    pruning=bool(pruning),
    epsilon=float(epsilon),
    conv_min=int(conv_min),
    conv_max=int(conv_max),
    conv_step=int(conv_step),
    curve_pts=int(curve_pts),
    curve_span_pct=float(curve_span_pct),
)

colA, colB, colC = st.columns([1, 1, 1])

market, option, T = build_market_option(inputs)

with colA:
    st.subheader("Prix – Arbre trinomial")
    tree_price_val, t_sec = tree_price(
        market,
        inputs.N,
        inputs.pruning,
        inputs.epsilon,
        inputs.pricing_date,
        option,
    )
    st.metric("Tree price", f"{tree_price_val:,.6f}", help=f"Temps de calcul: {t_sec:.3f} s, N={inputs.N}")

with colB:
    st.subheader("Prix – Black–Scholes (européen)")
    if inputs.option_class.lower() == "european":
        bs_val: float = bs_price_cached(
            inputs.S0,
            inputs.K,
            T,
            inputs.r,
            inputs.sigma,
            inputs.option_type,
            dividend=inputs.dividend,
            dividend_date=inputs.dividend_date,
        )
        st.metric("BS price", f"{bs_val:,.6f}")
        st.caption("Comparaison valide pour options européennes.")
    else:
        st.caption("Black–Scholes n’est affiché que pour les options **européennes**.")
        bs_val = float("nan")

with colC:
    st.subheader("Greeks – Arbre & BS")
    tree_greeks, g_tree = compute_tree_and_greeks(market, option, inputs)

    if inputs.option_class.lower() == "european":
        bs = BlackScholesPricer(
            inputs.S0,
            inputs.K,
            T,
            inputs.r,
            inputs.sigma,
            inputs.option_type,
            dividend=inputs.dividend,
            dividend_date=inputs.dividend_date,
        )
        g_bs = bs.greeks()
        bs_delta = g_bs.get("delta", np.nan)
        bs_gamma = g_bs.get("gamma", np.nan)
        bs_theta = g_bs.get("theta", np.nan)
        bs_vega = g_bs.get("vega", np.nan)
        bs_rho = g_bs.get("rho", np.nan)
    else:
        bs_delta = bs_gamma = bs_theta = bs_vega = bs_rho = np.nan

    greek_df = pd.DataFrame(
        {
            "Tree": [
                g_tree["delta"],
                g_tree["gamma"],
                g_tree["theta_day"],
                g_tree["vega"],
                g_tree["rho"],
            ],
            "Black–Scholes": [bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho],
        },
        index=[
            "Delta",
            "Gamma",
            "Theta (par jour)",
            "Vega (/% vol)",
            "Rho (/% taux)",
        ],
    )
    st.dataframe(greek_df.style.format("{:.6f}"))

# -------- Visualisation de l’arbre --------
st.markdown("---")
st.subheader("Visualisation de l’arbre")

tree = TrinomialTree(
    market,
    N=inputs.N,
    pruning=inputs.pruning,
    epsilon=inputs.epsilon,
    pricing_date=inputs.pricing_date,
)
_ = tree.price(option, build_tree=True)
tree.plot_tree(
    max_depth=max_depth,
    proba_min=proba_min,
    percentile_clip=float(percentile_clip),
    edge_alpha=edge_alpha,
    linewidth=linewidth,
)
fig = plt.gcf()
st.pyplot(fig, clear_figure=True)

# --- Convergence (prix vs. nombre d’étapes) ---
st.markdown("---")
st.subheader("Convergence (prix vs. nombre d’étapes)")

prev_ids = set(plt.get_fignums())
convergence.bs_convergence_by_step(
    market,
    option,
    max_n=int(inputs.conv_max),
    step=int(max(1, inputs.conv_step)),
    pruning=inputs.pruning,
    epsilon=inputs.epsilon,
)
new_ids = [fid for fid in plt.get_fignums() if fid not in prev_ids]
for fid in new_ids:
    st.pyplot(plt.figure(fid), clear_figure=False)

# --- Benchmark temps d'exécution ---
st.markdown("---")
st.subheader("Benchmark temps d’exécution (price() vs N)")

N_vals_runtime = np.logspace(
    np.log10(max(5, inputs.conv_min)),
    np.log10(max(inputs.conv_max, inputs.conv_min + 1)),
    8,
    dtype=int,
)

prev_ids = set(plt.get_fignums())
convergence.plot_runtime_vs_steps(
    market,
    option,
    N_values=N_vals_runtime,
    method="backward",
    build_tree=True,
    compute_greeks=False,
)
new_ids = [fid for fid in plt.get_fignums() if fid not in prev_ids]
for fid in new_ids:
    st.pyplot(plt.figure(fid), clear_figure=False)

# --- Courbe prix vs Strike ---
st.markdown("---")
st.subheader("Courbe de prix en fonction du strike — BS vs Arbre")

span = inputs.curve_span_pct / 100.0
strikes = np.linspace(
    inputs.K * (1.0 - span),
    inputs.K * (1.0 + span),
    inputs.curve_pts,
)

prev_ids = set(plt.get_fignums())
convergence.bs_convergence_by_strike(
    market,
    option,
    strikes,
    n_steps=int(inputs.N),
    pruning=inputs.pruning,
    epsilon=inputs.epsilon,
)
new_ids = [fid for fid in plt.get_fignums() if fid not in prev_ids]
for fid in new_ids:
    st.pyplot(plt.figure(fid), clear_figure=False)

st.markdown(
    """
    **Notes**
    - La maturité est transformée en *T* (années) avec une base ACT/365.
    - Le pricer Black–Scholes est affiché uniquement pour les options **européennes**.
    - Le dividende discret est passé aux deux pricers s’il est renseigné.
    - Grecs BS analytiques; côté arbre : Δ/Γ via nœuds racine, vega/rho en FD.
    """
)
