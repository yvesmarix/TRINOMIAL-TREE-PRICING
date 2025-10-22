# Streamlit UI pour TRINOMIAL-TREE-PRICING
# -------------------------------------------------------------
# Cette app propose une interface Streamlit pour :
#  - Pricer des options europeennes/americaines via l'arbre trinomial
#  - Comparer le prix europeen avec Black–Scholes
#  - Tracer la convergence (vs. nombre d'etapes)
#  - Tracer des courbes prix vs. strike
# -------------------------------------------------------------

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import os
import sys

# --- Imports directs depuis le package `pricing` ---
# On s'assure que le chemin vers `src` est dans le sys.path
REPO_SRC = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from pricing.market import Market
from pricing.option import Option
from pricing.trinomial_tree import TrinomialTree
from pricing.blackscholes import BlackScholesPricer
from pricing import convergence

# ------------------------------ Constantes & Helpers ------------------------------
DAY_COUNT = 365.0  # ACT/365F

# Petite fonction utilitaire pour calculer la fraction d'annee
def yearfrac(start: dt.datetime, end: dt.datetime, basis: float = DAY_COUNT) -> float:
    """Fraction d'annee ACT/365F (>=0)."""
    return max(0.0, (end - start).days / basis)

# ------------------------------ Dataclass pour les inputs ------------------------------
@dataclass
class AppInputs:
    pricing_date: dt.datetime
    maturity: dt.datetime
    S0: float
    K: float
    r: float
    sigma: float
    option_type: str  # 'call' | 'put'
    option_class: str  # 'european' | 'american'
    dividend: float
    dividend_date: Optional[dt.datetime]
    N: int
    pruning: bool
    epsilon: float
    # Convergence / courbes
    conv_min: int
    conv_max: int
    conv_step: int
    curve_pts: int
    curve_span_pct: float

# ------------------------------ Fonctions de pricing ------------------------------

# Cache pour le prix Black-Scholes (evite de recalculer inutilement)
@st.cache_data(show_spinner=False)
def bs_price_cached(S: float, K: float, T: float, r: float, sigma: float, opt_type: str,
                    dividend: float = 0.0, dividend_date: Optional[dt.datetime] = None) -> float:
    """Calcule le prix Black-Scholes avec cache."""
    bs = BlackScholesPricer(S, K, T, r, sigma, opt_type, dividend=dividend, dividend_date=dividend_date)
    return float(bs.price())

# Fonction pour construire le marche et l'option
def build_market_option(inputs: AppInputs) -> tuple[Market, Option, float]:
    """Construit les objets Market et Option a partir des inputs."""
    market = Market(inputs.S0, inputs.r, inputs.sigma, dividend=inputs.dividend, dividend_date=inputs.dividend_date)
    option = Option(inputs.K, inputs.maturity, inputs.option_type, inputs.option_class)
    T = yearfrac(inputs.pricing_date, inputs.maturity)
    return market, option, T

# Fonction pour calculer le prix via l'arbre trinomial
def tree_price(market: Market, N: int, pruning: bool, epsilon: float, pricing_date: dt.datetime,
               option: Option) -> Tuple[float, float]:
    """Calcule le prix via l'arbre trinomial et mesure le temps de calcul."""
    t0 = time.perf_counter()
    tree = TrinomialTree(market, N=N, pruning=pruning, epsilon=epsilon, pricingDate=pricing_date)
    price_val = float(tree.price(option))
    elapsed = time.perf_counter() - t0
    return price_val, elapsed

# Fonction pour calculer les grecs via l'arbre trinomial
def compute_tree_and_greeks(market: Market, option: Option, inputs: AppInputs) -> tuple[TrinomialTree, dict]:
    """Construit l'arbre, calcule le prix et les grecs cote arbre."""
    tree_greeks = TrinomialTree(market, N=inputs.N, pruning=inputs.pruning,
                                epsilon=inputs.epsilon, pricingDate=inputs.pricing_date)
    _ = tree_greeks.price(option, compute_greeks=True)

    # Grecs disponibles cote arbre
    delta = float(tree_greeks.delta())
    gamma = float(tree_greeks.gamma())
    vega = float(tree_greeks.vega(option))
    rho = float(tree_greeks.rho(option))

    # Theta (par jour) via difference finie sur la date de pricing
    theta_day = compute_theta(tree_greeks, market, option, inputs)

    greeks = {
        "delta": delta,
        "gamma": gamma,
        "theta_day": theta_day,
        "vega": vega,
        "rho": rho,
    }
    return tree_greeks, greeks

# Fonction pour calculer Theta
def compute_theta(tree_greeks: TrinomialTree, market: Market, option: Option, inputs: AppInputs) -> float:
    """Calcule Theta par difference finie sur +1 jour."""
    tree_next = TrinomialTree(market, N=inputs.N, pruning=inputs.pruning, epsilon=inputs.epsilon,
                              pricingDate=inputs.pricing_date + dt.timedelta(days=1))
    v0 = float(tree_greeks.root.option_value)
    v1 = float(tree_next.price(option))
    return v1 - v0

# Fonction pour tracer les prix en fonction du strike
def price_vs_strike(inputs: AppInputs) -> pd.DataFrame:
    """Calcule les prix (arbre et BS) pour une gamme de strikes."""
    span = inputs.curve_span_pct / 100.0
    strikes = np.linspace(inputs.K * (1.0 - span), inputs.K * (1.0 + span), inputs.curve_pts)

    market, _, T = build_market_option(inputs)

    rows = []
    for k in strikes:
        opt_k = Option(k, inputs.maturity, inputs.option_type, inputs.option_class)
        tree_val, _ = tree_price(market, inputs.N, inputs.pruning, inputs.epsilon, inputs.pricing_date, opt_k)
        bs_val = compute_bs_price(inputs, k, T) if inputs.option_class.lower() == "european" else np.nan
        rows.append((k, tree_val, bs_val))

    return pd.DataFrame(rows, columns=["Strike", "Tree", "BlackScholes"]).set_index("Strike")

# Fonction pour calculer le prix BS
def compute_bs_price(inputs: AppInputs, strike: float, T: float) -> float:
    """Calcule le prix Black-Scholes pour un strike donne."""
    return bs_price_cached(inputs.S0, strike, T, inputs.r, inputs.sigma, inputs.option_type,
                           dividend=inputs.dividend, dividend_date=inputs.dividend_date)

# ------------------------------ UI ------------------------------

st.set_page_config(page_title="Trinomial Tree Pricer", layout="wide")
st.title("📈 Trinomial Tree & Black–Scholes Pricer")
st.caption("Paramétrez l’arbre, comparez à Black–Scholes, et explorez graphiques/convergences.")

with st.sidebar:
    st.header("Paramètres")

    today = dt.datetime.now().date()
    pricing_date = st.date_input("Pricing date", value=today)
    maturity = st.date_input("Maturity", value=today + dt.timedelta(days=180))

    S0 = st.number_input("Spot S0", min_value=0.0, value=100.0)
    K = st.number_input("Strike K", min_value=0.0, value=100.0)
    r = st.number_input("Rate r", value=0.02, step=0.002, format="%0.6f")
    sigma = st.number_input("Volatility σ", min_value=0.0001, value=0.20, step=0.01, format="%0.6f")

    option_type = st.selectbox("Option type", ["call", "put"], index=0)
    option_class = st.selectbox("Style", ["european", "american"], index=0)

    has_div = st.toggle("Dividende discret ?", value=False)
    dividend = 0.0
    dividend_date: Optional[dt.datetime] = None
    if has_div:
        dividend = st.number_input("Montant dividende", min_value=0.0, value=0.0)
        div_date = st.date_input("Date dividende", value=today + dt.timedelta(days=90))
        dividend_date = dt.datetime.combine(div_date, dt.time())

    st.subheader("Arbre trinomial")
    N = st.number_input("Nombre d’étapes N", min_value=1, max_value=2000, value=100, step=10)
    pruning = st.toggle("Pruning", value=True)
    epsilon = st.number_input("Epsilon (tolérance)", min_value=1e-12, value=1e-7, step=1e-7, format="%0.1e")

    st.subheader("Convergence & Courbe")
    conv_min = st.number_input("Convergence: N min", min_value=1, max_value=5000, value=50, step=5)
    conv_max = st.number_input("Convergence: N max", min_value=1, max_value=5000, value=400, step=10)
    conv_step = st.number_input("Convergence: pas", min_value=1, max_value=1000, value=25, step=1)

    curve_pts = st.slider("Points pour courbe Strike", min_value=5, max_value=101, value=41, step=2)
    curve_span_pct = st.slider("Écart autour de K (%)", min_value=5, max_value=100, value=30, step=5)

    st.subheader("Affichage de l’arbre")
    max_depth = st.number_input("Profondeur max (None=tout)", min_value=1, max_value=5000, value=min(50, N), step=1)
    proba_min = st.number_input("Proba min pour afficher un nœud", min_value=0.0, value=1e-6, format="%0.1e")
    percentile_clip = st.slider("Clip vertical (percentile)", min_value=0, max_value=20, value=0, step=1)
    edge_alpha = st.slider("Transparence des arêtes", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    linewidth = st.slider("Épaisseur des arêtes", min_value=0.1, max_value=2.0, value=0.4, step=0.1)

    run = st.button("▶️ Calculer / Mettre à jour", type="primary")
    plot_btn = st.button("🌳 Tracer l’arbre")

# Rassemblement des inputs
inputs = AppInputs(
    pricing_date=dt.datetime.combine(pricing_date, dt.time()),
    maturity=dt.datetime.combine(maturity, dt.time()),
    S0=float(S0), K=float(K), r=float(r), sigma=float(sigma),
    option_type=option_type, option_class=option_class,
    dividend=float(dividend), dividend_date=dividend_date,
    N=int(N), pruning=bool(pruning), epsilon=float(epsilon),
    conv_min=int(conv_min), conv_max=int(conv_max), conv_step=int(conv_step),
    curve_pts=int(curve_pts), curve_span_pct=float(curve_span_pct),
)

# -------- Prix courant & Grecs --------
colA, colB, colC = st.columns([1, 1, 1])

market, option, T = build_market_option(inputs)

with colA:
    st.subheader("Prix – Arbre trinomial")
    tree_price_val, t_sec = tree_price(market, inputs.N, inputs.pruning, inputs.epsilon, inputs.pricing_date, option)
    st.metric("Tree price", f"{tree_price_val:,.6f}", help=f"Temps de calcul: {t_sec:.3f} s, N={inputs.N}")

with colB:
    st.subheader("Prix – Black–Scholes (européen)")
    if inputs.option_class.lower() == "european":
        bs_val = bs_price_cached(inputs.S0, inputs.K, T, inputs.r, inputs.sigma, inputs.option_type,
                                 dividend=inputs.dividend, dividend_date=inputs.dividend_date)
        st.metric("BS price", f"{bs_val:,.6f}")
        st.caption("Comparaison valide pour options européennes.")
    else:
        st.caption("Black–Scholes n’est affiché que pour les options **européennes**.")
        bs_val = np.nan

with colC:
    st.subheader("Greeks – Arbre & BS")
    tree_greeks, g_tree = compute_tree_and_greeks(market, option, inputs)

    if inputs.option_class.lower() == "european":
        bs = BlackScholesPricer(inputs.S0, inputs.K, T, inputs.r, inputs.sigma, inputs.option_type,
                                dividend=inputs.dividend, dividend_date=inputs.dividend_date)
        g_bs = bs.greeks()
        bs_delta = g_bs.get("delta", np.nan)
        bs_gamma = g_bs.get("gamma", np.nan)
        bs_theta = g_bs.get("theta", np.nan)
        bs_vega = g_bs.get("vega", np.nan)
        bs_rho = g_bs.get("rho", np.nan)
    else:
        bs_delta = bs_gamma = bs_theta = bs_vega = bs_rho = np.nan

    greek_df = pd.DataFrame({
        "Tree": [g_tree["delta"], g_tree["gamma"], g_tree["theta_day"], g_tree["vega"], g_tree["rho"]],
        "Black–Scholes": [bs_delta, bs_gamma, bs_theta, bs_vega, bs_rho],
    }, index=["Delta", "Gamma", "Theta (par jour)", "Vega (/% vol)", "Rho (/% taux)"])
    st.dataframe(greek_df.style.format("{:.6f}"))

# -------- Visualisation de l’arbre --------
st.markdown("---")
st.subheader("Visualisation de l’arbre")

# on crée l'arbre
tree = TrinomialTree(market, N=inputs.N, pruning=inputs.pruning,
                          epsilon=inputs.epsilon, pricingDate=inputs.pricing_date)
_ = tree.price(option, build_tree=True)
tree.plot_tree(y_max=280, y_min=30)
fig = plt.gcf()
st.pyplot(fig, clear_figure=True)

# --- Convergence (prix vs. nombre d’étapes) ---
st.markdown("---")
st.subheader("Convergence (prix vs. nombre d’étapes)")

prev_ids = set(plt.get_fignums())
convergence.bs_convergence_by_step(
    market, option,
    max_n=int(inputs.conv_max), step=int(max(1, inputs.conv_step)),
    pruning=inputs.pruning, epsilon=inputs.epsilon,
)
new_ids = [fid for fid in plt.get_fignums() if fid not in prev_ids]
for fid in new_ids:
    st.pyplot(plt.figure(fid), clear_figure=False)

# --- Courbe prix vs Strike ---
st.markdown("---")
st.subheader("Courbe de prix en fonction du strike — BS vs Arbre")

span = inputs.curve_span_pct / 100.0
strikes = np.linspace(inputs.K * (1.0 - span), inputs.K * (1.0 + span), inputs.curve_pts)

prev_ids = set(plt.get_fignums())
convergence.bs_convergence_by_strike(
    market, option, strikes,
    n_steps=int(inputs.N), pruning=inputs.pruning, epsilon=inputs.epsilon,
)
new_ids = [fid for fid in plt.get_fignums() if fid not in prev_ids]
for fid in new_ids:
    st.pyplot(plt.figure(fid), clear_figure=False)

# -------- Notes --------
st.markdown(
    """
    **Notes**
    - La maturité est transformée en *T* (années) avec une base ACT/365.
    - Le pricer Black–Scholes est affiché uniquement pour les options **européennes**.
    - `pricing.*` est supposé installé et fonctionnel; pas de chemins spéciaux ni de try/except inutiles.
    - Le dividende discret (montant et date) est passé aux deux pricers s’il est renseigné.
    - Grecs BS analytiques; côté arbre : Δ/Γ via nœuds racine, vega/rho en FD si implémenté, θ approximé par FD d’un jour.
    """
)
