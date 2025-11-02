# TRINOMIAL-TREE-PRICING

Bibliothèque et application pour pricer des options via un arbre trinomial et comparer avec le modèle analytique de Black–Scholes. Une application Streamlit est fournie pour l’exploration interactive (prix, grecs, convergence, visualisations).

## ⚙️ Prérequis
- Python ≥ 3.12 (voir `pyproject.toml`)
- macOS, Linux ou Windows

Optionnel (installé automatiquement avec l’extra `[dev]`) :
- streamlit, numpy, pandas, matplotlib, scipy, pytest, black, mypy

## 📦 Installation

> Créez d’abord un environnement virtuel (ou bien installez le package sur le site-packages local).
> ```bash
> python -m venv my_venv
Activez le :
> pour windows : source my_venv/Scripts/activate
> pour mac : source my_venv/bin/activate 
> ```

Cloner le dépôt puis installer le package localement. La découverte des packages se fait depuis `src/` (configurée dans `pyproject.toml`).

Installation utilisateur (bibliothèque uniquement) :
```bash
pip install .
```

Installation de développement (bibliothèque + dépendances utiles à l’app/aux tests) :
```bash
pip install -e .[dev]
```

Vérification rapide des tests (optionnel) :
```bash
pytest -q
```

## 🧩 Ce que contient `pyproject.toml`
Points clés de la configuration:
- `[project]` name=`"pricing"`, version, `requires-python=">=3.12"`, `readme="README.md"`.
- Dépendances principales: vides (la lib cœur n’impose rien). Les dépendances utiles à l’app/aux outils sont dans `[project.optional-dependencies].dev` (streamlit, pandas, matplotlib, scipy, pytest, etc.).
- Découverte des paquets: `[tool.setuptools.packages.find]` avec `where=["src"]`, `include=["pricing*"]`.
- Outils: configuration `black` et `mypy` incluse.

Conséquence pratique: `pip install -e .[dev]` installe la lib en editable et ajoute les outils/app nécessaires (Streamlit inclus).

## 🚀 Démarrer l’application Streamlit

Après installation (idéalement `[dev]`), lancez l’UI:

Exécutez :
```bash
streamlit run streamlit_app.py
```

L’interface permet de:
- Choisir les paramètres de marché et d’option (S0, K, r, σ, maturité, type call/put, style européen/américain)
- Activer un dividende discret (montant + date)
- Régler l’arbre (N, pruning, epsilon)
- Visualiser l’arbre et tracer les courbes de convergence et prix vs. strike

Notes UI:
- Le prix Black–Scholes n’est affiché que pour les options européennes.
- Base de temps utilisée ACT/365F.

## 🧠 API principale (package `pricing`)

Modules importants situés dans `src/pricing/`:

- `Market(S0, r, sigma, dividend=0.0, dividend_date=None)`
   Représente les conditions de marché.

- `Option(K, maturity: datetime, option_type: 'call'|'put', option_class: 'european'|'american')`
   Porte l’instrument. Méthode: `payoff(S)`.

- `TrinomialTree(market, N, pruning=False, epsilon=1e-7, pricing_date=None)`
   Modèle d’arbre trinomial.
   - `price(option, method='backward'|'recursive', build_tree=False, compute_greeks=False) -> float`
   - `delta()`, `gamma()`, `vega(option)`, `vanna(option)`, `rho(option)`
   - `plot_tree(...)` pour visualiser la structure

- `BlackScholesPricer(S, K, T, r, sigma, option_type, dividend=0.0, dividend_date=None)`
   Pricer analytique (européen) avec dividende discret possible (unique).
   - `price() -> float`
   - `greeks() -> dict[str, float]` renvoie delta, gamma, theta (par jour), vega/rho/vanna par 1% de variation
   - `update(**kwargs)` pour modifier des paramètres (ex: `update(K=105, sigma=0.25)`)

- `convergence`
   - `bs_convergence_by_step(market, option, max_n=400, step=25, ...)`
   - `bs_convergence_by_strike(market, option, strikes, n_steps=200, ...)`
   - `plot_runtime_vs_steps(market, option, N_values)`

- `funcs`
   - `compute_forward`, `compute_variance`, `compute_probabilities`, `iter_column`, `probas_valid`

## 🛠️ Exemples rapides (Python)

Black–Scholes (européen):
```python
import datetime as dt
from pricing.blackscholes import BlackScholesPricer

T = 0.5  # 6 mois
bs = BlackScholesPricer(S=100, K=100, T=T, r=0.02, sigma=0.20, option_type="call")
print("BS price:", bs.price())
print("BS greeks:", bs.greeks())
```

Arbre trinomial (européen ou américain):
```python
import datetime as dt
from pricing.market import Market
from pricing.option import Option
from pricing.trinomial_tree import TrinomialTree

market = Market(S0=100, r=0.05, sigma=0.30, dividend=3, dividend_date=dt.date(2026, 4, 21))
option = Option(K=102, option_type="call", maturity=dt.date(2026, 9, 1), option_class="american")
tree = TrinomialTree(market, N=700, pruning=True, epsilon=1e-10, pricing_date = dt.date(2025, 9, 1))

# prix de l'arbre
price = tree.price(option, compute_greeks=True, activate_timer=True)
print("Prix de l’option:", price)

# Grecs côté arbre
_ = tree.price(option, compute_greeks=True)
print("Delta, Gamma:", tree.delta(), tree.gamma())
print("Vega, Vanna, Rho:", tree.vega(option), tree.vanna(option), tree.rho(option))
```

Courbes de convergence (matplotlib):
```python
import numpy as np
from pricing import convergence

# by step
convergence.bs_convergence_by_step(market, option, max_n=400, step=25, pruning=True, epsilon=1e-7)

# by strike
strikes = np.linspace(80, 120, 41)
convergence.bs_convergence_by_strike(market, option, strikes, n_steps=200, pruning=True, epsilon=1e-7)
```

## ✅ Tests
Les tests unitaires (pytest) sont dans `tests/`.
```bash
pytest
```

## 🗂️ Structure
```
PRICER V1/
├── src/pricing/              # Package Python (lib principale)
├── tests/                    # Tests unitaires (pytest)
├── streamlit_app.py          # Application Streamlit
├── pyproject.toml            # Configuration build/outilage
├── pytest.ini                # Config pytest
└── README.md                 # Ce fichier
```

## 📝 Notes et limites
- Black–Scholes: uniquement pour options européennes.
- Theta (BS) est reporté par jour; côté arbre, Theta est approché par différence finie (+1 jour).
- Le dividende discret (montant/date) est pris en compte dans les deux pricers.

## 👥 Auteurs
Yves‑Marie Saliou — Yessine Mannai

Projet réalisé dans le cadre du cours INFO QUANT (M2 IEF 272).