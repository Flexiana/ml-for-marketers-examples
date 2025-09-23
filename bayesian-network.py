#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic Bayesian Network (retail pricing & demand)
- Developer-specified DAG (no external libraries)
- Synthetic data generation
- CPD learning (MLE + Laplace smoothing)
- Exact inference by enumeration
- Prints CPDs and a few scenario posteriors

Tested with Python 3.10+.
"""

from __future__ import annotations
import math
import random
from itertools import product
from collections import defaultdict, Counter

random.seed(42)

# -----------------------------
# 1) Developer-defined structure
# -----------------------------
# Nodes:
# Season, CompetitorPrice, Price, Promo, Marketing,
# RelPrice (Price vs Competitor), Demand, Inventory, Stockout, Revenue

STRUCTURE = [
    ("CompetitorPrice", "RelPrice"),
    ("Price", "RelPrice"),
    ("Season", "Demand"),
    ("RelPrice", "Demand"),
    ("Promo", "Demand"),
    ("Marketing", "Demand"),
    ("Inventory", "Stockout"),
    ("Demand", "Stockout"),
    ("Price", "Revenue"),
    ("Demand", "Revenue"),
]

STATES = {
    "Season": ["Low", "High"],
    "CompetitorPrice": ["Low", "High"],
    "Price": ["Low", "High"],
    "Promo": ["No", "Yes"],
    "Marketing": ["Low", "High"],
    "RelPrice": ["Lower", "Higher"],  # our price relative to competitor
    "Demand": ["Low", "High"],
    "Inventory": ["Low", "High"],
    "Stockout": ["No", "Yes"],
    "Revenue": ["Low", "High"],
}

VARIABLES = list(STATES.keys())
PARENTS = {v: [u for (u, w) in STRUCTURE if w == v] for v in VARIABLES}


# ---------------------------------
# 2) Synthetic data generation
# ---------------------------------
def choice_with_probs(options, probs):
    r = random.random()
    s = 0.0
    for opt, p in zip(options, probs):
        s += p
        if r <= s:
            return opt
    return options[-1]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_row() -> dict:
    Season = choice_with_probs(STATES["Season"], [0.6, 0.4])
    CompetitorPrice = choice_with_probs(STATES["CompetitorPrice"], [0.5, 0.5])
    Price = choice_with_probs(STATES["Price"], [0.55, 0.45])
    Promo = choice_with_probs(STATES["Promo"], [0.7, 0.3])
    Marketing = choice_with_probs(STATES["Marketing"], [0.5, 0.5])
    Inventory = choice_with_probs(STATES["Inventory"], [0.3, 0.7])

    # RelPrice
    if Price == "Low" and CompetitorPrice == "High":
        RelPrice = "Lower"
    elif Price == "High" and CompetitorPrice == "Low":
        RelPrice = "Higher"
    else:
        # Tie-ish cases: small bias toward "Lower" if our Price is Low
        if Price == "Low":
            RelPrice = choice_with_probs(STATES["RelPrice"], [0.55, 0.45])
        else:
            RelPrice = choice_with_probs(STATES["RelPrice"], [0.45, 0.55])

    # Demand via logistic add-ups
    logit = -0.5
    logit += 0.8 if Season == "High" else -0.2
    logit += 0.8 if RelPrice == "Lower" else -0.6
    logit += 0.5 if Promo == "Yes" else 0.0
    logit += 0.4 if Marketing == "High" else 0.0
    p_high = sigmoid(logit)
    Demand = choice_with_probs(STATES["Demand"], [1 - p_high, p_high])

    # Stockout depends on Demand & Inventory
    if Inventory == "Low" and Demand == "High":
        p_stockout = 0.6
    elif Inventory == "Low" and Demand == "Low":
        p_stockout = 0.15
    elif Inventory == "High" and Demand == "High":
        p_stockout = 0.2
    else:
        p_stockout = 0.05
    Stockout = choice_with_probs(STATES["Stockout"], [1 - p_stockout, p_stockout])

    # Revenue depends on Price & Demand (stockout hurts in DGP; captured indirectly)
    if Demand == "High" and Stockout == "No":
        p_rev_high = 0.85 if Price == "High" else 0.75
    elif Demand == "High" and Stockout == "Yes":
        p_rev_high = 0.4 if Price == "High" else 0.3
    elif Demand == "Low" and Stockout == "No":
        p_rev_high = 0.35 if Price == "High" else 0.2
    else:
        p_rev_high = 0.1 if Price == "High" else 0.05
    Revenue = choice_with_probs(STATES["Revenue"], [1 - p_rev_high, p_rev_high])

    return {
        "Season": Season,
        "CompetitorPrice": CompetitorPrice,
        "Price": Price,
        "Promo": Promo,
        "Marketing": Marketing,
        "RelPrice": RelPrice,
        "Demand": Demand,
        "Inventory": Inventory,
        "Stockout": Stockout,
        "Revenue": Revenue,
    }


def generate_dataset(n: int) -> list[dict]:
    return [generate_row() for _ in range(n)]


# ---------------------------------
# 3) Fit CPDs (MLE + Laplace)
# ---------------------------------
def fit_cpds(data: list[dict], parents: dict, states: dict, alpha: float = 1.0):
    """
    Returns CPDs: { var: {"parents": [...], "table": { parent_tuple: [p(state1),...,p(stateK)] }}}
    parent_tuple is ordered to match 'parents' list.
    """
    cpds = {}
    for var, var_states in states.items():
        ps = parents[var]
        # All parent configurations
        combos = [()]
        if ps:
            parent_state_lists = [states[p] for p in ps]
            combos = list(product(*parent_state_lists))

        table = {}
        for combo in combos:
            # filter rows matching parent assignment
            rows = (r for r in data if all(r[p] == v for p, v in zip(ps, combo))) if ps else data
            counts = Counter(r[var] for r in rows)
            # Laplace smoothing
            smoothed = [counts.get(s, 0) + alpha for s in var_states]
            total = float(sum(smoothed))
            probs = [c / total for c in smoothed] if total > 0 else [1.0 / len(var_states)] * len(var_states)
            table[combo] = probs
        cpds[var] = {"parents": ps, "table": table}
    return cpds


# ---------------------------------
# 4) Exact inference by enumeration
# ---------------------------------
def topo_sort(structure: list[tuple[str, str]], variables: list[str]) -> list[str]:
    order, visited = [], set()

    def visit(v: str):
        if v in visited:
            return
        for u, w in structure:
            if w == v and u not in visited:
                visit(u)
        visited.add(v)
        order.append(v)

    for v in variables:
        visit(v)
    return order


TOPO = topo_sort(STRUCTURE, VARIABLES)


def local_prob(var: str, val: str, assignment: dict, cpds, states) -> float:
    ps = cpds[var]["parents"]
    combo = tuple(assignment[p] for p in ps) if ps else ()
    probs = cpds[var]["table"][combo]
    idx = states[var].index(val)
    return probs[idx]


def joint_prob(assignment: dict, cpds, states) -> float:
    p = 1.0
    for v in TOPO:
        p *= local_prob(v, assignment[v], assignment, cpds, states)
    return p


def enumerate_query(query_var: str, evidence: dict, cpds, states) -> dict[str, float]:
    var_states = states[query_var]
    hidden_vars = [v for v in states.keys() if v != query_var and v not in evidence]
    probs = []
    for val in var_states:
        total = 0.0
        for combo in product(*[states[v] for v in hidden_vars]):
            assign = dict(evidence)
            assign[query_var] = val
            assign.update({v: s for v, s in zip(hidden_vars, combo)})
            total += joint_prob(assign, cpds, states)
        probs.append(total)
    s = sum(probs)
    return {st: (p / s if s > 0 else 1.0 / len(var_states)) for st, p in zip(var_states, probs)}


# ---------------------------------
# 5) Pretty printers
# ---------------------------------
def fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


def print_cpd(var: str, cpds, states, max_cols: int = 6):
    ps = cpds[var]["parents"]
    var_states = states[var]
    combos = list(cpds[var]["table"].keys())
    header = f"CPD: {var} | " + (", ".join(ps) if ps else "∅")
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    # paginate columns if too many
    chunks = [combos[i : i + max_cols] for i in range(0, len(combos), max_cols)]
    for chunk in chunks:
        # header row for this stripe
        col_names = []
        for c in chunk:
            if c == ():
                col_names.append("P")
            else:
                col_names.append(", ".join(f"{p}={s}" for p, s in zip(ps, c)))
        col_width = max(12, max(len(cn) for cn in col_names))
        print(" " * 12 + " ".join(cn.ljust(col_width) for cn in col_names))
        # each state row
        for si, st in enumerate(var_states):
            row = []
            for c in chunk:
                row.append(fmt_pct(cpds[var]["table"][c][si]).rjust(col_width))
            print(f"{(var+'='+st).ljust(12)}" + " ".join(row))
        print("")


# ---------------------------------
# 6) Main: generate, train, infer
# ---------------------------------
def main():
    # Data
    n = 20_000
    data = generate_dataset(n)

    # Train CPDs
    cpds = fit_cpds(data, PARENTS, STATES, alpha=1.0)

    # Show a few informative CPDs
    for v in ["RelPrice", "Demand", "Stockout", "Revenue"]:
        print_cpd(v, cpds, STATES, max_cols=8)

    # Inference examples
    def show_dist(label, dist: dict[str, float]):
        items = ", ".join(f"{k}: {fmt_pct(v)}" for k, v in dist.items())
        print(f"{label}: {items}")

    baseline_demand = enumerate_query("Demand", {}, cpds, STATES)
    baseline_revenue = enumerate_query("Revenue", {}, cpds, STATES)
    show_dist("Baseline  P(Demand)", baseline_demand)
    show_dist("Baseline  P(Revenue)", baseline_revenue)
    print("")

    scenario_A = {"Season": "High", "CompetitorPrice": "High", "Price": "Low", "Promo": "Yes", "Marketing": "High"}
    show_dist("Scenario A  P(Demand | evidence)", enumerate_query("Demand", scenario_A, cpds, STATES))
    show_dist("Scenario A  P(Revenue | evidence)", enumerate_query("Revenue", scenario_A, cpds, STATES))
    print("")

    scenario_B = {"Season": "Low", "CompetitorPrice": "Low", "Price": "High", "Promo": "No", "Marketing": "Low"}
    show_dist("Scenario B  P(Demand | evidence)", enumerate_query("Demand", scenario_B, cpds, STATES))
    show_dist("Scenario B  P(Revenue | evidence)", enumerate_query("Revenue", scenario_B, cpds, STATES))
    print("")

    show_dist(
        "Stockout risk  P(Stockout | Demand=High, Inventory=Low)",
        enumerate_query("Stockout", {"Demand": "High", "Inventory": "Low"}, cpds, STATES),
    )

    # Quick sanity checks
    # (Compute a few frequency summaries without pandas)
    def count_by(field):
        c = Counter(r[field] for r in data)
        total = sum(c.values())
        return {k: f"{v} ({v/total:.1%})" for k, v in c.items()}

    print("\nSanity checks (counts):")
    print("Season   :", count_by("Season"))
    print("RelPrice :", count_by("RelPrice"))
    print("Demand   :", count_by("Demand"))


if __name__ == "__main__":
    main()