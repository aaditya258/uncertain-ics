#!/usr/bin/env python3
"""
reproduce.py -- regenerates every quantitative result, table value, and
data-driven figure of the manuscript "Uncertainty-Aware Cyber-Physical
Threat Prioritisation for Industrial Control Systems" (revised version,
panel elicitation).

Usage:
    python3 reproduce.py [--samples 100000] [--seed 42] [--p-mode 0.6|empirical]

Outputs:
    outputs/results.json   all statistics referenced in the text
    figures/*.pdf          all data-driven figures (incl. supplementary)

Dependencies: numpy, matplotlib.
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Declared study inputs
# --------------------------------------------------------------------------
# Ordinal parameters per scenario: CONSENSUS of the three-assessor panel
# (raw per-rater record below; elicited against the rubric, Table 4).
# Order: P_OPR, P_SAFETY, L, V, A, D, E.
# A is the ATTACKABILITY scale: 5 = little skill or resources required.
SCENARIOS = {
    "S1": dict(P_OPR=4, P_SAF=3, L=3, V=4, A=3, D=3, E=4),
    "S2": dict(P_OPR=4, P_SAF=4, L=3, V=4, A=3, D=4, E=5),
    "S3": dict(P_OPR=4, P_SAF=4, L=3, V=4, A=3, D=4, E=4),
    "S4": dict(P_OPR=3, P_SAF=4, L=3, V=4, A=3, D=5, E=3),
}
ORDER = ["S1", "S2", "S3", "S4"]
PARAM_KEYS = ["P_OPR", "P_SAF", "L", "V", "A", "D", "E"]
W_OPR, W_SAF = 0.4, 0.6

# Asset-level parameters (V, E) are drawn once per asset per Monte Carlo
# iteration and shared by every scenario targeting that asset (Alg. 2).
# S1 and S3 both target the PLC; their consensus V and E coincide.
SHARED_ASSET_GROUPS = [("S1", "S3")]  # (leader, follower*) share V, E draws

# Pre-consensus record: 3 raters x 4 scenarios x 7 parameters = 84
# judgements (rows: scenario x PARAM_KEYS order).
RATERS = {
    "A": {
        "S1": [4, 3, 3, 4, 3, 3, 4],
        "S2": [4, 4, 3, 4, 3, 4, 5],
        "S3": [4, 4, 3, 4, 3, 4, 4],
        "S4": [3, 4, 3, 4, 3, 5, 3],
    },
    "B": {
        "S1": [4, 3, 2, 4, 3, 3, 4],
        "S2": [4, 4, 3, 4, 3, 4, 5],
        "S3": [4, 4, 3, 4, 3, 4, 4],
        "S4": [3, 4, 3, 4, 3, 5, 3],
    },
    "C": {
        "S1": [4, 3, 3, 3, 3, 3, 4],
        "S2": [4, 4, 2, 4, 3, 4, 5],
        "S3": [4, 4, 3, 4, 3, 4, 4],
        "S4": [3, 4, 2, 4, 3, 5, 3],
    },
}

P_M_DEFAULT = 0.6  # conservative primary setting

CLASS_BINS = [  # (lower, upper, label); lower inclusive, upper exclusive
    (20.0, 25.01, "Very High"),
    (15.0, 20.0, "High"),
    (10.0, 15.0, "Moderate"),
    (5.0, 10.0, "Low"),
    (0.0, 5.0, "Very Low"),
]

# Baseline 1: IT-centric 5x5 risk matrix (STRIDE + likelihood x impact).
# Likelihood classes = consensus L column (3,3,3,3); impact scored on
# service/data disruption only (no HAZOP input); Sec. 5.1.
BASELINE_RM_IMPACT = {"S1": 4, "S2": 3, "S3": 4, "S4": 3}

# Baseline 1b: physics-aware likelihood x impact product (RM-CP), Sec. 5.1.
# Impact = weighted HAZOP impact P_sev of Eq. (1) (continuous, hence a
# product rather than a discrete 5x5 cell).
def rmcp_impact(sid):
    p = SCENARIOS[sid]
    return W_OPR * p["P_OPR"] + W_SAF * p["P_SAF"]

# Shrinkage priors for the agreement-derived retention: shrunk p_m =
# (k + a) / (12 + a + b) for Beta(a, b), k = judgements equal to consensus.
SHRINK_PRIORS = {"Beta(3,2)": (3, 2), "Beta(6,4)": (6, 4), "Beta(12,8)": (12, 8)}

# Baseline 2: CVSS v3.1 base vectors (Sec. 5.1).
CVSS_VECTORS = {
    "S1": "AV:A/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
    "S2": "AV:A/AC:L/PR:N/UI:N/S:U/C:N/I:H/A:N",
    "S3": "AV:A/AC:H/PR:L/UI:N/S:U/C:N/I:H/A:H",
    "S4": "AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H",
}

# Criteria-based comparison (Sec. 5.2), six attributes, L/M/H -> 1/2/3.
VALIDATION = {
    "ICS-STRATA": [3, 3, 3, 3, 1, 3],
    "SSPN-RA": [3, 3, 2, 1, 2, 2],
    "Hybrid automaton": [3, 2, 2, 1, 3, 1],
    "Formal analysis": [3, 1, 2, 1, 3, 2],
    "Design-centric": [3, 3, 2, 2, 2, 2],
    "Dynamic watermarking": [2, 1, 2, 2, 1, 1],
}
ATTRS = ["CPI", "RQP", "PT", "EoU", "FAD", "UR"]

C_SCEN = {"S1": "#0072B2", "S2": "#009E73", "S3": "#E69F00", "S4": "#D55E00"}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
    }
)


# --------------------------------------------------------------------------
# Panel agreement statistics
# --------------------------------------------------------------------------
def agreement_stats():
    items, cons = [], []
    for sid in ORDER:
        for k in range(7):
            items.append([RATERS[r][sid][k] for r in ("A", "B", "C")])
            cons.append(SCENARIOS[sid][PARAM_KEYS[k]])
    items = np.array(items)  # 28 x 3
    cons = np.array(cons)
    n_items, n_raters = items.shape
    match = int((items == cons[:, None]).sum())
    per_param = {}
    for k, key in enumerate(PARAM_KEYS):
        sel = np.arange(k, 28, 7)
        m = int((items[sel] == cons[sel, None]).sum())
        per_param[key] = dict(judgements=3 * len(sel), match=m,
                              p_m_hat=round(m / (3 * len(sel)), 3))
    pairs = {}
    for i, a in enumerate("ABC"):
        for j, b in enumerate("ABC"):
            if i < j:
                pairs[f"{a}-{b}"] = int((items[:, i] == items[:, j]).sum())
    unanimous = int((items.max(axis=1) == items.min(axis=1)).sum())
    divergent = n_items - unanimous
    max_spread = int((items.max(axis=1) - items.min(axis=1)).max())
    # Fleiss' kappa over 5 categories
    cats = np.arange(1, 6)
    nij = np.array([[np.sum(row == c) for c in cats] for row in items])
    P_i = ((nij ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_j = nij.sum(axis=0) / (n_items * n_raters)
    Pbar, Pe = P_i.mean(), float((p_j ** 2).sum())
    fleiss = (Pbar - Pe) / (1 - Pe)
    return dict(
        judgements=n_items * n_raters,
        match_consensus=match,
        pooled_p_m_hat=round(match / (n_items * n_raters), 3),
        per_parameter=per_param,
        pairwise_exact_agreement={k: f"{v}/28" for k, v in pairs.items()},
        unanimous_items=unanimous,
        divergent_items=divergent,
        max_spread=max_spread,
        fleiss_kappa=round(float(fleiss), 3),
        krippendorff_alpha_ordinal=round(float(krippendorff_alpha_ordinal(items)), 3),
        alpha_bootstrap_95=[round(x, 3) for x in alpha_bootstrap(items)],
        alpha_bootstrap_procedure="percentile; 10,000 resamples of the 28 cells "
                                  "(3 ratings of a cell kept together); seed 42",
    )


def krippendorff_alpha_ordinal(items):
    """Ordinal-metric Krippendorff's alpha; items = units x raters, no missing."""
    items = np.asarray(items)
    n_units, m = items.shape
    vals = np.arange(1, 6)
    n_c = np.array([(items == c).sum() for c in vals], dtype=float)
    n = n_c.sum()
    # ordinal distance: (sum_{g=c..k} n_g - (n_c + n_k)/2)^2
    delta = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            if i != j:
                lo, hi = min(i, j), max(i, j)
                delta[i, j] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    o = np.zeros((5, 5))
    for row in items:
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[row[a] - 1, row[b] - 1] += 1.0 / (m - 1)
    D_o = (o * delta).sum() / n
    D_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    return 1.0 - D_o / D_e if D_e > 0 else float("nan")


def alpha_bootstrap(items, reps=10_000, seed=42):
    """Percentile bootstrap over the 28 cells (all raters of a cell resampled
    together), reps repetitions, fixed seed."""
    rng = np.random.default_rng(seed)
    items = np.asarray(items)
    boot = np.array([krippendorff_alpha_ordinal(items[rng.integers(0, len(items), len(items))])
                     for _ in range(reps)])
    boot = boot[~np.isnan(boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def empirical_pm():
    ag = agreement_stats()["per_parameter"]
    return {k: ag[k]["match"] / ag[k]["judgements"] for k in PARAM_KEYS}


def shrunk_pm(a, b):
    ag = agreement_stats()["per_parameter"]
    return {k: (ag[k]["match"] + a) / (ag[k]["judgements"] + a + b) for k in PARAM_KEYS}


# --------------------------------------------------------------------------
# Scoring model
# --------------------------------------------------------------------------
def av(V, A, D, E):
    return (V * A * D * E) ** 0.25


def cpr(p):
    return (p["L"] / 5.0) * (W_OPR * p["P_OPR"] + W_SAF * p["P_SAF"]) * av(
        p["V"], p["A"], p["D"], p["E"]
    )


def classify(x):
    for lo, hi, lab in CLASS_BINS:
        if lo <= x < hi:
            return lab
    return "Very Low"


def det_ranks(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r


# --------------------------------------------------------------------------
# Monte Carlo layer
# --------------------------------------------------------------------------
BOUNDARY = "mode"  # Eq. (4): out-of-range mass returns to the mode; "renorm" = alternative


def perturb_one(rng, base, p_m, N):
    """One perturbed column. BOUNDARY == "mode": mass outside {1..5} returns
    to the mode (Eq. 4). BOUNDARY == "renorm": at a scale end the mass is
    renormalised over the mode and the one available neighbour."""
    if BOUNDARY == "renorm" and base in (1, 5):
        half = (1 - p_m) / 2
        p_stay = p_m / (p_m + half)
        step = 1 if base == 1 else -1
        delta = rng.choice([0, step], size=N, p=[p_stay, 1 - p_stay])
        return (base + delta).astype(float)
    delta = rng.choice([-1, 0, 1], size=N, p=[(1 - p_m) / 2, p_m, (1 - p_m) / 2])
    return np.clip(base + delta, 1, 5).astype(float)


def run_mc(pm_map, N, seed, shared=True):
    """pm_map: dict param -> retention. Asset-level params (V, E) are
    drawn once per asset and shared across scenarios in a shared group."""
    rng = np.random.default_rng(seed)
    S = len(ORDER)
    X = {sid: {} for sid in ORDER}
    for sid in ORDER:
        for k in PARAM_KEYS:
            X[sid][k] = perturb_one(rng, SCENARIOS[sid][k], pm_map[k], N)
    if shared:
        for leader, *rest in SHARED_ASSET_GROUPS:
            for f in rest:
                for k in ("V", "E"):
                    assert SCENARIOS[leader][k] == SCENARIOS[f][k]
                    X[f][k] = X[leader][k]
    draws = np.empty((N, S))
    for j, sid in enumerate(ORDER):
        p = X[sid]
        draws[:, j] = (p["L"] / 5.0) * (W_OPR * p["P_OPR"] + W_SAF * p["P_SAF"]) * (
            p["V"] * p["A"] * p["D"] * p["E"]
        ) ** 0.25
    order = np.argsort(-draws, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(N)[:, None]
    ranks[rows, order] = np.arange(1, S + 1)
    run_mc.last_X = X  # kept for the additive-aggregation check
    return draws, ranks


def additive_draws(X):
    """Monotone additive rule of Sec. 5.2: equal-weight mean of L, P_sev and
    the arithmetic mean of (V, A, D, E), each on the 1-5 scale."""
    S = len(ORDER)
    N = len(next(iter(X[ORDER[0]].values())))
    out = np.empty((N, S))
    for j, sid in enumerate(ORDER):
        p = X[sid]
        out[:, j] = (p["L"] + (W_OPR * p["P_OPR"] + W_SAF * p["P_SAF"])
                     + (p["V"] + p["A"] + p["D"] + p["E"]) / 4.0) / 3.0
    return out


def mc_statistics(draws, ranks, det_scores):
    S = len(ORDER)
    detr = det_ranks(det_scores)
    stats = {}
    R = np.zeros((S, S))
    for j, sid in enumerate(ORDER):
        d = draws[:, j]
        q05, q50, q95 = np.percentile(d, [5, 50, 95])
        cls = {lab: float(np.mean((d >= lo) & (d < hi))) for lo, hi, lab in CLASS_BINS}
        for r in range(1, S + 1):
            R[j, r - 1] = np.mean(ranks[:, j] == r)
        stats[sid] = dict(
            det=round(float(det_scores[j]), 2),
            det_rank=int(detr[j]),
            median=round(float(q50), 2),
            ci90=[round(float(q05), 2), round(float(q95), 2)],
            class_probs={k: round(v, 3) for k, v in cls.items() if v > 0.0005},
            p_det_rank=round(float(R[j, detr[j] - 1]), 3),
        )
    p_exact = float(np.mean((ranks == detr[None, :]).all(axis=1)))
    # --- tie rule of Alg. 2: fractional rank occupancy; a draw with a tie
    #     among the ranked scenarios does not count as a reproduction.
    greater = np.zeros_like(draws); ties = np.zeros_like(draws)
    for a in range(S):
        for b in range(S):
            if a != b:
                greater[:, a] += draws[:, b] > draws[:, a]
                ties[:, a] += draws[:, b] == draws[:, a]
    R_frac = np.zeros((S, S))
    for a in range(S):
        for r in range(S):
            R_frac[a, r] = np.mean(((r >= greater[:, a]) & (r <= greater[:, a] + ties[:, a]))
                                   / (ties[:, a] + 1))
    any_tie = ties.sum(axis=1) > 0
    ok = ~any_tie
    for a in range(S):
        ok &= greater[:, a] == (detr[a] - 1)
    tie_stats = dict(share_of_draws_with_tie=round(float(any_tie.mean()), 4),
                     p_exact_tie_excluded=round(float(ok.mean()), 3),
                     rank_probability_matrix_fractional={
                         ORDER[i]: [round(float(x), 3) for x in R_frac[i]] for i in range(S)})
    for j, sid in enumerate(ORDER):
        stats[sid]["p_det_rank_fractional"] = round(float(R_frac[j, detr[j] - 1]), 3)
    R = R_frac  # report the fractional-occupancy matrix (Alg. 2)
    # top-k set recovery
    det_top = np.argsort(detr)  # scenario indices by det rank
    topk = {}
    for k in (1, 2):
        sel = det_top[:k]
        topk[f"top{k}"] = round(float(np.mean((ranks[:, sel] <= k).all(axis=1))), 3)
    reversals = {}
    for i in range(S):
        for j in range(S):
            if detr[i] < detr[j]:
                reversals[f"{ORDER[j]}>{ORDER[i]}"] = round(
                    float(np.mean(draws[:, j] > draws[:, i])), 3
                )
    return stats, R, p_exact, topk, reversals, detr, tie_stats


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def cvss31_base(vector):
    w = {
        "AV": {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2},
        "AC": {"L": 0.77, "H": 0.44},
        "PR": {"N": 0.85, "L": 0.62, "H": 0.27},
        "UI": {"N": 0.85, "R": 0.62},
        "CIA": {"N": 0.0, "L": 0.22, "H": 0.56},
    }
    m = dict(item.split(":") for item in vector.split("/"))
    iss = 1 - (1 - w["CIA"][m["C"]]) * (1 - w["CIA"][m["I"]]) * (1 - w["CIA"][m["A"]])
    impact = 6.42 * iss
    expl = 8.22 * w["AV"][m["AV"]] * w["AC"][m["AC"]] * w["PR"][m["PR"]] * w["UI"][m["UI"]]
    if impact <= 0:
        return 0.0
    return math.ceil(min(impact + expl, 10.0) * 10) / 10


def competition_ranks(scores):
    s = np.asarray(scores, dtype=float)
    return [int(1 + np.sum(s > x)) for x in s]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_method_comparison(det_scores, stats, rm_scores, rmcp_scores, cvss_scores, path):
    methods = ["Proposed\n(CPPI)", "RM-IT", "RM-CP", "CVSS v3.1"]
    rank_sets = [
        det_ranks(det_scores).tolist(),
        competition_ranks(rm_scores),
        competition_ranks(rmcp_scores),
        competition_ranks(cvss_scores),
    ]
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    ypos = np.arange(len(ORDER))[::-1]
    for k, sid in enumerate(ORDER):
        st = stats[sid]
        ax1.errorbar(
            st["median"], ypos[k],
            xerr=[[st["median"] - st["ci90"][0]], [st["ci90"][1] - st["median"]]],
            fmt="o", color=C_SCEN[sid], capsize=3, markersize=5, lw=1.4,
        )
        ax1.plot(st["det"], ypos[k], marker="D", color="black", markersize=4, zorder=5)
    ax1.set_yticks(ypos, ORDER)
    ax1.set_xlabel("CPPI (dot = MC median, whiskers = 90% interval,\nblack diamond = deterministic)")
    ax1.set_xlim(0, 25)
    for lo, hi, lab in CLASS_BINS:
        ax1.axvspan(lo, min(hi, 25), alpha=0.05, color="grey")
        ax1.axvline(lo, color="grey", lw=0.4, alpha=0.5)
    ax1.set_title("(a) CPPI scores and uncertainty", fontsize=9)
    for k, sid in enumerate(ORDER):
        rr = [rank_sets[m][k] for m in range(4)]
        jitter = (k - 1.5) * 0.035
        ax2.plot([0, 1, 2, 3], [r + jitter for r in rr], marker="o",
                 color=C_SCEN[sid], lw=1.4, markersize=4, label=sid)
    ax2.set_xticks([0, 1, 2, 3], methods)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_ylim(4.5, 0.5)
    ax2.set_ylabel("rank (1 = highest risk)")
    ax2.set_title("(b) Rank agreement across methods", fontsize=9)
    ax2.legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.42), columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rankprob(R, detr, path):
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = R[i, j]
            ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "--", ha="center",
                    va="center", fontsize=8, color="white" if v > 0.55 else "black")
        ax.add_patch(plt.Rectangle((detr[i] - 1 - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="#D55E00", lw=1.8))
    ax.set_xticks(range(4), [f"rank {r}" for r in range(1, 5)])
    ax.set_yticks(range(4), ORDER)
    fig.colorbar(im, ax=ax, shrink=0.8, label="probability")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_sensitivity(top_sid, pm_grid, pm_exact, pm_top, path):
    base = dict(SCENARIOS[top_sid])
    c0 = cpr(base)
    deltas = []
    for k in PARAM_KEYS:
        lo, hi = dict(base), dict(base)
        lo[k] = max(1, base[k] - 1)
        hi[k] = min(5, base[k] + 1)
        deltas.append((k, cpr(lo) - c0, cpr(hi) - c0))
    deltas.sort(key=lambda t: -(abs(t[1]) + abs(t[2])))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))
    labels = {"P_OPR": r"$P_{OPR}$", "P_SAF": r"$P_{SAFETY}$", "L": r"$L$",
              "V": r"$V$", "A": r"$A$", "D": r"$D$", "E": r"$E$"}
    y = np.arange(len(deltas))[::-1]
    for (k, dlo, dhi), yy in zip(deltas, y):
        ax1.barh(yy, dhi, color="#0072B2", alpha=0.85, height=0.6)
        ax1.barh(yy, dlo, color="#D55E00", alpha=0.85, height=0.6)
    ax1.set_yticks(y, [labels[k] for k, _, _ in deltas])
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_xlabel(rf"$\Delta$CPPI({top_sid}) for a one-level ordinal change")
    ax1.set_title(f"(a) Tornado, scenario {top_sid}", fontsize=9)
    ax2.plot(pm_grid, pm_top, marker="o", ms=3, color="#0072B2",
             label=rf"$\Pr[${top_sid} at rank 1$]$")
    ax2.plot(pm_grid, pm_exact, marker="s", ms=3, color="#D55E00", label=r"$p_{exact}$")
    ax2.axvline(0.6, color="grey", lw=0.8, ls="--")
    ax2.text(0.61, 0.02, "default", fontsize=7, color="grey", va="bottom")
    ax2.set_xlabel(r"retention parameter $p_m$")
    ax2.set_ylabel("probability")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(b) Ranking robustness vs. $p_m$", fontsize=9)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_convergence(draws, ranks, detr, top_idx, top_sid, path):
    N = draws.shape[0]
    n = np.arange(1, N + 1)
    top = np.cumsum(ranks[:, top_idx] == 1) / n
    exact = np.cumsum((ranks == detr[None, :]).all(axis=1)) / n
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    sl = slice(99, N, 100)
    for series, lab, col in [
        (top, rf"$\Pr[${top_sid} at rank 1$]$", "#0072B2"),
        (exact, r"$p_{exact}$", "#D55E00"),
    ]:
        se = np.sqrt(series * (1 - series) / n)
        ax.plot(n[sl], series[sl], color=col, lw=1.2, label=lab)
        ax.fill_between(n[sl], (series - 2 * se)[sl], (series + 2 * se)[sl],
                        color=col, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo samples $N$")
    ax.set_ylabel("running estimate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_spider(path):
    K = len(ATTRS)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(4.6, 4.8), subplot_kw=dict(polar=True))
    fig.subplots_adjust(top=0.92, bottom=0.24, left=0.12, right=0.88)
    styles = {
        "ICS-STRATA": dict(color="#D55E00", lw=2.2, zorder=5),
        "SSPN-RA": dict(color="#0072B2", lw=1.1),
        "Hybrid automaton": dict(color="#009E73", lw=1.1),
        "Formal analysis": dict(color="#CC79A7", lw=1.1),
        "Design-centric": dict(color="#E69F00", lw=1.1),
        "Dynamic watermarking": dict(color="#56B4E9", lw=1.1),
    }
    for m, sc in VALIDATION.items():
        vals = [v / 3 * 100 for v in sc] + [sc[0] / 3 * 100]
        ax.plot(ang, vals, label=m, **styles[m])
    ax.set_xticks(ang[:-1], ATTRS)
    ax.tick_params(pad=8)
    ax.set_yticks([33.3, 66.7, 100], ["L", "M", "H"])
    ax.set_ylim(0, 100)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               handlelength=1.8, columnspacing=1.4, bbox_to_anchor=(0.5, 0.01))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_weights(path):
    ws = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for sid in ORDER:
        p = SCENARIOS[sid]
        vals = [(p["L"] / 5.0) * ((1 - w) * p["P_OPR"] + w * p["P_SAF"])
                * av(p["V"], p["A"], p["D"], p["E"]) for w in ws]
        ax.plot(ws, vals, color=C_SCEN[sid], lw=1.4, label=sid)
    ax.axvline(W_SAF, color="grey", lw=0.8, ls="--")
    ax.text(W_SAF + 0.01, 1.0, "default $w_{saf}=0.6$", fontsize=7, color="grey")
    ax.set_xlabel(r"safety weight $w_{saf}$  ($w_{opr}=1-w_{saf}$)")
    ax.set_ylabel("CPPI")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    ax.set_ylim(0, 15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-mode", type=str, default="0.6",
                    help="uniform retention (float) or 'empirical'")
    ap.add_argument("--boundary", choices=["mode", "renorm"], default="mode",
                    help="Eq. (4) boundary rule (mode) or renormalised alternative")
    ap.add_argument("--scale-benchmark", action="store_true",
                    help="time a synthetic 300-scenario register at N samples")
    args = ap.parse_args()
    global BOUNDARY
    BOUNDARY = args.boundary

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    agree = agreement_stats()
    pm_emp = empirical_pm()

    det_scores = [cpr(SCENARIOS[sid]) for sid in ORDER]
    detr = det_ranks(det_scores)
    top_sid = ORDER[int(np.argmin(detr))]
    top_idx = ORDER.index(top_sid)
    det_table = {
        sid: dict(
            **SCENARIOS[sid],
            AV=round(av(*[SCENARIOS[sid][k] for k in ["V", "A", "D", "E"]]), 2),
            CPPI=round(det_scores[j], 2),
            rank=int(detr[j]),
            priority_class=classify(det_scores[j]),
        )
        for j, sid in enumerate(ORDER)
    }

    # primary run: conservative uniform retention
    pm_primary = ({k: P_M_DEFAULT for k in PARAM_KEYS}
                  if args.p_mode != "empirical" else pm_emp)
    if args.p_mode not in ("0.6", "empirical"):
        pm_primary = {k: float(args.p_mode) for k in PARAM_KEYS}
    draws, ranks = run_mc(pm_primary, args.samples, args.seed, shared=True)
    stats, R, p_exact, topk, reversals, _, ties = mc_statistics(draws, ranks, det_scores)
    X_primary = run_mc.last_X

    # --- additive aggregation check (same draws), Sec. 5.2
    add_det = additive_draws({sid: {k: np.array([SCENARIOS[sid][k]], float)
                                    for k in PARAM_KEYS} for sid in ORDER})[0]
    add_draws = additive_draws(X_primary)
    add_order = np.argsort(-add_draws, axis=1, kind="stable")
    add_ranks = np.empty_like(add_order); add_ranks[np.arange(len(add_draws))[:, None], add_order] = np.arange(1, len(ORDER) + 1)
    add_stats, add_R, add_pex, _, add_rev, add_detr, add_ties = mc_statistics(add_draws, add_ranks, add_det.tolist())
    same_top = float(np.mean(add_draws.argmax(axis=1) == draws.argmax(axis=1)))
    same_order = float(np.mean((add_ranks == ranks).all(axis=1)))
    def spearman(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float); n = len(a)
        return 1 - 6 * ((a - b) ** 2).sum() / (n * (n * n - 1))
    def kendall(a, b):
        n = len(a); c = d = 0
        for i in range(n):
            for j in range(i + 1, n):
                sgn = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                c += sgn > 0; d += sgn < 0
        return (c - d) / (n * (n - 1) / 2)
    additive = dict(
        deterministic_scores={sid: round(float(add_det[j]), 3) for j, sid in enumerate(ORDER)},
        deterministic_ranks={sid: int(add_detr[j]) for j, sid in enumerate(ORDER)},
        spearman_rho_vs_multiplicative=round(float(spearman(add_detr, detr)), 3),
        kendall_tau_vs_multiplicative=round(float(kendall(add_detr, detr)), 3),
        p_rank1={sid: add_stats[sid]["p_det_rank_fractional"] if add_detr[j] == 1 else round(float(add_R[j, 0]), 3)
                 for j, sid in enumerate(ORDER)},
        p_exact=add_ties["p_exact_tie_excluded"],
        all_pairs_flagged_tau_0_1=bool(all(v > 0.1 for v in add_rev.values())),
        same_top_scenario_share=round(same_top, 3),
        same_full_order_share=round(same_order, 3))

    # --- compound conjunctive paths C1 = S2^S4, C2 = S3^S4 (Sec. 5.2)
    def path_index(init, l_rule):
        a = min(av(*[SCENARIOS[init][k] for k in ("V", "A", "D", "E")]),
                av(*[SCENARIOS["S4"][k] for k in ("V", "A", "D", "E")]))
        l_i, l_4 = SCENARIOS[init]["L"] / 5.0, SCENARIOS["S4"]["L"] / 5.0
        L = min(l_i, l_4) if l_rule == "common_campaign" else l_i * l_4
        return round(L * (W_OPR * 4 + W_SAF * 4) * a, 2)
    compound = {f"C{i}={init}^S4": {rule: path_index(init, rule)
                                    for rule in ("common_campaign", "independent_compromise")}
                for i, init in ((1, "S2"), (2, "S3"))}

    # --- threshold sweep for the order-undecided flag
    tau_sweep = {str(t): int(sum(v > t for v in reversals.values())) for t in (0.05, 0.10, 0.20)}

    # empirical (optimistic) bracket run
    draws_e, ranks_e = run_mc(pm_emp, args.samples, args.seed + 777, shared=True)
    stats_e, R_e, p_exact_e, topk_e, reversals_e, _, ties_e = mc_statistics(
        draws_e, ranks_e, det_scores)

    # shrunk agreement-derived retention: Beta(6,4) is the manuscript's
    # second setting; Beta(3,2) and Beta(12,8) are the prior sensitivity.
    shrunk = {}
    for lab, (a, b) in SHRINK_PRIORS.items():
        pm_s = shrunk_pm(a, b)
        d_s, r_s = run_mc(pm_s, args.samples, args.seed, shared=True)
        st_s, R_s, pe_s, tk_s, rv_s, _, ti_s = mc_statistics(d_s, r_s, det_scores)
        shrunk[lab] = dict(
            p_m={k: round(v, 3) for k, v in pm_s.items()},
            p_top_rank1=st_s[top_sid]["p_det_rank_fractional"],
            p_exact=ti_s["p_exact_tie_excluded"],
            min_reversal=round(min(rv_s.values()), 3),
            max_reversal=round(max(rv_s.values()), 3),
            all_pairs_flagged_tau_0_1=bool(all(v > 0.1 for v in rv_s.values())),
            pairwise_reversal_probabilities=rv_s)

    # independent-draw variant (robustness of the shared-draw specification)
    draws_i, ranks_i = run_mc(pm_primary, args.samples, args.seed, shared=False)
    _, _, p_exact_i, topk_i, reversals_i, _, _ = mc_statistics(draws_i, ranks_i, det_scores)

    # p_m sweep
    pm_grid = np.round(np.arange(0.3, 0.96, 0.05), 2)
    pm_exact, pm_top, pm_exact_argsort, pm_tieshare = [], [], [], []
    for i, pm in enumerate(pm_grid):
        d, r = run_mc({k: float(pm) for k in PARAM_KEYS},
                      args.samples, args.seed + 1 + i, shared=True)
        st_p, _, pe_p, _, _, _, ti_p = mc_statistics(d, r, det_scores)
        pm_exact.append(ti_p["p_exact_tie_excluded"])      # tie rule of Alg. 2
        pm_exact_argsort.append(float(pe_p))               # ties broken by index order
        pm_tieshare.append(ti_p["share_of_draws_with_tie"])
        pm_top.append(st_p[top_sid]["p_det_rank_fractional"])

    # convention sensitivity: conjoint impact readings for S4
    conv = {}
    for lab, (po, ps) in {"panel_split": (3, 4), "conjoint_elicited": (4, 4),
                          "conjoint_worst_case": (5, 5)}.items():
        p = dict(SCENARIOS["S4"], P_OPR=po, P_SAF=ps)
        sc = [cpr(SCENARIOS[s]) if s != "S4" else cpr(p) for s in ORDER]
        conv[lab] = dict(cpr_S4=round(cpr(p), 2),
                         rank_S4=int(det_ranks(sc)[ORDER.index("S4")]))

    # weight sweep crossing (S1 vs S4)
    a1, a4 = av(4, 3, 3, 4), av(4, 3, 5, 3)
    w_cross = round((4 * a1 - 3 * a4) / (a1 + a4), 2)

    rm_scores = [SCENARIOS[sid]["L"] * BASELINE_RM_IMPACT[sid] for sid in ORDER]
    rmcp_scores = [round(SCENARIOS[sid]["L"] * rmcp_impact(sid), 2) for sid in ORDER]
    cvss_scores = [cvss31_base(CVSS_VECTORS[sid]) for sid in ORDER]

    fig_method_comparison(det_scores, stats, rm_scores, rmcp_scores, cvss_scores,
                          "figures/fig_method_comparison.pdf")
    fig_rankprob(R, detr, "figures/fig_rankprob.pdf")
    fig_sensitivity(top_sid, pm_grid, pm_exact, pm_top, "figures/fig_sensitivity.pdf")
    fig_convergence(draws, ranks, detr, top_idx, top_sid, "figures/fig_convergence.pdf")
    # fig_spider("figures/fig_spider.pdf")  # criteria comparison removed from the paper
    fig_weights("figures/fig_weights.pdf")

    results = dict(
        settings=dict(N=args.samples, seed=args.seed, p_mode=args.p_mode,
                      w_opr=W_OPR, w_saf=W_SAF,
                      p_m_primary={k: round(v, 3) for k, v in pm_primary.items()},
                      p_m_empirical={k: round(v, 3) for k, v in pm_emp.items()}),
        panel_agreement=agree,
        deterministic=det_table,
        monte_carlo=stats,
        rank_probability_matrix={ORDER[i]: [round(float(x), 3) for x in R[i]]
                                 for i in range(len(ORDER))},
        p_exact=round(p_exact, 3),
        tie_statistics=ties,
        top_k_set_recovery=topk,
        pairwise_reversal_probabilities=reversals,
        empirical_bracket=dict(
            monte_carlo=stats_e, p_exact=round(p_exact_e, 3),
            tie_statistics=ties_e,
            top_k_set_recovery=topk_e,
            pairwise_reversal_probabilities=reversals_e),
        shrunk_retention_settings=shrunk,
        additive_aggregation_check=additive,
        compound_paths=compound,
        pairs_flagged_by_tau=tau_sweep,
        boundary_rule=BOUNDARY,
        independent_draw_check=dict(
            p_exact=round(p_exact_i, 3), top_k_set_recovery=topk_i,
            pairwise_reversal_probabilities=reversals_i),
        convention_sensitivity_S4=conv,
        weight_crossing_S1_S4=w_cross,
        pm_sweep=dict(p_m=[float(x) for x in pm_grid],
                      p_exact=[round(x, 3) for x in pm_exact],
                      p_exact_ties_broken_by_index=[round(x, 3) for x in pm_exact_argsort],
                      share_of_draws_with_tie=[round(x, 4) for x in pm_tieshare],
                      p_top_rank1=[round(x, 3) for x in pm_top]),
        baselines=dict(
            risk_matrix=dict(
                likelihood={sid: SCENARIOS[sid]["L"] for sid in ORDER},
                impact=BASELINE_RM_IMPACT,
                scores={sid: rm_scores[j] for j, sid in enumerate(ORDER)},
                ranks={sid: r for sid, r in zip(ORDER, competition_ranks(rm_scores))},
            ),
            rm_cp=dict(
                likelihood={sid: SCENARIOS[sid]["L"] for sid in ORDER},
                impact_P_sev={sid: round(rmcp_impact(sid), 2) for sid in ORDER},
                scores={sid: rmcp_scores[j] for j, sid in enumerate(ORDER)},
                ranks={sid: r for sid, r in zip(ORDER, competition_ranks(rmcp_scores))},
            ),
            cvss=dict(
                vectors=CVSS_VECTORS,
                scores={sid: cvss_scores[j] for j, sid in enumerate(ORDER)},
                ranks={sid: r for sid, r in zip(ORDER, competition_ranks(cvss_scores))},
            ),
        ),
        tornado_top={
            k: dict(
                minus=round(cpr({**SCENARIOS[top_sid], k: max(1, SCENARIOS[top_sid][k] - 1)})
                            - det_scores[top_idx], 2),
                plus=round(cpr({**SCENARIOS[top_sid], k: min(5, SCENARIOS[top_sid][k] + 1)})
                           - det_scores[top_idx], 2),
            )
            for k in PARAM_KEYS
        },
    )
    if args.scale_benchmark:
        import time
        rng = np.random.default_rng(args.seed)
        S_big = 300
        t0 = time.perf_counter()
        big = np.empty((args.samples, S_big))
        for j in range(S_big):
            cols = {k: perturb_one(rng, int(rng.integers(1, 6)), P_M_DEFAULT, args.samples)
                    for k in PARAM_KEYS}
            big[:, j] = (cols["L"] / 5.0) * (W_OPR * cols["P_OPR"] + W_SAF * cols["P_SAF"]) * (
                cols["V"] * cols["A"] * cols["D"] * cols["E"]) ** 0.25
        np.argsort(-big, axis=1, kind="stable")
        results["scale_benchmark"] = dict(scenarios=S_big, samples=args.samples,
                                          seconds=round(time.perf_counter() - t0, 1))
    with open("outputs/results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
