"""Figures for the AES Fall 2026 poster (48 x 36 in).

Every number is read from shipped files under ../cca_synth/runs/: the two
manipulation reports, the check records, single_vs_council.json, and
cross_role.json / cross_role_grok.json (written by analysis_cross_role.py).
Nothing is typed in by hand.

    python3 poster/make_poster_figures.py        # from the bundle root
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "..", "cca_synth")
RUNS = os.path.join(BUNDLE, "runs")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, BUNDLE)
from prompts import ROLE_PROMPTS  # noqa: E402

for f in os.listdir(os.path.join(HERE, "fonts")):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))

INK, SECONDARY, MUTED, GRID = "#090C27", "#454754", "#8A8C99", "#DADBE3"
TERRA, GROK = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the validated default palette
plt.rcParams.update({
    "font.family": "Open Sans", "font.size": 26, "text.color": INK,
    "axes.labelcolor": SECONDARY, "axes.edgecolor": MUTED, "axes.linewidth": 1.2,
    "xtick.color": SECONDARY, "ytick.color": SECONDARY, "xtick.major.size": 0,
    "ytick.major.size": 0, "pdf.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False,
})

ROLES = list(ROLE_PROMPTS)
LABEL = {"counter_evidence_checker": "Counter-evidence", "consistency_checker": "Consistency",
         "local_claim_checker": "Local-claim", "explore_then_prune": "Explore-then-prune",
         "rare_hypothesis_checker": "Rare-hypothesis", "confidence_calibrator": "Confidence calib."}
SHORT = {"counter_evidence_checker": "Counter-ev.", "consistency_checker": "Consist.",
         "local_claim_checker": "Local-claim", "explore_then_prune": "Explore",
         "rare_hypothesis_checker": "Rare-hyp.", "confidence_calibrator": "Calib."}


def report(fam):
    r = json.load(open(os.path.join(RUNS, fam, "manipulation_report.json")))
    return {x["role"]: x for x in r["results"]}


def fig_verdicts():
    reps = {"gpt-5.6-terra": (report("checks"), TERRA, -0.17), "grok-4.3": (report("checks-grok"), GROK, 0.17)}
    fig, ax = plt.subplots(figsize=(11.4, 7.5))
    for row in range(len(ROLES)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="#F4F3FA", zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=1.5, zorder=1)
    ax.axvline(0.40, color=SECONDARY, lw=1.8, ls=(0, (5, 4)), zorder=1)
    ax.text(0.36, -0.78, "pass threshold $d_z$ = 0.40", color=SECONDARY, fontsize=21, va="bottom", ha="right")
    for name, (rep, colour, off) in reps.items():
        for row, role in enumerate(ROLES):
            x = rep[role]
            d = x["vs_neutral"]["standardized_effect"]
            y = row + off
            # Interval: the bootstrap CI on the raw difference, rescaled to d units.
            diff = x["vs_neutral"]["diff"]
            lo, hi = (x["vs_neutral"]["lo"] / diff * d, x["vs_neutral"]["hi"] / diff * d) if abs(diff) > 1e-12 else (0, 0)
            ax.plot([min(lo, hi), max(lo, hi)], [y, y], color=colour, lw=3, solid_capstyle="round", zorder=2)
            ax.scatter([d], [y], s=330, zorder=3, linewidths=3.2, edgecolors=colour,
                       facecolors=colour if x["passed"] else "white")
    ax.set_yticks(range(len(ROLES)), [LABEL[r] for r in ROLES])
    ax.set_ylim(len(ROLES) - 0.5, -0.9)
    ax.set_xlim(-1.95, 1.3)
    ax.set_xlabel("Effect on own check vs. neutral (paired $d_z$, 95% CI)", fontsize=22)
    ax.tick_params(axis="y", labelsize=25, colors=INK)
    ax.spines["left"].set_visible(False)
    h = [plt.Line2D([], [], marker="o", ls="", ms=17, mew=3, mec=c, mfc=c) for c in (TERRA, GROK)]
    h += [plt.Line2D([], [], marker="o", ls="", ms=17, mew=3, mec=SECONDARY, mfc="white")]
    ax.legend(h, ["gpt-5.6-terra", "grok-4.3", "hollow = fails the rule"], ncol=3,
              loc="upper center", bbox_to_anchor=(0.40, -0.17), frameon=False, fontsize=22,
              handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_verdicts.pdf"))


def fig_matrix():
    M = {fam: json.load(open(os.path.join(RUNS, name)))
         for fam, name in (("checks", "cross_role.json"), ("checks-grok", "cross_role_grok.json"))}
    cmap = LinearSegmentedColormap.from_list("div", ["#c43a3a", "#e98a87", "#f0efec", "#86b6ef", "#1c5cab"])
    norm = TwoSlopeNorm(vmin=-1.8, vcenter=0, vmax=1.8)
    fig, axes = plt.subplots(1, 2, figsize=(18.2, 8.1))
    for ax, (fam, title) in zip(axes, [("checks", "gpt-5.6-terra"), ("checks-grok", "grok-4.3")]):
        c = M[fam]["matrix"]
        A = np.array([[c[f"{i}|{j}"]["d"] for j in ROLES] for i in ROLES])
        ax.imshow(A, cmap=cmap, norm=norm, aspect="equal")
        for i in range(6):
            for j in range(6):
                v = A[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=20,
                        color="white" if abs(v) > 1.05 else INK, fontweight="bold" if i == j else "normal")
        for k in np.arange(-0.5, 6, 1):
            ax.axhline(k, color="white", lw=3)
            ax.axvline(k, color="white", lw=3)
        for i in range(6):
            ax.add_patch(plt.Rectangle((i - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=INK, lw=4, zorder=5))
        ax.set_xticks(range(6), [SHORT[r] for r in ROLES], rotation=35, ha="right", fontsize=20)
        ax.set_yticks(range(6), [SHORT[r] for r in ROLES], fontsize=20)
        ax.set_title(title, fontsize=26, color=INK, pad=14, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Prompt given", fontsize=23)
    for ax in axes:
        ax.set_xlabel("Check measured", fontsize=23)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(OUT, "fig_matrix.pdf"))


def fig_claimcount():
    rows = [json.loads(l) for l in open(os.path.join(RUNS, "checks", "check_records.jsonl")) if l.strip()]
    recs = [r[i][v] for r in rows for i in ROLES for v in ("operation", "neutral", "persona")]
    n = np.array([len(x.get("atomic_claims") or []) for x in recs], float)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 6.0), sharex=True)
    for ax, role in zip(axes, ["local_claim_checker", "explore_then_prune", "rare_hypothesis_checker"]):
        y = np.array([ROLE_PROMPTS[role].check([x]) for x in recs], float)
        r = np.corrcoef(n, y)[0, 1]
        ax.scatter(n + rng.uniform(-0.25, 0.25, n.size), y + rng.uniform(-0.12, 0.12, n.size),
                   s=22, color=TERRA, alpha=0.18, linewidths=0)
        ax.set_title(f"{LABEL[role]}   r = {r:.2f}", fontsize=24, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=20)
    xs = np.array([0, n.max()])
    axes[0].plot(xs, 2 * xs, color=INK, lw=2, ls=(0, (5, 4)))
    axes[0].text(n.max() * 0.42, 2 * n.max() * 0.73, "score = 2 × claims", fontsize=20, color=INK)
    axes[0].set_ylabel("Check score", fontsize=22)
    axes[1].set_xlabel("Number of atomic claims in the output (gpt-5.6-terra, 3,600 outputs)", fontsize=22)
    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(OUT, "fig_claimcount.pdf"))


def fig_council():
    d = json.load(open(os.path.join(RUNS, "single_vs_council.json")))
    names = {"cca_council": "Operation council", "persona_only_council": "Persona council",
             "neutral_council": "Neutral council"}
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.axvline(0, color=MUTED, lw=1.5)
    for key, colour, off, lab in (("primary_ties_count_as_failure", INK, -0.15, "ties count as failure (registered)"),
                                  ("robustness_ties_excluded", MUTED, 0.15, "ties excluded (robustness)")):
        for row, c in enumerate(d[key]):
            y = row + off
            ax.plot([100 * c["ci_lo"], 100 * c["ci_hi"]], [y, y], color=colour, lw=3.5, solid_capstyle="round")
            ax.scatter([100 * c["delta"]], [y], s=260, color=colour, zorder=3, label=lab if row == 0 else None)
    ax.set_yticks(range(3), [names[c["contrast"].split(" - ")[0]] for c in d["primary_ties_count_as_failure"]])
    ax.set_ylim(2.6, -0.6)
    ax.set_xlabel("Accuracy minus a single call (percentage points, 95% CI)", fontsize=22)
    ax.tick_params(axis="y", labelsize=23, colors=INK)
    ax.tick_params(axis="x", labelsize=20)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=20,
              handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_council.pdf"))


if __name__ == "__main__":
    fig_verdicts()
    fig_matrix()
    fig_claimcount()
    fig_council()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.endswith(".pdf")))
