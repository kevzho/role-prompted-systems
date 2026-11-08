"""Render the paper's figures from run artifacts.

Figures are generated, never drawn by hand, for the same reason the results
tables are filled from artifacts: a figure is a claim, and a claim that cannot
be regenerated from the record is not checkable.

The checks panel replaces two tables -- the per-role checks and the
cross-family contrast -- with one forest plot, because they are the same six
roles measured twice and the comparison is the point.

Only the checks panel is included by the manuscript. The method schematic and
the classification grid are drawn in TikZ inside the paper so their type is the
document's own; the functions below are kept for reference and are not written
by default.

Confidence intervals are bootstrapped on each check's raw difference, whose
units differ per role (a rate, a claim count, a diversity product). Cohen's *d*
is that difference divided by a per-role pooled standard deviation, so the
interval is carried into *d* units by the same constant -- a linear rescaling,
not a re-estimation.

Usage::

    python3 make_paper_figures.py --report runs/cross_family.json \
        --out-dir ../paper/figures
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Wong (2011) Nature Methods palette -- the accessible-figure standard.
#: Validated on the six checks: CVD separation dE 21.9 protan, 30.9 tritan,
#: contrast >= 3:1 on both. Marker shape carries the same distinction so the
#: figure survives greyscale printing as well as CVD -- identity is never
#: colour-alone.
PRIMARY_COLOUR = "#0072B2"
REPLICATION_COLOUR = "#D55E00"

#: One display name per role, used by both panels so a role is spelled the same
#: way everywhere it appears, and spelled as the body text spells it.
DISPLAY_NAME = {
    "counter_evidence_checker": "counter-evidence",
    "consistency_checker": "consistency",
    "rare_hypothesis_checker": "rare-hypothesis",
    "confidence_calibrator": "confidence calibrator",
    "local_claim_checker": "local-claim",
    "explore_then_prune": "explore-then-prune",
}

#: NeurIPS text block is 5.5in wide. Figures are generated at the width they
#: are printed at, never scaled by \includegraphics: a figure drawn at 5.1in
#: and set at 0.78\linewidth renders its 8pt labels at 6.2pt, below the 7pt
#: floor these journals set, and inconsistently with any figure scaled
#: differently.
TEXT_WIDTH_IN: float = 5.5

#: The pre-registered pass threshold. The rule is one-sided (d >= 0.40); the
#: line is drawn on both sides because two roles move the wrong way and the
#: reader needs the same yardstick there.
D_THRESHOLD = 0.40


def _rescale(d: float, diff: float, lo: float, hi: float) -> tuple:
    """Carry a difference-scale interval into *d* units."""
    if diff == 0:
        return d, d
    scale = d / diff
    bounds = sorted((lo * scale, hi * scale))
    return bounds[0], bounds[1]


def _forest_axes(ax, report: Dict[str, Any], labels: List[str]) -> List[str]:
    """Draw the two-family effect-size forest plot.

    Returns the role order it drew, so the factor panel beneath can follow it
    rather than inventing a second ordering for the same roles.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,   # embedded TrueType; Type 3 is rejected by NeurIPS
        "ps.fonttype": 42,
    })

    roles = sorted(report["roles"], key=lambda r: r["d_primary"], reverse=True)

    # Alternate barely visible row bands so each role remains a readable
    # category when the two family estimates are vertically separated.
    for row in range(len(roles)):
        y = len(roles) - row
        if row % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#F7F7F7", zorder=0)
    for boundary in range(1, len(roles)):
        ax.axhline(boundary + 0.5, color="#DDDDDD", linewidth=0.65, zorder=0)

    # Recessive reference lines: zero solid, pass threshold dashed.
    ax.axvline(0.0, color="#666666", linewidth=0.7, zorder=1)
    for x in (-D_THRESHOLD, D_THRESHOLD):
        ax.axvline(x, color="#BBBBBB", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)

    # Separate the two families enough that their intervals and caps remain
    # legible when the panel is printed at workshop-column size.
    offset = 0.22
    for row, entry in enumerate(roles):
        y = len(roles) - row
        for suffix, colour, marker, sign in (
            ("primary", PRIMARY_COLOUR, "o", +1),
            ("replication", REPLICATION_COLOUR, "s", -1),
        ):
            d = entry[f"d_{suffix}"]
            lo, hi = _rescale(d, entry[f"diff_{suffix}"],
                              entry[f"lo_{suffix}"], entry[f"hi_{suffix}"])
            yy = y + sign * offset
            ax.errorbar(
                d, yy, xerr=[[d - lo], [hi - d]], fmt=marker,
                color=colour, markersize=5.0, markeredgecolor="white",
                markeredgewidth=0.7, elinewidth=1.8, capsize=2.8,
                capthick=1.2, solid_capstyle="round", zorder=3,
            )

    ax.set_yticks(range(1, len(roles) + 1))
    # "checker" appears on four of six rows and distinguishes nothing.
    ax.set_yticklabels([DISPLAY_NAME[r["role"]] for r in reversed(roles)],
                       fontsize=9.2)
    ax.set_xlabel("effect vs. neutral control (Cohen's $d$)")
    ax.set_ylim(0.4, len(roles) + 0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)

    handles = [
        plt.Line2D([], [], color=PRIMARY_COLOUR, marker="o", markersize=4.2,
                   linewidth=1.8, markeredgecolor="white", label=labels[0]),
        plt.Line2D([], [], color=REPLICATION_COLOUR, marker="s", markersize=4.2,
                   linewidth=1.8, markeredgecolor="white", label=labels[1]),
    ]
    # Above the axes, not inside them: at "upper left" the key sat on the
    # counter-evidence and consistency rows in the same glyphs as the data.
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=2, frameon=False, fontsize=8.5, handlelength=1.6,
              borderaxespad=0.0, columnspacing=1.6)
    return [r["role"] for r in roles]



def classification(check_report: Dict[str, Any], ablation: Dict[str, Any],
                   out_path: str) -> None:
    """Draw the registered two-axis classification of the six roles.

    The point of the figure is the empty column. Prose can state that no role
    moved the benchmark; a grid shows that the cells which would pair a
    benchmark movement with a passing or failing check are unoccupied, which is
    what the instrument exists to detect.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    passed = {r["role"]: bool(r["passed"]) for r in check_report["results"]}
    moves = {r["role"]: bool(r["moves_benchmark"])
             for r in ablation["leave_one_out"]["roles"]}

    cells: Dict[tuple, List[str]] = {(a, b): [] for a in (False, True)
                                     for b in (False, True)}
    for role in passed:
        cells[(passed[role], moves.get(role, False))].append(
            role.replace("_checker", "").replace("_", " "))

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN * 0.52, 1.65))
    for (does_pass, does_move), members in cells.items():
        x, y = (1 if does_move else 0), (1 if does_pass else 0)
        occupied = bool(members)
        ax.add_patch(plt.Rectangle(
            (x, y), 1, 1,
            facecolor="#EAF3F8" if occupied else "#FFFFFF",
            edgecolor="#CCCCCC", linewidth=0.7, zorder=1))
        if occupied:
            ax.text(x + 0.5, y + 0.5, "\n".join(members), ha="center",
                    va="center", fontsize=6.4, color="#1a1a1a", zorder=2)
        else:
            ax.text(x + 0.5, y + 0.5, "empty", ha="center", va="center",
                    fontsize=6.8, color="#AAAAAA", style="italic", zorder=2)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["no", "yes"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["no", "yes"])
    ax.set_xlabel("moves the benchmark"); ax.set_ylabel("passes its check")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)



def method(out_path: str) -> None:
    """Draw the check pipeline: three variants in, one gated verdict out.

    This is a schematic, not data, so colour does almost no work: one accent
    marks the gate, everything else is neutral. The point it carries is the
    thing prose needs a paragraph for --- that all three variants traverse an
    identical path and differ only at the prompt, so the contrast at the end
    isolates the prompt.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 1.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#999999", fs=7.0, weight=None):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=0.7, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.35, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#777777",
                                    linewidth=0.7, shrinkA=0, shrinkB=0), zorder=1)

    # Three prompt variants -- the only thing that differs.
    for i, (label, fill) in enumerate((
            ("operation", "#EAF3F8"), ("persona", "#FFFFFF"), ("neutral", "#FFFFFF"))):
        box(1, 24 - i * 9, 19, 7, label, fc=fill)
    ax.text(10.5, 2.5, "length-matched;\nidentical output contract",
            ha="center", va="center", fontsize=6.2, color="#666666", style="italic")

    for i in range(3):
        arrow(20, 27.5 - i * 9, 27, 17.5)

    box(27, 14, 17, 7, "same model,\nfixed decoding")
    arrow(44, 17.5, 51, 17.5)
    box(51, 14, 17, 7, "structured\nrecords")
    arrow(68, 17.5, 75, 17.5)
    box(75, 14, 24, 7, "check $m_i$ vs.\nneutral: $d$, CI")

    # The gate is the only accented element: it is where interpretation stops.
    arrow(87, 14, 87, 9)
    box(63, 2, 36, 7, "gate: $q \\leq 0.05$ and $d \\geq 0.40$",
        fc="#EAF3F8", ec=PRIMARY_COLOUR, fs=7.0)
    ax.text(61, 5.5, "attribution licensed only if", ha="right", va="center",
            fontsize=6.2, color="#666666", style="italic")

    fig.tight_layout(pad=0.2)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)



def _factor_axes(ax, decomposition: Dict[str, Any],
                 role_order: List[str] = None) -> None:
    """Show that each fallen check statistic fell through one factor only.

    Two of the six checks are products. A product can fall while one factor
    rises, so a negative contrast does not by itself show the operation ran
    backwards --- it could be a trade, which for explore-then-prune is arguably
    what "prune" means. This figure answers that objection with the data: the
    second factor is saturated in both arms, so it cannot be trading against
    anything, and the entire movement is carried by the first.

    Bars are change relative to the neutral control, which puts factors of
    different units on one axis honestly.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    pretty = {
        "diversity": "hypothesis diversity",
        "concentration": "concentration",
        "claim_count": "atomic claim count",
        "sourced_weight": "sourced-claim weight",
    }
    decomposed = ("explore_then_prune", "local_claim_checker")
    # Follow the forest plot's ordering. Two stacked panels that disagree about
    # where the same role sits are read as two different roles.
    order = [r for r in (role_order or decomposed) if r in decomposed]
    rows: List[tuple] = []
    for role in order:
        entry = decomposition["roles"][role]
        for name, stats in entry["factors"].items():
            neutral = stats["mean_neutral"]
            change = 0.0 if neutral == 0 else 100.0 * stats["diff"] / neutral
            rows.append((role, pretty.get(name, name), change,
                         bool(stats["interval_excludes_zero"])))

    ax.axvline(0.0, color="#666666", linewidth=0.6, zorder=3)
    for i, (_, factor, change, moved) in enumerate(rows):
        y = len(rows) - i
        colour = PRIMARY_COLOUR if moved else "#BBBBBB"
        ax.barh(y, change, height=0.5, color=colour, zorder=2)
        if moved:
            # U+2212, the glyph the axis ticks use; an ASCII hyphen here reads
            # as a different character at this size.
            ax.text(change - 1.2, y, f"{change:.0f}%".replace("-", "\u2212"),
                    ha="right", va="center",
                    fontsize=8.5, color="#1a1a1a", zorder=4)
        else:
            # Left of zero, where these rows have no bar, so the panel does not
            # have to carry an empty positive half to hold the note.
            ax.text(-1.2, y, "saturated \u2014 nothing to trade",
                    ha="right", va="center", fontsize=8, color="#777777",
                    style="italic", zorder=4)

    ax.set_yticks(range(1, len(rows) + 1))
    ax.set_yticklabels([f"{DISPLAY_NAME[r]}: {f}" for r, f, _, _ in reversed(rows)],
                       fontsize=9)
    ax.set_xlabel("change in factor vs. neutral control (%)")
    ax.set_xlim(-42, 2)
    ax.set_ylim(0.4, len(rows) + 0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)

    # Two roles of two factors: a light rule marks the split. The role rides in
    # the row label rather than floating in the plot area, where it crowded the
    # bars and their value labels.
    ax.axhline(2.5, color="#DDDDDD", linewidth=0.5, zorder=0)



def checks_panel(report: Dict[str, Any], decomposition: Dict[str, Any],
                 out_path: str, labels: List[str]) -> None:
    """Panels (a) the two-family effect sizes and (b) the factor decomposition.

    They are one figure because they answer one question in sequence: (a) shows
    two roles moving opposite to their instruction, and (b) forecloses the
    obvious reply that a product statistic fell through a trade rather than a
    failure. Splitting them across two floats separates a claim from its
    rebuttal and costs a caption.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9, "axes.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(TEXT_WIDTH_IN, 3.50), layout="constrained",
        gridspec_kw={"height_ratios": [1.70, 0.85]})
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.01, hspace=0.14, wspace=0.0)

    order = _forest_axes(top, report, labels)
    _factor_axes(bottom, decomposition, order)

    # Panel letters flush with the leftmost ink of either panel, which is the
    # long row labels of (b). Anchoring them to the figure canvas instead put
    # them in the middle of that label block, attached to nothing.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    left = min(ax.get_tightbbox(renderer).transformed(inv).x0
               for ax in (top, bottom))
    for ax, tag in ((top, "a)"), (bottom, "b)")):
        fig.text(left, ax.get_position().y1, tag, fontsize=9.5,
                 fontweight="bold", va="bottom", ha="left")

    # No bbox_inches: the canvas is already the text width, so the figure is
    # printed at 1:1 and the type sizes above are the type sizes on the page.
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="runs/cross_family.json")
    parser.add_argument("--out-dir", default="../paper/figures")
    parser.add_argument("--primary-label", default="gpt-5.6-terra")
    parser.add_argument("--replication-label", default="grok-4.3")
    parser.add_argument("--check-report", default="runs/checks/manipulation_report.json")
    parser.add_argument("--ablation", default="runs/arms-cca_council/ablation.json")
    parser.add_argument("--decomposition",
                        default="runs/checks/factor_decomposition.json")
    parser.add_argument("--legacy-figures", action="store_true",
                        help="also write the matplotlib method schematic and "
                             "classification grid; the manuscript draws both "
                             "in TikZ and includes neither")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.report, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    with open(args.decomposition, "r", encoding="utf-8") as handle:
        decomposition = json.load(handle)
    out = os.path.join(args.out_dir, "checks_panel.pdf")
    checks_panel(report, decomposition, out,
                 [args.primary_label, args.replication_label])
    print(f"wrote {out}")

    if args.legacy_figures:
        with open(args.check_report, "r", encoding="utf-8") as handle:
            checks = json.load(handle)
        with open(args.ablation, "r", encoding="utf-8") as handle:
            ablation = json.load(handle)
        grid = os.path.join(args.out_dir, "classification.pdf")
        classification(checks, ablation, grid)
        print(f"wrote {grid}")
        schematic = os.path.join(args.out_dir, "method.pdf")
        method(schematic)
        print(f"wrote {schematic}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
