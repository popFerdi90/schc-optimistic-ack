"""
Figure generation for Section 5.

Every figure is built from the JSON files produced by :mod:`experiments`, so the
plots and the numbers quoted in the text cannot drift apart.

Design constraints, in order of priority:

* identity is never carried by colour alone -- each series also has its own
  marker and line style, so the figures remain readable in greyscale print and
  under colour-vision deficiency;
* the categorical palette is fixed and ordered (blue, red, green, purple) and is
  never cycled or reassigned between panels;
* no dual-axis panel: quantities of different scales go into separate panels;
* 95 % confidence intervals are drawn as shaded bands on every simulated curve.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"

# Fixed categorical palette (validated: CVD dE >= 10.6, contrast >= 3:1)
C_PESS, C_OPT, C_AASS, C_ALT = "#3573B9", "#CC3311", "#009E73", "#AA4499"
STYLE = {
    "pessimistic": dict(color=C_PESS, marker="o", ls="-", label="Pessimistic AoE"),
    "optimistic": dict(color=C_OPT, marker="s", ls="--",
                       label=r"Optimistic AoE (SF12, $K_{\max}{=}3$)"),
    "aass": dict(color=C_AASS, marker="^", ls="-.", label="AASS-H (adaptive)"),
    "optimistic_rev": dict(color=C_ALT, marker="D", ls=":",
                           label=r"Optimistic AoE (SF7, $K_{\max}{=}7$)"),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _band(ax, x, series, **kw):
    """Plot a simulated curve with its 95 % confidence band."""
    m = np.array([s["mean"] for s in series])
    lo = np.array([s["ci_lo"] for s in series])
    hi = np.array([s["ci_hi"] for s in series])
    ax.plot(x, m, **kw)
    ax.fill_between(x, lo, hi, color=kw.get("color", "k"), alpha=0.18, lw=0)
    return m


def _ratio_band(ax, x, num, den, offset=0.0, scale=1.0, **kw):
    """Plot a ratio of two independently estimated means with its 95 % band.

    The two samples come from separate runs, so the relative half-widths are
    combined in quadrature (delta method) to obtain the band on the ratio.
    """
    a = np.array([d["mean"] for d in num]); b = np.array([d["mean"] for d in den])
    ra = np.array([(d["ci_hi"] - d["ci_lo"]) / 2 for d in num]) / np.maximum(a, 1e-12)
    rb = np.array([(d["ci_hi"] - d["ci_lo"]) / 2 for d in den]) / np.maximum(b, 1e-12)
    ratio = a / b
    hw = ratio * np.sqrt(ra ** 2 + rb ** 2)
    y = (ratio + offset) * scale
    ax.plot(x, y, **kw)
    ax.fill_between(x, (ratio - hw + offset) * scale, (ratio + hw + offset) * scale,
                    color=kw.get("color", "k"), alpha=0.18, lw=0)
    return y


P_FEASIBLE = 0.40   # largest p_u at which pessimistic AoE still delivers >= 95 %


def _shade_infeasible(ax, label=True):
    """Grey out the region where SCHC F/R itself stops delivering."""
    lo, hi = ax.get_xlim()
    ax.axvspan(P_FEASIBLE, hi, color="0.85", alpha=0.45, lw=0, zorder=0)
    if label:
        ax.text(P_FEASIBLE + 0.015, ax.get_ylim()[1], "delivery ratio $<0.95$",
                fontsize=6.5, color="0.35", rotation=90, va="top", ha="left")
    ax.set_xlim(lo, hi)


def _legend_below(ax, ncol=2, dy=-0.24, **kw):
    """Legend outside the axes, under the x label, so it can never hide a curve."""
    return ax.legend(frameon=False, loc="upper center",
                     bbox_to_anchor=(0.5, dy), ncol=ncol, **kw)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"  -> figs/{name}.pdf")


# ---------------------------------------------------------------------------

def fig_validation():
    d = _load("validation")
    rows = d["rows"]
    x = [r["p_u"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0))
    axes = axes.ravel()

    ax = axes[0]
    _band(ax, x, [r["sim_pess"]["n_ack_tx"] for r in rows],
          color=C_PESS, marker="o", ls="none", label="Simulation")
    ax.plot(x, [r["eq17_ack_pess_full"] for r in rows], color=C_OPT, ls="-",
            label="Model, Eq. (7)")
    ax.set_yscale("log")
    ax.set_xlabel(r"uplink loss probability $p_u$")
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{ACK}]$")
    ax.set_title("(a) Pessimistic AoE")
    ax.legend(frameon=False, loc="upper left")
    _shade_infeasible(ax)

    ax = axes[1]
    _band(ax, x, [r["sim_opt"]["n_ack_tx"] for r in rows],
          color=C_PESS, marker="s", ls="none", label="Simulation")
    ax.plot(x, [r["eq7_ack_opt"] for r in rows], color="0.35", ls="--",
            label="Single-round treatment")
    ax.plot(x, [r["eq16_ack_opt"] for r in rows], color=C_OPT, ls="-",
            label="Model, Eq. (10)")
    ax.set_xlabel(r"uplink loss probability $p_u$")
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{ACK}]$")
    ax.set_title("(b) Optimistic AoE")
    ax.legend(frameon=False, loc="upper left")
    _shade_infeasible(ax, label=False)

    ax = axes[2]
    _band(ax, x, [r["sim_opt"]["false_ack"] for r in rows],
          color=C_PESS, marker="^", ls="none", label="Simulation")
    ax.plot(x, [r["eq14_false_ack"] for r in rows], color="0.35", ls="--",
            label="Single-round treatment")
    ax.plot(x, [r["eq17_false_ack"] for r in rows], color=C_OPT, ls="-",
            label="Model, Eq. (11)")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2.0)
    ax.set_xlabel(r"uplink loss probability $p_u$")
    ax.set_ylabel(r"$P_{\mathrm{falseACK}}^{\mathrm{opt}}$")
    ax.set_title("(c) False-ACK probability")
    ax.legend(frameon=False, loc="lower right")
    _shade_infeasible(ax, label=False)

    ax = axes[3]
    _band(ax, x, [r["sim_pess"]["delay_model"] for r in rows],
          color=C_PESS, marker="o", ls="none", label="Simulation, pessimistic")
    ax.plot(x, [r["eq23_delay_pess"] for r in rows], color=C_PESS, ls="-",
            label="Model, Eq. (12), pessimistic")
    _band(ax, x, [r["sim_opt"]["delay_model"] for r in rows],
          color=C_OPT, marker="s", ls="none", label="Simulation, optimistic")
    ax.plot(x, [r["eq23_delay_opt"] for r in rows], color=C_OPT, ls="-",
            label="Model, Eq. (12), optimistic")
    ax.set_xlabel(r"uplink loss probability $p_u$")
    ax.set_ylabel("protocol-level transfer time (s)")
    ax.set_ylim(0, 95)   # the curves diverge beyond the feasible region
    ax.set_title("(d) Transfer delay")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _shade_infeasible(ax, label=False)

    fig.tight_layout()
    _save(fig, "fig9_validation")


def fig_comparison():
    d = _load("comparison")
    rows = d["rows"]
    x = [r["p_u"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))

    ax = axes[0, 0]
    for key in ("pessimistic", "optimistic", "optimistic_rev", "aass"):
        _band(ax, x, [r[key]["n_downlink_tx"] for r in rows], **STYLE[key])
    ax.set_ylabel("downlink frames per SCHC Packet")
    ax.set_title("(a) Downlink traffic")

    ax = axes[0, 1]
    for key in ("pessimistic", "optimistic", "optimistic_rev", "aass"):
        _band(ax, x, [r[key]["delivered"] for r in rows], **STYLE[key])
    ax.set_ylabel("delivery ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Reliability")

    ax = axes[1, 0]
    for key in ("pessimistic", "optimistic", "optimistic_rev", "aass"):
        _band(ax, x, [r[key]["energy_j"] for r in rows], **STYLE[key])
    ax.set_ylabel("node radio energy per packet (J)")
    ax.set_title("(c) Energy")

    ax = axes[1, 1]
    for key in ("pessimistic", "optimistic", "optimistic_rev", "aass"):
        _band(ax, x, [r[key]["delay_realistic"] for r in rows], **STYLE[key])
    ax.set_ylabel("transfer time with duty cycle (s)")
    ax.set_title("(d) Latency under EU868 duty cycle")

    for ax in axes.ravel():
        ax.set_xlabel(r"uplink loss probability $p_u$")
        _shade_infeasible(ax, label=False)
    axes[0, 1].axhline(0.95, color="0.45", lw=0.8, ls=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig10_comparison")


def fig_robustness():
    ge = _load("gilbert_elliott")["rows"]
    al = _load("alpha")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

    ax = axes[0]
    x = [r["p_u"] for r in ge]
    specs = (("iid", C_PESS, "o", "-", "i.i.d. losses"),
             ("ge_L3", C_OPT, "s", "--", r"Gilbert-Elliott, $\bar{L}_B=3$"),
             ("ge_L6", C_AASS, "^", "-.", r"Gilbert-Elliott, $\bar{L}_B=6$"))
    for tag, c, mk, ls, lab in specs:
        _band(ax, x, [r["models"][tag]["optimistic"]["false_ack"] for r in ge],
              color=c, marker=mk, ls=ls, label=lab)
    ax.set_xlabel(r"mean uplink loss probability $p_u$")
    ax.set_ylabel(r"$P_{\mathrm{falseACK}}^{\mathrm{opt}}$")
    ax.set_title("(a) Effect of correlated losses")
    _legend_below(ax, ncol=2, fontsize=7.5)

    ax = axes[1]
    alphas = sorted({r["alpha"] for r in al})
    for i, p_u in enumerate(sorted({r["p_u"] for r in al})):
        sub = [r for r in al if r["p_u"] == p_u]
        sub.sort(key=lambda r: r["alpha"])
        c, mk, ls = [(C_PESS, "o", "-"), (C_OPT, "s", "--"),
                     (C_AASS, "^", "-."), (C_ALT, "D", ":")][i]
        _band(ax, alphas, [r["sim"]["false_ack"] for r in sub],
              color=c, marker=mk, ls=ls, label=rf"$p_u={p_u}$ (simulation)")
        ax.plot(alphas, [r["eq17_false_ack"] for r in sub],
                color=c, ls="none", marker="x", ms=4, alpha=0.8)
    ax.set_xlabel(r"asymmetry factor $\alpha$  ($p_d=p_{u,\mathrm{All\text{-}1}}=\alpha p_u$)")
    ax.set_ylabel(r"$P_{\mathrm{falseACK}}^{\mathrm{opt}}$")
    ax.set_title(r"(b) Sensitivity to $\alpha$")
    ax.axvline(0.5, color="0.5", lw=0.8, ls=":")
    _legend_below(ax, ncol=2, fontsize=7.5)
    ax.text(0.47, 0.98, r"$\alpha=0.5$ (Sec. 4.2)", fontsize=7, color="0.4",
            rotation=90, va="top", ha="right")

    fig.tight_layout()
    _save(fig, "fig11_robustness")


def fig_energy():
    d = _load("energy_budget")
    v = d["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    specs = ((12, C_OPT, "s", "--", "All-1 hardened at SF12"),
             (7, C_PESS, "o", "-", "All-1 kept at SF7 (ADR)"))

    ax = axes[0]
    for sf, c, mk, ls, lab in specs:
        sub = sorted([r for r in v if r["sf_all1"] == sf and r["p_u"] == 0.3],
                     key=lambda r: r["kmax"])
        _ratio_band(ax, [r["kmax"] for r in sub],
                    [r["optimistic"]["energy_j"] for r in sub],
                    [r["pessimistic"]["energy_j"] for r in sub],
                    offset=-1.0, scale=100.0,
                    color=c, marker=mk, ls=ls, label=lab)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$K_{\max}$")
    ax.set_ylabel("node energy vs pessimistic AoE (%)")
    ax.set_title(r"(a) Energy, $p_u=0.3$")
    ax.legend(frameon=False, fontsize=7.5)

    ax = axes[1]
    for sf, c, mk, ls, lab in specs:
        sub = sorted([r for r in v if r["sf_all1"] == sf and r["p_u"] == 0.3],
                     key=lambda r: r["kmax"])
        _band(ax, [r["kmax"] for r in sub],
              [r["optimistic"]["false_ack"] for r in sub],
              color=c, marker=mk, ls=ls, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"$K_{\max}$")
    ax.set_ylabel(r"$P_{\mathrm{falseACK}}^{\mathrm{opt}}$")
    ax.set_title(r"(b) Reliability, $p_u=0.3$")
    ax.legend(frameon=False, fontsize=7.5)

    fig.tight_layout()
    _save(fig, "fig12_energy")


def fig_aass():
    d = _load("aass")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))

    ax = axes[0]
    for tag, c, ls, lab in (("v1", C_PESS, "-", "AASS (moving average)"),
                            ("v2", C_OPT, "--", "AASS-H (EWMA + hysteresis)")):
        ph = np.array(d["step"][tag]["phat"])
        ax.plot(np.arange(ph.size), ph, color=c, ls=ls, lw=1.0, alpha=0.9,
                label=rf"{lab}: {d['step'][tag]['n_switches']:.0f} switches (mean of {d['step'][tag]['n_rep']})")
        mode = np.array(d["step"][tag]["ctrl_optimistic"])
        y = 0.90 if tag == "v1" else 0.86
        ax.fill_between(np.arange(mode.size), y, y + 0.022,
                        where=mode > 0.5, color=c, alpha=0.55, lw=0, step="mid")
        ax.text(905, y + 0.011, "optimistic", fontsize=6, color=c,
                ha="left", va="center")
    prof = np.concatenate([np.full(300, 0.10), np.full(300, 0.60), np.full(300, 0.10)])
    ax.plot(np.arange(prof.size), prof, color="0.35", ls=":", lw=1.0,
            label=r"true $p_u$")
    ax.axhline(0.3, color="0.6", lw=0.8)
    ax.text(5, 0.315, r"$p_{tu}=0.3$", fontsize=7, color="0.4")
    ax.set_xlabel("packet index")
    ax.set_ylabel(r"estimated $\hat{p}_u$")
    ax.set_ylim(0.0, 0.95)
    ax.set_title("(a) Response to a channel step")
    ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=7,
              bbox_to_anchor=(0.01, 0.83))

    ax = axes[1]
    th = d["thresholds"]
    ax.semilogx([t["lambda"] for t in th], [t["p_tu_star"] for t in th],
                color=C_OPT, marker="s", ls="--", label=r"$p_{tu}^{\star}(\lambda)$")
    ax.axhline(0.3, color="0.5", lw=0.8, ls=":")
    ax.text(1.2, 0.315, r"$p_{tu}=0.3$ (Table 8)", fontsize=7, color="0.4")
    ax.set_xlabel(r"price of one false ACK, $\lambda$ (in ACK units)")
    ax.set_ylabel(r"cost-optimal threshold $p_{tu}^{\star}$")
    ax.set_title("(b) Cost-optimal switching threshold")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _save(fig, "fig13_aass")


def fig_adversary():
    d = _load("adversary")["rows"]
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for i, p_u in enumerate(sorted({r["p_u"] for r in d})):
        sub = sorted([r for r in d if r["p_u"] == p_u], key=lambda r: r["p_jam"])
        x = [r["p_jam"] for r in sub]
        ls = "--" if i == 0 else "-."
        mk = "s" if i == 0 else "D"
        _band(ax, x, [r["optimistic"]["false_ack"] for r in sub],
              color=C_OPT, marker=mk, ls=ls,
              label=rf"Optimistic: silent loss, $p_u={p_u}$")
        _band(ax, x, [r["pessimistic"]["aborted"] for r in sub],
              color=C_PESS, marker="o" if i == 0 else "v", ls=ls,
              label=rf"Pessimistic: detected abort, $p_u={p_u}$")
    ax.set_xlabel(r"selective jamming probability $p_{\mathrm{jam}}$")
    ax.set_ylabel("probability of transfer failure")
    ax.set_title("Active adversary targeting All-1 and ACKs")
    _legend_below(ax, ncol=2, dy=-0.22, fontsize=6.5, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig14_adversary")


FIGURES = {
    "validation": fig_validation,
    "comparison": fig_comparison,
    "robustness": fig_robustness,
    "energy": fig_energy,
    "aass": fig_aass,
    "adversary": fig_adversary,
}


def main():
    import sys
    names = sys.argv[1:] or list(FIGURES)
    for n in names:
        FIGURES[n]()


if __name__ == "__main__":
    main()
