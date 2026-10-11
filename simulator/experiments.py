"""
Experiment campaign backing Section 5 of the paper.

Every experiment writes a JSON file into ``results/``; the figures and the
tables of the manuscript are generated from those files only, so that the whole
evaluation is reproducible from a single entry point::

    python -m simulator.experiments --all

Random seeds are derived deterministically from the experiment name and the
configuration index, so re-running the campaign reproduces the published
numbers bit for bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import analytic as an
from . import lora_phy as phy
from . import stats as st
from .aass import AASSv1, AASSv2, simulate_stream, optimal_threshold, switching_cost
from .schc_aoe import SchcConfig, Timing, simulate_batch, simulate_packet

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_RUNS = 10_000
N_REP_STREAM = 200          # independent streams per controller setting (Sec. 5.4)
P_GRID = [round(0.10 + 0.05 * i, 3) for i in range(17)]   # 0.10 ... 0.90
CFG = SchcConfig(max_rounds=60)                    # proposed design: All-1 at SF12
TIM = Timing.from_config(CFG)

# Revised design recommended in Section 4.3: the All-1 fragment keeps the
# ADR-managed data rate and is instead repeated more often.
CFG_R = SchcConfig(max_rounds=60, kmax=7, sf_all1_opt=7)
TIM_R = Timing.from_config(CFG_R)


def _seed(tag: str, k: int = 0) -> int:
    """Deterministic seed from an experiment name and a configuration index.

    Python salts ``hash()`` of a string differently in every process, so a
    stable digest is used instead: without it the campaign would not reproduce
    across runs, which is the property claimed in the reproducibility statement of Section 5.1.
    """
    digest = hashlib.blake2b(tag.encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") % (2 ** 31)) ^ (k * 7919 + 17)


def _model_params() -> an.ModelParams:
    """Analytical parameters, with timings taken from the LoRaWAN PHY model."""
    t_a = phy.RX2_DELAY + TIM.toa_ack
    return an.ModelParams(
        F=CFG.F, w=CFG.w, w_last=CFG.w_last, W=CFG.W,
        kmax=CFG.kmax, M=CFG.M, alpha=0.5,
        T_f=TIM.toa_frag,
        T_f_all1=TIM.toa_all1_opt,
        T_a=t_a,
        T_req=TIM.toa_ackreq,
        T_r_all1=t_a,
        T_r_req=t_a,
        T_w=phy.RX2_DELAY + TIM.listen_rx2,
    )


def _summarise(raw: dict, keys=("n_ack_tx", "n_downlink_tx", "n_all1_tx",
                                "n_frag_tx", "n_ackreq_tx",
                                "delay_model", "delay_protocol",
                                "delay_realistic", "energy_j", "toa_up_s",
                                "toa_down_s")) -> dict:
    out = {}
    for k in keys:
        m, lo, hi = st.mean_ci(raw[k])
        out[k] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    n = raw["delivered"].size
    for k in ("delivered", "false_ack", "aborted"):
        p, lo, hi = st.proportion_ci(int(raw[k].sum()), n)
        out[k] = {"mean": p, "ci_lo": lo, "ci_hi": hi}
    out["n_runs"] = n
    return out


def _dump(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=float))
    print(f"  -> {path.name}")
    return path


# ---------------------------------------------------------------------------
# E1 -- validation of the analytical model
# ---------------------------------------------------------------------------

def exp_validation() -> dict:
    prm = _model_params()
    rows = []
    for k, p_u in enumerate(P_GRID):
        p_d = prm.alpha * p_u
        pess = simulate_batch("pessimistic", p_u, N_RUNS, CFG,
                              _seed("val-p", k), tim=TIM)
        opt = simulate_batch("optimistic", p_u, N_RUNS, CFG,
                             _seed("val-o", k), tim=TIM)
        row = {
            "p_u": p_u, "p_d": p_d,
            "sim_pess": _summarise(pess),
            "sim_opt": _summarise(opt),
            "eq4_ack_pess": an.e_ack_pessimistic(p_u, p_d, prm),
            "eq4_published_table": an.e_ack_pessimistic_simplified(p_u, prm),
            "eq18_ack_pess": an.e_ack_pessimistic_corrected(p_u, p_d, prm),
            "eq18r_ack_pess": an.e_ack_pessimistic_renewal(p_u, p_d, prm),
            "eq17_ack_pess_full": an.e_ack_pessimistic_full(p_u, p_d, prm),
            "eq7_ack_opt": an.e_ack_optimistic(p_u, p_d, prm),
            "eq16_ack_opt": an.e_ack_optimistic_multiround(p_u, p_d, prm),
            "eq9_delay_pess": an.e_delay_pessimistic(p_u, p_d, prm),
            "eq12_delay_opt": an.e_delay_optimistic(p_u, p_d, prm),
            "eq23_delay_opt": an.e_delay_optimistic_corrected(p_u, p_d, prm),
            "eq23_delay_pess": an.e_delay_pessimistic_corrected(
                p_u, p_d, prm, TIM.toa_all1_pess),
            "delay_counts_opt": an.delay_counts_optimistic(p_u, p_d, prm),
            "delay_counts_pess": an.delay_counts_pessimistic(p_u, p_d, prm),
            "eq14_false_ack": an.p_false_ack_optimistic(p_u, p_d, prm),
            "eq17_false_ack": an.p_false_ack_multiround(p_u, p_d, prm),
            "eq19_mu_deferred": an.p_deferred_repair(p_u, p_d),
        }
        rows.append(row)
        print(f"   p_u={p_u:.2f}  ACKpess={row['sim_pess']['n_ack_tx']['mean']:6.2f}"
              f"  ACKopt={row['sim_opt']['n_ack_tx']['mean']:6.2f}"
              f"  FA={row['sim_opt']['false_ack']['mean']:.3f}")
    return {"n_runs": N_RUNS, "params": asdict(prm), "rows": rows}




# ---------------------------------------------------------------------------
# E2 -- comparative performance, including AASS
# ---------------------------------------------------------------------------

def exp_comparison() -> dict:
    prm = _model_params()
    rows = []
    for k, p_u in enumerate(P_GRID):
        pess = simulate_batch("pessimistic", p_u, N_RUNS, CFG,
                              _seed("cmp", k), tim=TIM)
        opt = simulate_batch("optimistic", p_u, N_RUNS, CFG,
                             _seed("cmp", k), tim=TIM)   # common random numbers
        optr = simulate_batch("optimistic", p_u, N_RUNS, CFG_R,
                              _seed("cmp", k), tim=TIM_R)
        aass = _aass_steady(p_u, k)
        rows.append({
            "p_u": p_u,
            "pessimistic": _summarise(pess),
            "optimistic": _summarise(opt),
            "optimistic_rev": _summarise(optr),
            "aass": aass,
            "tests": {
                "ack": st.paired_test(opt["n_ack_tx"], pess["n_ack_tx"]),
                "downlink": st.paired_test(opt["n_downlink_tx"], pess["n_downlink_tx"]),
                "energy": st.paired_test(opt["energy_j"], pess["energy_j"]),
                "delay": st.paired_test(opt["delay_realistic"], pess["delay_realistic"]),
            },
            "ack_reduction_pct": 100.0 * (1.0 - opt["n_ack_tx"].mean()
                                          / max(pess["n_ack_tx"].mean(), 1e-12)),
            "downlink_reduction_pct": 100.0 * (1.0 - opt["n_downlink_tx"].mean()
                                               / max(pess["n_downlink_tx"].mean(), 1e-12)),
            "energy_overhead_pct": 100.0 * (opt["energy_j"].mean()
                                            / pess["energy_j"].mean() - 1.0),
            "energy_overhead_rev_pct": 100.0 * (optr["energy_j"].mean()
                                                / pess["energy_j"].mean() - 1.0),
            "downlink_reduction_rev_pct": 100.0 * (1.0 - optr["n_downlink_tx"].mean()
                                                   / max(pess["n_downlink_tx"].mean(), 1e-12)),
        })
        print(f"   p_u={p_u:.2f}  ACK -{rows[-1]['ack_reduction_pct']:5.1f}%"
              f"  DL -{rows[-1]['downlink_reduction_pct']:5.1f}%"
              f"  E {rows[-1]['energy_overhead_pct']:+6.1f}%"
              f"  | revised: DL -{rows[-1]['downlink_reduction_rev_pct']:5.1f}%"
              f"  E {rows[-1]['energy_overhead_rev_pct']:+6.1f}%"
              f"  FA {rows[-1]['optimistic_rev']['false_ack']['mean']:.3f}")
    return {"n_runs": N_RUNS, "rows": rows}


def _aass_steady(p_u: float, k: int, n_packets: int = 2000,
                 warmup: int = 200, controller: str = "v2") -> dict:
    ctrl = AASSv2() if controller == "v2" else AASSv1()
    prof = np.full(n_packets, p_u)
    rec = simulate_stream(ctrl, prof, CFG, _seed(f"aass-{controller}", k))
    sl = slice(warmup, None)
    out = {}
    for key, name in (("ack", "n_ack_tx"), ("downlink", "n_downlink_tx"),
                      ("energy", "energy_j"), ("delay", "delay_realistic")):
        m, lo, hi = st.mean_ci(rec[key][sl])
        out[name] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
    for key, name in (("false_ack", "false_ack"), ("delivered", "delivered")):
        arr = rec[key][sl]
        p, lo, hi = st.proportion_ci(int(arr.sum()), arr.size)
        out[name] = {"mean": p, "ci_lo": lo, "ci_hi": hi}
    out["optimistic_share"] = float(rec["optimistic"][sl].mean())
    out["n_switches"] = float(rec["n_switches"])
    out["n_runs"] = int(n_packets - warmup)
    return out


# ---------------------------------------------------------------------------
# E3 -- correlated (Gilbert-Elliott) losses
# ---------------------------------------------------------------------------

def exp_gilbert_elliott() -> dict:
    rows = []
    for k, p_u in enumerate(P_GRID):
        entry = {"p_u": p_u, "models": {}}
        for tag, chan, burst in (("iid", "iid", 1.0),
                                 ("ge_L3", "ge", 3.0),
                                 ("ge_L6", "ge", 6.0)):
            pess = simulate_batch("pessimistic", p_u, N_RUNS // 2, CFG,
                                  _seed(f"ge-p-{tag}", k), channel=chan,
                                  burst_len=burst, tim=TIM)
            opt = simulate_batch("optimistic", p_u, N_RUNS // 2, CFG,
                                 _seed(f"ge-o-{tag}", k), channel=chan,
                                 burst_len=burst, tim=TIM)
            entry["models"][tag] = {"pessimistic": _summarise(pess),
                                    "optimistic": _summarise(opt)}
        rows.append(entry)
        m = entry["models"]
        print(f"   p_u={p_u:.2f}  FA iid={m['iid']['optimistic']['false_ack']['mean']:.3f}"
              f"  L3={m['ge_L3']['optimistic']['false_ack']['mean']:.3f}"
              f"  L6={m['ge_L6']['optimistic']['false_ack']['mean']:.3f}")
    return {"n_runs": N_RUNS // 2, "rows": rows}


# ---------------------------------------------------------------------------
# E4 -- sensitivity to alpha
# ---------------------------------------------------------------------------

def exp_alpha() -> dict:
    prm = _model_params()
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for k, alpha in enumerate(alphas):
        for j, p_u in enumerate((0.1, 0.2, 0.3, 0.5)):
            p_d = min(alpha * p_u, 0.99)
            opt = simulate_batch("optimistic", p_u, N_RUNS // 2, CFG,
                                 _seed("alpha", k * 11 + j), alpha=alpha, tim=TIM)
            prm_a = an.ModelParams(**{**asdict(prm), "alpha": alpha})
            rows.append({
                "alpha": alpha, "p_u": p_u, "p_d": p_d,
                "sim": _summarise(opt),
                "eq14_false_ack": an.p_false_ack_optimistic(p_u, p_d, prm_a),
                "eq17_false_ack": an.p_false_ack_multiround(p_u, p_d, prm_a),
                "eq16_ack": an.e_ack_optimistic_multiround(p_u, p_d, prm_a),
            })
        print(f"   alpha={alpha:.2f} done")
    return {"n_runs": N_RUNS // 2, "rows": rows}


# ---------------------------------------------------------------------------
# E5 -- Time-on-Air, duty cycle and energy balance
# ---------------------------------------------------------------------------

def exp_energy_budget() -> dict:
    """Airtime / duty-cycle / energy budget of the All-1 hardening strategy."""
    em = phy.DEFAULT_ENERGY
    toa_tbl = {}
    for sf in (7, 9, 12):
        toa_tbl[f"SF{sf}"] = {
            "toa_frag_51B": phy.toa_uplink(CFG.b_frag, sf),
            "toa_all1_5B": phy.toa_uplink(CFG.b_all1, sf),
            "toa_ack_4B_dl": phy.toa_downlink(CFG.b_ack, sf),
            "toff_all1_1pct": phy.time_off(phy.toa_uplink(CFG.b_all1, sf)),
            "e_tx_all1_mJ": 1e3 * em.e_tx(phy.toa_uplink(CFG.b_all1, sf)),
        }

    variants = []
    for kmax in (1, 2, 3, 5, 7):
        for sf_all1 in (7, 12):
            cfg = SchcConfig(max_rounds=60, kmax=kmax, sf_all1_opt=sf_all1)
            tim = Timing.from_config(cfg)
            for p_u in (0.1, 0.3, 0.5):
                opt = simulate_batch("optimistic", p_u, N_RUNS // 2, cfg,
                                     _seed(f"en-{kmax}-{sf_all1}", int(p_u * 100)),
                                     tim=tim)
                pess = simulate_batch("pessimistic", p_u, N_RUNS // 2, cfg,
                                      _seed(f"enp-{kmax}-{sf_all1}", int(p_u * 100)),
                                      tim=tim)
                variants.append({
                    "kmax": kmax, "sf_all1": sf_all1, "p_u": p_u,
                    "optimistic": _summarise(opt),
                    "pessimistic": _summarise(pess),
                    "energy_ratio": opt["energy_j"].mean() / pess["energy_j"].mean(),
                    "airtime_ratio": opt["toa_up_s"].mean() / pess["toa_up_s"].mean(),
                    "downlink_ratio": (opt["n_downlink_tx"].mean()
                                       / max(pess["n_downlink_tx"].mean(), 1e-12)),
                })
    print(f"   {len(variants)} (KMAX, SF, p_u) configurations evaluated")
    return {"toa_table": toa_tbl, "variants": variants,
            "duty_cycle": CFG.duty_cycle, "n_runs": N_RUNS // 2}


# ---------------------------------------------------------------------------
# E6 -- AASS: reactivity, stability and optimal threshold
# ---------------------------------------------------------------------------

def exp_aass() -> dict:
    prm = _model_params()

    # (a) cost-optimal threshold as a function of the price of a false ACK
    lambdas = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    thresholds = [{"lambda": lam, "p_tu_star": optimal_threshold(prm, lam)}
                  for lam in lambdas]

    # (b) reactivity to a step change of the channel: 10 % -> 60 % -> 10 %
    n = 900
    prof = np.concatenate([np.full(300, 0.10), np.full(300, 0.60),
                           np.full(300, 0.10)])
    step = {}
    for tag, factory in (("v1", AASSv1), ("v2", AASSv2)):
        agg = {"sw": [], "ad": [], "fas": [], "fat": [], "dl": [], "de": [], "en": []}
        for rep in range(N_REP_STREAM):
            rr = simulate_stream(factory(), prof, CFG, _seed(f"step-{tag}", rep))
            ad = _adaptation_delay(rr["ctrl_optimistic"], 300)
            agg["sw"].append(rr["n_switches"]); agg["ad"].append(ad if ad >= 0 else 300)
            agg["fas"].append(rr["false_ack"][300:600].mean())
            agg["fat"].append(rr["false_ack"].mean())
            agg["dl"].append(rr["downlink"].sum()); agg["de"].append(rr["delivered"].mean())
            agg["en"].append(rr["energy"].sum())
        # The traces drawn in Fig. 8a come from one representative stream, not
        # from stream 0: see _representative_rep.
        trace_rep = _representative_rep(agg["sw"], agg["ad"])
        rec = simulate_stream(factory(), prof, CFG, _seed(f"step-{tag}", trace_rep))
        step[tag] = {
            "phat": rec["phat"].tolist(),
            "optimistic": rec["optimistic"].tolist(),
            "false_ack": rec["false_ack"].tolist(),
            "n_switches": float(np.mean(agg["sw"])),
            "n_switches_ci": list(st.mean_ci(np.array(agg["sw"]))[1:]),
            "n_probes": rec.get("n_probes", 0.0),
            "n_rep": N_REP_STREAM,
            "trace_rep": trace_rep,
            "trace_switches": float(agg["sw"][trace_rep]),
            "trace_delay": float(agg["ad"][trace_rep]),
            "ctrl_optimistic": rec["ctrl_optimistic"].tolist(),
            "adaptation_delay_pkts": float(np.mean(agg["ad"])),
            "adaptation_delay_ci": list(st.mean_ci(np.array(agg["ad"]))[1:]),
            "false_ack_after_step": float(np.mean(agg["fas"])),
            "false_ack_after_step_ci": list(st.mean_ci(np.array(agg["fas"]))[1:]),
            "false_ack_total": float(np.mean(agg["fat"])),
            "delivered_total": float(np.mean(agg["de"])),
            "downlink_total": float(np.mean(agg["dl"])),
            "energy_total": float(np.mean(agg["en"])),
        }
        print(f"   AASS-{tag}: switches={rec['n_switches']:.0f}"
              f"  adapt={step[tag]['adaptation_delay_pkts']} pkts"
              f"  FA(step)={step[tag]['false_ack_after_step']:.3f}")

    # (b') stability / reactivity trade-off of the EWMA and hysteresis settings
    tuning, n_rep = [], N_REP_STREAM
    for k, beta in enumerate((0.70, 0.80, 0.90, 0.95)):
        for h in (0.0, 0.05, 0.10):
            acc = {"sw": [], "ad": [], "fas": [], "fat": [], "dl": []}
            for rep in range(n_rep):
                rec = simulate_stream(AASSv2(beta=beta, h=h), prof, CFG,
                                      _seed(f"tune-{beta}-{h}", rep))
                ad = _adaptation_delay(rec["ctrl_optimistic"], 300)
                acc["sw"].append(rec["n_switches"])
                acc["ad"].append(ad if ad >= 0 else 300)   # never adapted
                acc["fas"].append(rec["false_ack"][300:600].mean())
                acc["fat"].append(rec["false_ack"].mean())
                acc["dl"].append(rec["downlink"].sum())
            ci = {k2: st.mean_ci(np.array(v)) for k2, v in acc.items()}
            tuning.append({
                "beta": beta, "h": h, "n_rep": n_rep,
                "n_switches": ci["sw"][0],
                "adaptation_delay_pkts": ci["ad"][0],
                "adaptation_delay_ci": [ci["ad"][1], ci["ad"][2]],
                "false_ack_after_step": ci["fas"][0],
                "false_ack_after_step_ci": [ci["fas"][1], ci["fas"][2]],
                "false_ack_total": ci["fat"][0],
                "downlink_total": ci["dl"][0],
                "never_adapted": float(np.mean([a >= 300 for a in acc["ad"]])),
            })
    print(f"   tuning grid: {len(tuning)} (beta, h) settings")

    # (c) sensitivity of the steady state to the threshold value
    sweep = []
    for k, p_tu in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
        for p_u in (0.15, 0.35, 0.55):
            ctrl = AASSv2(p_tu=p_tu)
            rec = simulate_stream(ctrl, np.full(800, p_u), CFG,
                                  _seed("sweep", k * 13 + int(p_u * 100)))
            sl = slice(100, None)
            sweep.append({
                "p_tu": p_tu, "p_u": p_u,
                "optimistic_share": float(rec["optimistic"][sl].mean()),
                "ack": float(rec["ack"][sl].mean()),
                "downlink": float(rec["downlink"][sl].mean()),
                "false_ack": float(rec["false_ack"][sl].mean()),
                "delivered": float(rec["delivered"][sl].mean()),
                "n_switches": float(rec["n_switches"]),
            })

    # (d) benefit curve used to derive p_tu*
    grid = np.linspace(0.02, 0.9, 89)
    benefit = {str(lam): [switching_cost(float(p), prm, lam) for p in grid]
               for lam in (1, 10, 100)}
    return {"thresholds": thresholds, "step": step, "tuning": tuning, "sweep": sweep,
            "benefit_grid": grid.tolist(), "benefit": benefit}


def _representative_rep(switches, delays) -> int:
    """Index of the stream shown in Fig. 8a.

    Among the streams whose number of mode changes is within a small tolerance
    of the mean over all streams (1.1 changes for a mean below 10, 3 otherwise;
    the count is even for AASS-H, so 2 and 4 both qualify around a mean of 2.97),
    take the one whose delay in leaving the optimistic mode is closest to the
    mean delay. Ties go to the lowest index, so the choice is deterministic.
    """
    sw = np.asarray(switches, float)
    ad = np.asarray(delays, float)
    tol = 1.1 if sw.mean() < 10 else 3.0
    ok = np.flatnonzero(np.abs(sw - sw.mean()) <= tol)
    if ok.size == 0:                       # cannot happen with 200 streams
        ok = np.arange(sw.size)
    return int(ok[np.argmin(np.abs(ad[ok] - ad.mean()))])


def _adaptation_delay(optimistic: np.ndarray, step_index: int) -> int:
    """Packets needed to leave optimistic mode after the channel degrades."""
    after = optimistic[step_index:]
    idx = np.where(after == 0.0)[0]
    return int(idx[0]) if idx.size else -1


# ---------------------------------------------------------------------------
# E7 -- active adversary
# ---------------------------------------------------------------------------

def exp_adversary() -> dict:
    rows = []
    for k, p_jam in enumerate((0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
        for p_u in (0.1, 0.3):
            jam = {"all1": p_jam, "ack": p_jam}
            opt = simulate_batch("optimistic", p_u, N_RUNS // 2, CFG,
                                 _seed("adv-o", k * 7 + int(p_u * 10)),
                                 jam=jam, tim=TIM)
            pess = simulate_batch("pessimistic", p_u, N_RUNS // 2, CFG,
                                  _seed("adv-p", k * 7 + int(p_u * 10)),
                                  jam=jam, tim=TIM)
            rows.append({"p_jam": p_jam, "p_u": p_u,
                         "optimistic": _summarise(opt),
                         "pessimistic": _summarise(pess)})
        print(f"   p_jam={p_jam:.1f}  FA_opt={rows[-1]['optimistic']['false_ack']['mean']:.3f}"
              f"  abort_pess={rows[-1]['pessimistic']['aborted']['mean']:.3f}")
    return {"n_runs": N_RUNS // 2, "rows": rows}


# ---------------------------------------------------------------------------
# E8 -- sensitivity to the probing period of AASS-H
# ---------------------------------------------------------------------------

def exp_probing() -> dict:
    """Effect of the probing period N_probe on cost, risk and estimator bias.

    Probing exists because, in optimistic mode, the sender is only given a
    bitmap when an error is both detected and successfully reported.  Two
    scenarios are therefore simulated on the same 900-packet step profile
    (p_u: 0.10 -> 0.60 -> 0.10):

    * ``natural``    -- the channel degrades on its own; feedback is plentiful
      once it does, so probing is expected to matter little;
    * ``jammed``     -- an adversary suppresses the acknowledgements during the
      degraded phase, which is precisely the state that starves the estimator.

    A steady-state run on a good channel is added to expose the downlink floor
    that probing imposes, namely one acknowledgement every ``N_probe`` packets.
    """
    prof = np.concatenate([np.full(300, 0.10), np.full(300, 0.60),
                           np.full(300, 0.10)])
    jam_prof = [None] * 300 + [{"ack": 0.9}] * 300 + [None] * 300
    periods = (5, 10, 20, 40, 80, 0)           # 0 disables probing
    n_rep = N_REP_STREAM
    rows = []
    for k, period in enumerate(periods):
        entry = {"probe_period": period, "n_rep": n_rep}
        for tag, jam in (("natural", None), ("jammed", jam_prof)):
            acc = {"dl": [], "fa": [], "err": [], "deliv": [], "opt": []}
            for rep in range(n_rep):
                rec = simulate_stream(AASSv2(probe_period=period), prof, CFG,
                                      _seed(f"probe-{period}-{tag}", rep),
                                      jam_profile=jam)
                sl = slice(300, 600)
                acc["dl"].append(rec["downlink"].mean())
                acc["fa"].append(rec["false_ack"][sl].mean())
                # how far the estimate sits from the truth while degraded
                acc["err"].append(np.abs(rec["phat"][sl] - prof[sl]).mean())
                acc["deliv"].append(rec["delivered"][sl].mean())
                acc["opt"].append(rec["ctrl_optimistic"][sl].mean())
            entry[tag] = {kk: st.mean_ci(np.array(v))[0] for kk, v in acc.items()}
            entry[tag]["fa_ci"] = list(st.mean_ci(np.array(acc["fa"]))[1:])
        # steady state on a good channel: the downlink floor set by probing
        ctrl = AASSv2(probe_period=period)
        rec = simulate_stream(ctrl, np.full(800, 0.10), CFG,
                              _seed(f"probe-floor-{period}", k))
        sl = slice(100, None)
        entry["good_channel"] = {
            "downlink": float(rec["downlink"][sl].mean()),
            "false_ack": float(rec["false_ack"][sl].mean()),
            "probe_share": float(rec["probe"][sl].mean()),
        }

        # stealth adversary: the uplink stays good, so the controller settles in
        # optimistic mode, while the acknowledgements are suppressed.  This is
        # the state probing is meant to break, and the question it answers is
        # whether a silent loss is converted into a detected abort.
        acc = {"silent": [], "detected": [], "deliv": [], "dl": [], "opt": []}
        for rep in range(n_rep):
            rec = simulate_stream(AASSv2(probe_period=period),
                                  np.full(600, 0.15), CFG,
                                  _seed(f"probe-stealth-{period}", rep),
                                  jam_profile=[{"ack": 0.9}] * 600)
            sl = slice(100, None)
            fa = rec["false_ack"][sl].mean(); dv = rec["delivered"][sl].mean()
            acc["silent"].append(fa); acc["deliv"].append(dv)
            acc["detected"].append(max(0.0, 1.0 - fa - dv))
            acc["dl"].append(rec["downlink"][sl].mean())
            acc["opt"].append(rec["ctrl_optimistic"][sl].mean())
        entry["stealth"] = {kk: st.mean_ci(np.array(v))[0] for kk, v in acc.items()}
        entry["stealth"]["silent_ci"] = list(st.mean_ci(np.array(acc["silent"]))[1:])
        entry["stealth"]["detected_ci"] = list(st.mean_ci(np.array(acc["detected"]))[1:])
        rows.append(entry)
        print(f"   N_probe={period if period else 'none':>4}  "
              f"natural: DL={entry['natural']['dl']:.2f} FA={entry['natural']['fa']:.3f}  |  "
              f"jammed: DL={entry['jammed']['dl']:.2f} FA={entry['jammed']['fa']:.3f} "
              f"err={entry['jammed']['err']:.3f}  |  good: DL={entry['good_channel']['downlink']:.2f}  |  "
              f"stealth: silent={entry['stealth']['silent']:.3f} detected={entry['stealth']['detected']:.3f} "
              f"deliv={entry['stealth']['deliv']:.3f}")
    return {"rows": rows, "n_rep": n_rep}


# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "validation": exp_validation,
    "comparison": exp_comparison,
    "gilbert_elliott": exp_gilbert_elliott,
    "alpha": exp_alpha,
    "energy_budget": exp_energy_budget,
    "aass": exp_aass,
    "adversary": exp_adversary,
    "probing": exp_probing,
}


def main() -> None:
    global RESULTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None, metavar="DIR",
                    help="write the JSON files to DIR instead of results/ "
                         "(used by `make verify` to leave the reference "
                         "results untouched)")
    ap.add_argument("--all", action="store_true",
                    help="run every experiment (the default when none is named)")
    ap.add_argument("names", nargs="*", metavar="EXPERIMENT",
                    help="one or more of: " + ", ".join(EXPERIMENTS))
    args = ap.parse_args()
    unknown = [n for n in args.names if n not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(EXPERIMENTS)}")
    if args.all and args.names:
        ap.error("--all runs every experiment and cannot be combined with an "
                 "experiment name; drop --all to run only "
                 f"{', '.join(args.names)}")
    if args.out is not None:
        RESULTS = args.out.resolve()
        RESULTS.mkdir(parents=True, exist_ok=True)
    names = list(EXPERIMENTS) if args.all or not args.names else args.names
    for name in names:
        print(f"[{name}]")
        t0 = time.time()
        _dump(name, EXPERIMENTS[name]())
        print(f"   {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
