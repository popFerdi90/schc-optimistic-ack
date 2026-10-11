"""Audit: every numeric cell of every data table, against its source."""
import json, re, pathlib, sys
from dataclasses import asdict
from simulator import analytic as an
from simulator import lora_phy as phy
from simulator.schc_aoe import SchcConfig, Timing

ROOT = pathlib.Path(__file__).resolve().parent
tex = (ROOT / "sn-article.tex").read_text()
cfg = SchcConfig(); tim = Timing.from_config(cfg)
t_a = phy.RX2_DELAY + tim.toa_ack
PRM = an.ModelParams(F=25,w=7,w_last=4,W=4,kmax=3,M=5,alpha=0.5,
    T_f=tim.toa_frag, T_f_all1=tim.toa_all1_opt, T_a=t_a,
    T_req=tim.toa_ackreq, T_r_all1=t_a, T_r_req=t_a,
    T_w=phy.RX2_DELAY+tim.listen_rx2)
J = {n: json.loads((ROOT / "results" / f"{n}.json").read_text())
     for n in ("validation","comparison","energy_budget","aass","probing")}

def table(anchor):
    """Extract a table body, whether it is a table or a full-width table* float."""
    i = tex.index(anchor)
    b = max(tex.rindex("\\begin{table}", 0, i), tex.rindex("\\begin{table*}", 0, i))
    e_plain = tex.find("\\end{table}", i)
    e_star = tex.find("\\end{table*}", i)
    cands = [x for x in (e_plain, e_star) if x != -1]
    if not cands:
        raise ValueError(f"no table end after anchor {anchor!r}")
    e = min(cands)
    blk = tex[b:e]
    return blk[blk.index("\\midrule"):blk.index("\\bottomrule")]

def rows(anchor):
    out=[]
    for line in table(anchor).split("\\\\"):
        line=line.strip()
        line=line.replace("\\midrule","").strip()
        if not line or "multicolumn" in line: continue
        cells=[c.strip() for c in line.split("&")]
        if len(cells)>1: out.append(cells)
    return out

def nums(cells):
    out=[]
    for c in cells:
        m=re.findall(r"[-+]?\d+\.?\d*", c.replace("\\textbf{","").replace("$",""))
        out.append(float(m[0]) if m else None)
    return out

FAIL=[]
def cmp(tag, claimed, actual, tol, dp=None):
    """A cell is conform when it equals the source rounded to its own precision."""
    if claimed is None:
        FAIL.append((tag, None, round(actual, 4))); return False
    ok = abs(claimed - actual) <= tol
    if not ok and dp is not None:
        ok = abs(claimed - round(actual, dp)) < 10.0 ** (-dp - 3)
    if not ok:
        ok = abs(claimed - round(actual, len(str(claimed).split(".")[-1])
                                 if "." in str(claimed) else 0)) < 1e-9
    if not ok: FAIL.append((tag, claimed, round(actual, 4)))
    return ok

def report(name, n_ok, n_tot):
    mark = "ok  " if n_ok==n_tot else "FAIL"
    print(f"  [{mark}] {name:44s} {n_ok}/{n_tot} cells")

# ---- Table 4 (tab3): impact of KMAX, p_u = 0.5 ----------------------------
n=k=0
for cells in rows("Impact of $KMAX$"):
    v=nums(cells); kmax=int(v[0]); p=0.5; pd=0.25
    prm=an.ModelParams(**{**asdict(PRM),"kmax":kmax})
    exp=[100*an.p_false_ack_multiround(p,pd,prm),
         an.e_ack_optimistic_multiround(p,pd,prm),
         an.e_delay_optimistic_corrected(p,pd,prm)]
    for j,x in enumerate(exp):
        n+=1; k+=cmp(f"T4 K={kmax} col{j+1}", v[j+1], x, 0.006*max(1,abs(x)))
report("Table 4  impact of Kmax (analytical)", k, n)

# ---- Table 5 (tab:toa): PHY ----------------------------------------------
n=k=0
sfs=[7,9,12]
lbl_fn={"Regular fragment": lambda sf: 1e3*phy.toa_uplink(cfg.b_frag,sf),
        "All-1 fragment":   lambda sf: 1e3*phy.toa_uplink(cfg.b_all1,sf),
        "ACK (downlink)":   lambda sf: 1e3*phy.toa_downlink(cfg.b_ack,sf),
        "Off-time":         lambda sf: phy.time_off(phy.toa_uplink(cfg.b_all1,sf)),
        "Transmit energy":  lambda sf: 1e3*phy.DEFAULT_ENERGY.e_tx(phy.toa_uplink(cfg.b_all1,sf))}
for cells in rows("Time-on-Air, regulatory off-time"):
    key=next((kk for kk in lbl_fn if cells[0].startswith(kk)), None)
    if key is None: continue
    v=nums(cells[1:])
    for j,sf in enumerate(sfs):
        x=lbl_fn[key](sf); n+=1; k+=cmp(f"T5 {key} SF{sf}", v[j], x, 0.05+0.002*abs(x))
report("Table 5  Time-on-Air / duty cycle / energy (PHY)", k, n)

# ---- Table 6 (tab:designspace): simulation -------------------------------
V={(r["kmax"],r["sf_all1"]):r for r in J["energy_budget"]["variants"] if r["p_u"]==0.3}
n=k=0
for cells in rows("Cost of the two ways of hardening"):
    sf=int(cells[0].replace("SF","")); kmax=int(nums([cells[1]])[0]); r=V[(kmax,sf)]
    v=nums(cells[2:])
    exp=[100*(r["energy_ratio"]-1), 100*(r["airtime_ratio"]-1),
         -100*(1-r["downlink_ratio"]), r["optimistic"]["false_ack"]["mean"],
         r["optimistic"]["delivered"]["mean"]]
    tols=[0.06,0.06,0.06,5e-5,5e-4]
    for j,(x,t) in enumerate(zip(exp,tols)):
        n+=1; k+=cmp(f"T6 SF{sf} K={kmax} col{j+1}", v[j], x, t)
report("Table 6  All-1 design space (simulation)", k, n)

R={round(r["p_u"],2):r for r in J["validation"]["rows"]}

# ---- Table 9 (tab:validation) -------------------------------------------
P13=[0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]
spec13_full=[("sim_pess",lambda r:r["sim_pess"]["n_ack_tx"]["mean"],6e-4),
        ("mod_pess",lambda r:r["eq17_ack_pess_full"],6e-4),
        ("dev_pess",lambda r:100*(r["eq17_ack_pess_full"]-r["sim_pess"]["n_ack_tx"]["mean"])/r["sim_pess"]["n_ack_tx"]["mean"],0.06),
        ("sim_opt",lambda r:r["sim_opt"]["n_ack_tx"]["mean"],6e-4),
        ("mod_opt",lambda r:r["eq16_ack_opt"],6e-4),
        ("dev_opt",lambda r:100*(r["eq16_ack_opt"]-r["sim_opt"]["n_ack_tx"]["mean"])/r["sim_opt"]["n_ack_tx"]["mean"],0.06),
        ("sim_fa",lambda r:r["sim_opt"]["false_ack"]["mean"],6e-5),
        ("mod_fa",lambda r:r["eq17_false_ack"],6e-5),
        ("dev_fa",lambda r:100*(r["eq17_false_ack"]-r["sim_opt"]["false_ack"]["mean"]),6e-3),
        ("sim_dp",lambda r:r["sim_pess"]["delay_model"]["mean"],6e-3),
        ("mod_dp",lambda r:r["eq23_delay_pess"],6e-3),
        ("dev_dp",lambda r:100*(r["eq23_delay_pess"]-r["sim_pess"]["delay_model"]["mean"])/r["sim_pess"]["delay_model"]["mean"],0.06),
        ("sim_do",lambda r:r["sim_opt"]["delay_model"]["mean"],6e-3),
        ("mod_do",lambda r:r["eq23_delay_opt"],6e-3),
        ("dev_do",lambda r:100*(r["eq23_delay_opt"]-r["sim_opt"]["delay_model"]["mean"])/r["sim_opt"]["delay_model"]["mean"],0.06)]
DROP13={"sim_pess","mod_pess","sim_dp","mod_dp","sim_do","mod_do"}
spec13=[x for x in spec13_full if x[0] not in DROP13]
n=k=0
rr13=rows("Analytical model confronted with simulation")
assert len(rr13)==len(spec13), f"{len(rr13)} lignes lues, {len(spec13)} attendues"
for cells,(tag,fn,tol) in zip(rr13,spec13):
    v=nums(cells[1:])
    for j,p_ in enumerate(P13):
        n+=1; k+=cmp(f"T9 {tag} p={p_}", v[j], fn(R[round(p_,2)]), tol)
report("Table 9  model validation (simulation)", k, n)

# ---- Table 10 (tab:comparative) ------------------------------------------
P=[0.10,0.20,0.30,0.40,0.50,0.60]
C={round(r["p_u"],2):r for r in J["comparison"]["rows"]}
order=[("Pessimistic","pessimistic"),("Optimistic (nominal)","optimistic"),
       ("Optimistic (revised)","optimistic_rev"),("AASS-H","aass")]
blocks=[("n_downlink_tx",5e-3,lambda d:d["mean"]),("delivered",5e-4,lambda d:d["mean"]),
        ("false_ack",5e-5,lambda d:d["mean"])]
n=k=0
rr=rows("Comparative performance ($10\\,000$ runs per point)")
idx=0
for metric,tol,get in blocks:
    lbls=[o for o in order if not (metric=="false_ack" and o[1]=="pessimistic")]
    for lbl,key in lbls:
        v=nums(rr[idx][1:]); idx+=1
        for j,p in enumerate(P):
            n+=1; k+=cmp(f"T10 {metric} {lbl} p={p}", v[j], get(C[p][key][metric]), tol)
for lbl,fld in (("Optimistic (nominal)","energy_overhead_pct"),
                ("Optimistic (revised)","energy_overhead_rev_pct")):
    v=nums(rr[idx][1:]); idx+=1
    for j,p in enumerate(P):
        n+=1; k+=cmp(f"T10 energy {lbl} p={p}", v[j], C[p][fld], 0.06)
report("Table 10 comparative performance (simulation)", k, n)

# ---- Table 11 (tab:aasstuning) -------------------------------------------
T={(r["beta"],r["h"]):r for r in J["aass"]["tuning"]}
n=k=0
for cells in rows("Stability/reactivity trade-off"):
    v=nums(cells); beta,h=v[0],v[1]; r=T[(beta,h)]
    exp=[r["n_switches"], r["adaptation_delay_pkts"],
         r["false_ack_after_step"], r["downlink_total"]]
    tols=[0.06,0.06,5e-4,0.6]
    for j,(x,t) in enumerate(zip(exp,tols)):
        n+=1; k+=cmp(f"T11 b={beta} h={h} col{j+1}", v[j+2], x, t)
report("Table 11 AASS-H tuning (simulation)", k, n)

# ---- Table 12 (tab:probing) ----------------------------------------------
PB={ (r["probe_period"] or 0): r for r in J["probing"]["rows"] }
base=PB[0]["good_channel"]["downlink"]
n=k=0
for cells in rows("Effect of the probing period"):
    lab=cells[0].strip()
    period=0 if lab.lower().startswith("none") else int(nums([lab])[0])
    r=PB[period]
    # this table packs "value [lo, hi]" in single cells: read every number
    v=[float(x) for x in re.findall(r"[-+]?\d+\.\d+",
         " ".join(cells[1:]).replace("\\textbf{","").replace("$",""))]
    exp=[r["good_channel"]["downlink"],
         None if period==0 else 100*(r["good_channel"]["downlink"]/base-1),
         r["natural"]["fa"], r["natural"]["fa_ci"][0], r["natural"]["fa_ci"][1],
         r["stealth"]["silent"], r["stealth"]["silent_ci"][0], r["stealth"]["silent_ci"][1],
         r["stealth"]["detected"], r["stealth"]["detected_ci"][0], r["stealth"]["detected_ci"][1]]
    exp=[e for e in exp if e is not None]
    for j,x in enumerate(exp):
        n+=1; k+=cmp(f"T12 N={period} col{j+1}", v[j] if j<len(v) else None, x, 6e-4 if x<2 else 0.06)
report("Table 12 probing period (simulation)", k, n)

print()
print("ALL CELLS MATCH THEIR SOURCE" if not FAIL else f"{len(FAIL)} CELL(S) DO NOT MATCH THEIR SOURCE:")
for f in FAIL[:40]: print("   ", f)
sys.exit(1 if FAIL else 0)
