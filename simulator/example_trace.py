"""Minimal trace-driven example.

Runs the two SCHC ACK-on-Error variants against a *measured* reception trace
instead of a parametric loss model, and prints the downlink cost, the delivery
ratio and the false-ACK probability under that trace.

Usage::

    python -m simulator.example_trace                       # bundled example trace
    python -m simulator.example_trace path/to/your_trace.txt

The trace file holds one 0/1 flag per transmitted frame, 1 meaning the frame
was not received, either one value per line or comma/space separated; ``#``
introduces a comment.  The only component that changes between a parametric
run and a trace-driven one is the channel object (``simulator/channel.py``);
the fragmentation engine, the statistics and the figure scripts are untouched.

Note that the empirical process supersedes the parametric rates: ``p_u``,
``p_d`` and the asymmetry factor alpha play no role here, every frame class
replaying the same trace from an independent random offset.  A capture that
distinguishes uplink from downlink losses would be handled by passing two
distinct traces to ``_channels``, which is a three-line change.

The bundled ``traces/example_trace.txt`` is a synthetic placeholder, clearly
marked as such in its header, so that this script runs out of the box; it is
not field data and no result of the paper is derived from it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from .channel import load_trace
from .schc_aoe import SchcConfig, Timing, simulate_batch
from . import stats as st

DEFAULT_TRACE = Path(__file__).resolve().parent / "traces" / "example_trace.txt"
N_RUNS = 5_000


def _seed(tag: str) -> int:
    digest = hashlib.blake2b(tag.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 31)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = Path(argv[0]) if argv else DEFAULT_TRACE
    if not path.exists():
        print(f"trace file not found: {path}", file=sys.stderr)
        return 1

    trace = load_trace(str(path))
    print(f"trace          : {path}")
    print(f"samples        : {trace.size}")
    print(f"mean loss rate : {trace.mean():.4f}")
    print(f"replications   : {N_RUNS} per scheme\n")

    cfg = SchcConfig(max_rounds=60)
    tim = Timing.from_config(cfg)
    chan = f"trace:{path}"

    hdr = f"{'scheme':<26}{'downlink':>20}{'delivered':>18}{'false ACK':>18}"
    print(hdr)
    print("-" * len(hdr))

    for mode, label in (("pessimistic", "Pessimistic AoE"),
                        ("optimistic", "Optimistic AoE")):
        out = simulate_batch(mode, 0.0, N_RUNS, cfg, _seed(f"trace/{mode}"),
                             channel=chan, tim=tim)
        dl = st.mean_ci(out["n_downlink_tx"])
        de = st.proportion_ci(out["delivered"].sum(), N_RUNS)
        fa = st.proportion_ci(out["false_ack"].sum(), N_RUNS)
        print(f"{label:<26}"
              f"{dl[0]:>9.3f} [{dl[1]:.3f},{dl[2]:.3f}]"
              f"{de[0]:>8.4f} [{de[1]:.4f},{de[2]:.4f}]"
              f"{fa[0]:>8.4f} [{fa[1]:.4f},{fa[2]:.4f}]")

    print("\nValues in brackets are 95% confidence intervals "
          "(Student for means, Wilson for proportions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
