"""
Adaptive AoE Switching Scheme (AASS), Section 4.3, and its hardened variant.

Two controllers are implemented:

``AASSv1``
    The scheme as described in the submitted manuscript: the uplink loss rate is
    estimated as a plain moving average over a sliding window of ``SW``
    fragments and compared with a single fixed threshold ``p_tu``.

``AASSv2``
    The hardened controller introduced in the revision.  It differs on three
    points, each answering one weakness of ``AASSv1``:

    1. the estimator is an exponentially weighted moving average (EWMA), which
       bounds the memory of the estimate and reacts faster to a step change;
    2. the decision uses a two-sided hysteresis band ``+/- h`` around
       ``p_tu``, which suppresses mode oscillations;
    3. a pessimistic *probe* packet is forced every ``probe_period`` packets.
       This is required because, in optimistic mode, the sender receives
       feedback only when an error is both detected and acknowledged: without
       probing the estimator is blind, which is precisely the state an active
       adversary can maintain by jamming selectively.

The module also derives the cost-optimal threshold ``p_tu*`` from the
analytical model, instead of fixing it arbitrarily.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import analytic as an
from .schc_aoe import SchcConfig, Timing, simulate_packet


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

@dataclass
class AASSv1:
    p_tu: float = 0.3
    sw: int = 100                     # sliding window, in fragments
    mode: str = "pessimistic"         # AASS starts in pessimistic mode
    _lost: int = 0
    _seen: int = 0

    def estimate(self) -> float:
        return self._lost / self._seen if self._seen else 0.0

    def update(self, n_observed: int, n_lost: int) -> None:
        if n_observed == 0:
            return
        # plain moving average over the last `sw` observed fragments
        self._lost += n_lost
        self._seen += n_observed
        if self._seen > self.sw:
            scale = self.sw / self._seen
            self._lost *= scale
            self._seen = self.sw
        self.mode = "optimistic" if self.estimate() <= self.p_tu else "pessimistic"

    def next_mode(self, _packet_index: int) -> str:
        return self.mode


@dataclass
class AASSv2:
    p_tu: float = 0.3
    h: float = 0.10                   # hysteresis half-band (tuned, Table 11)
    beta: float = 0.80                # EWMA memory factor (tuned, Table 11)
    probe_period: int = 20            # forced pessimistic probe, in packets
    mode: str = "pessimistic"
    _phat: float | None = None

    def estimate(self) -> float:
        return 0.0 if self._phat is None else self._phat

    def update(self, n_observed: int, n_lost: int) -> None:
        if n_observed == 0:
            return
        sample = n_lost / n_observed
        self._phat = sample if self._phat is None else \
            self.beta * self._phat + (1.0 - self.beta) * sample
        if self.mode == "pessimistic":
            if self._phat < self.p_tu - self.h:
                self.mode = "optimistic"
        else:
            if self._phat > self.p_tu + self.h:
                self.mode = "pessimistic"

    def next_mode(self, packet_index: int) -> str:
        if self.probe_period and packet_index % self.probe_period == 0:
            return "pessimistic"      # probe: refresh the estimate
        return self.mode


# ---------------------------------------------------------------------------
# Stream simulation
# ---------------------------------------------------------------------------

def simulate_stream(controller, p_u_profile, cfg: SchcConfig, seed: int,
                    alpha: float = 0.5, channel: str = "iid",
                    burst_len: float = 3.0, jam_profile=None) -> dict:
    """Simulate a stream of packets under a (possibly time-varying) loss rate.

    ``p_u_profile`` is a sequence giving the uplink loss probability in force
    for each packet; ``jam_profile`` optionally gives a per-packet jamming
    configuration dict.
    """
    tim = Timing.from_config(cfg)
    rng = np.random.default_rng(seed)
    n = len(p_u_profile)

    rec = {k: np.zeros(n) for k in
           ("ack", "downlink", "energy", "delay", "false_ack", "delivered",
            "phat", "optimistic", "ctrl_optimistic", "probe")}

    for i, p_u in enumerate(p_u_profile):
        decided = controller.mode
        mode = controller.next_mode(i)
        rec["probe"][i] = 1.0 if mode != decided else 0.0
        jam = jam_profile[i] if jam_profile is not None else None
        r = simulate_packet(mode, float(p_u), cfg, tim, rng, alpha,
                            channel, burst_len, jam=jam)
        controller.update(r.n_frag_observed, r.n_lost_observed)

        rec["ack"][i] = r.n_ack_tx
        rec["downlink"][i] = r.n_downlink_tx
        rec["energy"][i] = r.energy_j
        rec["delay"][i] = r.delay_realistic
        rec["false_ack"][i] = float(r.false_ack)
        rec["delivered"][i] = float(r.delivered)
        rec["phat"][i] = controller.estimate()
        rec["optimistic"][i] = 1.0 if mode == "optimistic" else 0.0
        rec["ctrl_optimistic"][i] = 1.0 if controller.mode == "optimistic" else 0.0

    # mode oscillation is measured on the controller's own decision, so that
    # the deliberate pessimistic probes of AASS-H are not counted as switches
    rec["n_switches"] = float(np.abs(np.diff(rec["ctrl_optimistic"])).sum())
    rec["n_probes"] = float(rec["probe"].sum())
    return rec


# ---------------------------------------------------------------------------
# Cost-optimal threshold
# ---------------------------------------------------------------------------

def switching_cost(p_u: float, prm: an.ModelParams, lam: float) -> float:
    """Net benefit of running optimistic rather than pessimistic AoE.

    ``lam`` is the application-dependent price of one false ACK, expressed in
    units of "one downlink ACK".  The optimistic mode is preferable while

        Delta_ACK(p_u) - lam * P_falseACK(p_u) > 0
    """
    p_d = prm.alpha * p_u
    d_ack = (an.e_ack_pessimistic_full(p_u, p_d, prm)
             - an.e_ack_optimistic_multiround(p_u, p_d, prm))
    return d_ack - lam * an.p_false_ack_multiround(p_u, p_d, prm)


def optimal_threshold(prm: an.ModelParams, lam: float,
                      grid: np.ndarray | None = None) -> float:
    """Largest ``p_u`` for which optimistic AoE still pays off, given ``lam``.

    This replaces the arbitrary ``p_tu = 0.3`` of the submitted manuscript by a
    threshold derived from Eq. (4), (7) and (14).
    """
    grid = np.linspace(0.01, 0.95, 379) if grid is None else grid
    gains = np.array([switching_cost(p, prm, lam) for p in grid])
    positive = grid[gains > 0]
    if positive.size == 0:
        return 0.0
    # the decision region is an interval starting at 0; take its upper edge
    first_neg = np.argmax(gains <= 0) if np.any(gains <= 0) else grid.size
    return float(grid[max(first_neg - 1, 0)])
