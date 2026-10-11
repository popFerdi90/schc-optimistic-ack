"""
Closed-form evaluation of the analytical model of Section 4.2, Eq. (1)-(14).

This module is the reference implementation used to (re)generate Tables 3 and 4
of the paper and to confront the model with the Monte-Carlo simulator of
:mod:`schc_aoe`.  Every function maps one-to-one onto a numbered equation of
the manuscript, so that Section 5.2 can quote an analytical value and its
simulated counterpart without ambiguity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Parameters of the analytical model (notation of Table 2)."""
    F: int = 25              # regular fragments in the SCHC Packet
    w: int = 7               # window size (all but the last window)
    w_last: int = 4          # fragments in the last window
    W: int = 4               # number of windows
    kmax: int = 3            # All-1 attempts, optimistic AoE
    M: int = 5               # ACK_REQ cycles, pessimistic AoE
    alpha: float = 0.5       # p_{u,All-1} = p_d = alpha * p_u

    # Timing (seconds) -- filled from the PHY model when delays are evaluated
    T_f: float = 0.0
    T_f_all1: float = 0.0
    T_a: float = 0.0
    T_req: float = 0.0
    T_r_all1: float = 0.0
    T_r_req: float = 0.0
    T_w: float = 0.0         # blocking wait that ends without any downlink


# ---------------------------------------------------------------------------
# Eq. (1) -- derived loss probabilities
# ---------------------------------------------------------------------------

def q_window(p_u: float, w: int) -> float:
    """Probability of at least one fragment loss among ``w`` fragments."""
    return 1.0 - (1.0 - p_u) ** w


def derived_probabilities(p_u: float, prm: ModelParams):
    """Eq. (1): ``q``, ``q_last`` and ``q_F``."""
    return (q_window(p_u, prm.w),
            q_window(p_u, prm.w_last),
            q_window(p_u, prm.F))


# ---------------------------------------------------------------------------
# Eq. (2)-(4) -- expected number of ACKs, pessimistic AoE
# ---------------------------------------------------------------------------

def p_succ_ack_req(p_u_ackreq: float, p_d: float) -> float:
    """Eq. (3): success probability of one (ACK_REQ, ACK) cycle."""
    return (1.0 - p_u_ackreq) * (1.0 - p_d)


def e_cycles_ack_req(p_succ: float, M: int) -> float:
    """Eq. (2): truncated-geometric number of (ACK_REQ, ACK) cycles."""
    fail = 1.0 - p_succ
    total = sum(i * fail ** (i - 1) * p_succ for i in range(1, M + 1))
    return total + M * fail ** M


def e_ack_pessimistic(p_u: float, p_d: float, prm: ModelParams,
                      p_u_ackreq: float | None = None) -> float:
    """Eq. (4): expected number of ACKs, pessimistic AoE."""
    q, q_last, _ = derived_probabilities(p_u, prm)
    if p_u_ackreq is None:
        p_u_ackreq = prm.alpha * p_u
    cyc = e_cycles_ack_req(p_succ_ack_req(p_u_ackreq, p_d), prm.M)
    return ((prm.W - 1) * q / (1.0 - p_u)
            + 1.0
            + cyc * (q_last / (1.0 - p_u) + 1.0))


def e_ack_pessimistic_simplified(p_u: float, prm: ModelParams) -> float:
    """Earlier, ACK_REQ-free form of the pessimistic ACK count.

    Retained because it is the expression that produced the values published in
    Table 4 of the submitted manuscript; it is reported in Section 5.2 to
    document the correction.
    """
    q, q_last, _ = derived_probabilities(p_u, prm)
    return ((prm.W - 1) * q / (1.0 - p_u)
            + 1.0
            + q_last * (1.0 / (1.0 - p_u) - 1.0))


# ---------------------------------------------------------------------------
# Eq. (5)-(7) -- expected number of ACKs, optimistic AoE
# ---------------------------------------------------------------------------

def p_succ_all1(p_u_all1: float, p_d: float) -> float:
    """Eq. (5): success probability of one (All-1, ACK) cycle."""
    return (1.0 - p_u_all1) * (1.0 - p_d)


def e_cycles_all1(p_succ: float, kmax: int) -> float:
    """Eq. (6): expected number of (All-1, ACK) correction cycles.

    As published, the failure branch is truncated without the usual
    ``kmax * (1 - p_succ) ** kmax`` residual term; see
    :func:`e_cycles_all1_complete`.
    """
    fail = 1.0 - p_succ
    return sum(k * fail ** (k - 1) * p_succ for k in range(1, kmax + 1))


def e_cycles_all1_complete(p_succ: float, kmax: int) -> float:
    """Proper truncated-geometric expectation, including the failure branch."""
    fail = 1.0 - p_succ
    return e_cycles_all1(p_succ, kmax) + kmax * fail ** kmax


def e_ack_optimistic(p_u: float, p_d: float, prm: ModelParams,
                     p_u_all1: float | None = None) -> float:
    """Eq. (7): expected number of ACKs, optimistic AoE."""
    _, _, q_F = derived_probabilities(p_u, prm)
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    return q_F * e_cycles_all1(p_succ_all1(p_u_all1, p_d), prm.kmax)


def e_ack_optimistic_refined(p_u: float, p_d: float, prm: ModelParams,
                             p_u_all1: float | None = None) -> float:
    """Refined count of ACK frames actually emitted in optimistic AoE.

    The receiver emits an ACK every time an All-1 reaches it while its bitmap is
    still incomplete.  Over at most ``kmax`` attempts this yields

        q_F * (1 - p_All1) * (1 - (1 - P_succ) ** kmax) / P_succ

    which is the quantity counted by the simulator (Section 5.2, Eq. (15)).
    """
    _, _, q_F = derived_probabilities(p_u, prm)
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    ps = p_succ_all1(p_u_all1, p_d)
    if ps <= 0.0:
        return 0.0
    return q_F * (1.0 - p_u_all1) * (1.0 - (1.0 - ps) ** prm.kmax) / ps


# ---------------------------------------------------------------------------
# Eq. (8)-(12) -- transmission delay
# ---------------------------------------------------------------------------

def t_window_error(p_u: float, prm: ModelParams) -> float:
    """Eq. (8): delay of a window in which at least one fragment is lost."""
    return (prm.w * prm.T_f
            + (1.0 / (1.0 - p_u) - 1.0) * (prm.w * prm.T_f + prm.T_a)
            + prm.T_a)


def t_cycle_req(p_u: float, prm: ModelParams) -> float:
    """Eq. (10): average duration of one ACK_REQ recovery cycle."""
    return (prm.T_req + prm.T_r_req + prm.w_last * p_u * prm.T_f
            + prm.T_f_all1 + prm.T_r_all1)


def t_last_window(p_u: float, p_d: float, prm: ModelParams,
                  p_u_ackreq: float | None = None) -> float:
    """Eq. (11): expected duration of the last window, pessimistic AoE."""
    if p_u_ackreq is None:
        p_u_ackreq = prm.alpha * p_u
    cyc = e_cycles_ack_req(p_succ_ack_req(p_u_ackreq, p_d), prm.M)
    return (prm.w_last * prm.T_f + prm.T_f_all1 + prm.T_r_all1
            + cyc * t_cycle_req(p_u, prm))


def e_delay_pessimistic(p_u: float, p_d: float, prm: ModelParams,
                        p_u_ackreq: float | None = None) -> float:
    """Eq. (9): expected end-to-end delay, pessimistic AoE."""
    q, _, _ = derived_probabilities(p_u, prm)
    t_err = t_window_error(p_u, prm)
    return ((prm.W - 1) * ((1.0 - q) * prm.w * prm.T_f + q * t_err)
            + t_last_window(p_u, p_d, prm, p_u_ackreq))


def t_cycle_all1(p_u: float, prm: ModelParams) -> float:
    """Eq. (13): duration of one (All-1, ACK) correction cycle."""
    return prm.T_r_all1 + prm.F * p_u * prm.T_f


def e_delay_optimistic(p_u: float, p_d: float, prm: ModelParams,
                       p_u_all1: float | None = None) -> float:
    """Eq. (12): expected end-to-end delay, optimistic AoE."""
    _, _, q_F = derived_probabilities(p_u, prm)
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    cyc = e_cycles_all1(p_succ_all1(p_u_all1, p_d), prm.kmax)
    return prm.F * prm.T_f + q_F * cyc * t_cycle_all1(p_u, prm)


# ---------------------------------------------------------------------------
# Eq. (14) -- probability of a false ACK
# ---------------------------------------------------------------------------

def p_false_ack_optimistic(p_u: float, p_d: float, prm: ModelParams,
                           p_u_all1: float | None = None) -> float:
    """Eq. (14): probability that the sender wrongly concludes success."""
    _, _, q_F = derived_probabilities(p_u, prm)
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    k = prm.kmax
    return q_F * ((1.0 - p_u_all1 ** k) * p_d ** k + p_u_all1 ** k)


def p_false_ack_pessimistic(*_args, **_kw) -> float:
    """Section 4.2.4: negligible in the pessimistic approach."""
    return 0.0


# ---------------------------------------------------------------------------
# Refinements introduced in the revision (Section 5.2, Eq. (15)-(19))
# ---------------------------------------------------------------------------

def e_ack_pessimistic_corrected(p_u: float, p_d: float, prm: ModelParams,
                                p_u_ackreq: float | None = None) -> float:
    """Eq. (18): Eq. (4) with the double-counted final ACK removed.

    As published, Eq. (4) adds the mandatory final ACK twice: once through the
    explicit ``+1`` term and once through the ``+1`` inside the ACK_REQ factor.
    On a lossless channel it therefore predicts two ACKs where exactly one is
    exchanged.  The corrected expression is exact in that limit.
    """
    q, q_last, _ = derived_probabilities(p_u, prm)
    if p_u_ackreq is None:
        p_u_ackreq = prm.alpha * p_u
    cyc = e_cycles_ack_req(p_succ_ack_req(p_u_ackreq, p_d), prm.M)
    return (prm.W - 1) * q / (1.0 - p_u) + q_last / (1.0 - p_u) + cyc


def p_deferred_repair(p_u: float, p_d: float) -> float:
    """Eq. (19): probability that a regular-window fragment is still missing
    when the last window begins.

    A fragment is repaired as long as the error ACKs of its window reach the
    sender; the repair loop stops at the first lost ACK, which defers the
    missing fragment to the final Compound ACK phase.  Summing over the number
    of successful repair rounds gives a closed form::

        mu = p_d * p_u / (1 - (1 - p_d) * p_u)

    This term is what Eq. (4) neglects, and it accounts for the residual
    deviation reported in Section 5.2.
    """
    return p_d * p_u / (1.0 - (1.0 - p_d) * p_u)


def e_correction_rounds(p_u: float, prm: ModelParams, rho: float = 1.0,
                        j_max: int = 400) -> float:
    """Eq. (15): discounted expected number of correction rounds.

    After ``j`` Compound-ACK-driven repairs, a given fragment is still missing
    with probability ``p_u ** (j + 1)``, so the SCHC Packet is still incomplete
    with probability ``1 - (1 - p_u ** (j + 1)) ** F``.  Each round is reached
    only if every previous round actually produced an ACK, which happens with
    probability ``rho`` per round; hence the discount factor ``rho ** j``::

        S(rho) = sum_j  rho**j * [1 - (1 - p_u**(j+1))**F]

    With ``rho = 1`` this is the plain expected number of correction rounds.
    """
    if p_u <= 0.0:
        return 0.0
    total = 0.0
    for j in range(j_max):
        term = rho ** j * (1.0 - (1.0 - p_u ** (j + 1)) ** prm.F)
        total += term
        if term < 1e-12 and j > 4:
            break
    return total


def _rho(p_succ: float, kmax: int) -> float:
    """Probability that a correction round yields an ACK at all."""
    return 1.0 - (1.0 - p_succ) ** kmax


def e_ack_optimistic_multiround(p_u: float, p_d: float, prm: ModelParams,
                                p_u_all1: float | None = None) -> float:
    """Eq. (16): expected number of ACKs in optimistic AoE, all rounds included.

    Eq. (7) counts a single correction round.  Each round that is reached emits

        (1 - p_All1) * (1 - (1 - P_succ)**KMAX) / P_succ

    ACK frames in expectation, and rounds are reached according to Eq. (15).
    """
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    ps = p_succ_all1(p_u_all1, p_d)
    if ps <= 0.0:
        return 0.0
    rho = _rho(ps, prm.kmax)
    per_round = (1.0 - p_u_all1) * rho / ps
    return e_correction_rounds(p_u, prm, rho) * per_round


def p_false_ack_multiround(p_u: float, p_d: float, prm: ModelParams,
                           p_u_all1: float | None = None) -> float:
    """Eq. (17): probability of a false ACK over the whole transfer.

    A false ACK is declared at the first correction round that produces no ACK
    at all, which happens with probability ``1 - rho`` per reached round::

        P_falseACK = (1 - rho) * S(rho)

    Eq. (14) is the restriction of this expression to a single round and
    therefore underestimates the risk by a factor that grows with ``p_u``.
    """
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    ps = p_succ_all1(p_u_all1, p_d)
    rho = _rho(ps, prm.kmax)
    return min(1.0, (1.0 - rho) * e_correction_rounds(p_u, prm, rho))


def e_ack_pessimistic_renewal(p_u: float, p_d: float, prm: ModelParams) -> float:
    """Eq. (18): pessimistic ACK count with both published defects removed.

    Two corrections are applied to Eq. (4):

    1. the mandatory final ACK is counted once instead of twice;
    2. the regular-window term ``(W-1) * q / (1 - p_u)`` -- an approximation of
       the number of repair rounds -- is replaced by the exact renewal term
       ``(W-1) * R(p_u, w, 1 - p_d)`` of Eq. (15), in which the discount factor
       ``1 - p_d`` accounts for repair rounds that never happen because the
       error ACK itself was lost.
    """
    from dataclasses import asdict as _asdict
    _, q_last, _ = derived_probabilities(p_u, prm)
    cyc = e_cycles_ack_req(p_succ_ack_req(prm.alpha * p_u, p_d), prm.M)
    reg = e_correction_rounds(
        p_u, ModelParams(**{**_asdict(prm), "F": prm.w}), rho=1.0 - p_d)
    return (prm.W - 1) * reg + q_last / (1.0 - p_u) + cyc


def e_final_rounds(p_u: float, mu: float, prm: ModelParams,
                   j_max: int = 400) -> float:
    """Expected number of *additional* Compound-ACK rounds in the final phase.

    When the last window opens, a regular-window fragment is still missing with
    probability ``mu`` (Eq. (16)) and a last-window fragment with probability
    ``p_u``.  After ``j`` further repair rounds these become ``mu * p_u**j`` and
    ``p_u**(j+1)``, so the packet is still incomplete with probability

        1 - (1 - mu*p_u**j)**((W-1)*w) * (1 - p_u**(j+1))**w_last
    """
    n_reg = (prm.W - 1) * prm.w
    total = 0.0
    for j in range(j_max):
        term = 1.0 - ((1.0 - mu * p_u ** j) ** n_reg
                      * (1.0 - p_u ** (j + 1)) ** prm.w_last)
        total += term
        if term < 1e-13 and j > 4:
            break
    return total


def e_ack_pessimistic_full(p_u: float, p_d: float, prm: ModelParams) -> float:
    """Eq. (17): pessimistic ACK count derived from the renewal identity.

    Three defects of Eq. (4) are removed at once: the final ACK is counted
    once, the number of repair rounds per regular window is computed exactly
    rather than approximated by ``q / (1 - p_u)``, and the repairs deferred by
    a lost error ACK are charged to the final phase through Eq. (16).
    """
    from dataclasses import asdict as _asdict
    mu = p_deferred_repair(p_u, p_d)
    reg = e_correction_rounds(
        p_u, ModelParams(**{**_asdict(prm), "F": prm.w}), rho=1.0 - p_d)
    cyc = e_cycles_ack_req(p_succ_ack_req(prm.alpha * p_u, p_d), prm.M)
    return (prm.W - 1) * reg + (1.0 + e_final_rounds(p_u, mu, prm)) * cyc


# ---------------------------------------------------------------------------
# Corrected delay expressions (Section 5.2, Eq. (23) and Table 9)
# ---------------------------------------------------------------------------
#
# Eq. (9) and (12) charge a delay per *correction cycle*, which hides two
# facts that dominate the measured transfer time: the sender performs K_max
# All-1 transmissions in every transfer, error-free ones included, and it
# blocks on a receive window after each of them whether or not an ACK arrives.
# We therefore decompose the delay into message counts multiplied by unit
# costs, and derive each count from the renewal identity of Eq. (15):
#
#     E[T] = N_frag*T_f + N_All1*T_f,All-1 + N_req*T_req + n_a*T_a + n_w*T_w
#
# where T_a is a blocking wait that ends with an ACK and T_w one that ends
# empty.

def delay_counts_optimistic(p_u: float, p_d: float, prm: ModelParams,
                            p_u_all1: float | None = None) -> dict:
    """Message counts entering the delay of optimistic AoE.

    ``N_frag``  each fragment is re-sent once per round in which it is still
                missing *and* the sender has been told so, hence the factor
                ``1/(1 - p_u*rho_K)``;
    ``N_All1``  the All-1 loop runs once per correction round, plus a full
                ``K_max`` attempts in the round that terminates the transfer
                without feedback -- the term Eq. (12) omits entirely;
    ``n_a``     one ACK-terminated wait per round that produced feedback;
    ``n_w``     every other All-1 attempt ends on an empty receive window.
    """
    if p_u_all1 is None:
        p_u_all1 = prm.alpha * p_u
    ps = p_succ_all1(p_u_all1, p_d)
    rho = _rho(ps, prm.kmax)
    R = e_correction_rounds(p_u, prm, rho)
    p_fa = min(1.0, (1.0 - rho) * R)
    n_all1 = R * rho / ps + (1.0 - p_fa) * prm.kmax
    n_a = R * rho
    return {"N_frag": prm.F / (1.0 - p_u * rho), "N_all1": n_all1,
            "N_req": 0.0, "n_a": n_a, "n_w": n_all1 - n_a}


def delay_counts_pessimistic(p_u: float, p_d: float, prm: ModelParams,
                             p_u_ackreq: float | None = None) -> dict:
    """Message counts entering the delay of pessimistic AoE.

    The fragment count has three terms: the repairs performed inside a regular
    window while its ACKs keep arriving, the repairs deferred by a lost ACK and
    settled in the final phase (Eq. (20)), and the last window itself.
    """
    from dataclasses import asdict as _asdict
    if p_u_ackreq is None:
        p_u_ackreq = prm.alpha * p_u
    mu = p_deferred_repair(p_u, p_d)
    r_reg = e_correction_rounds(
        p_u, ModelParams(**{**_asdict(prm), "F": prm.w}), rho=1.0 - p_d)
    rounds = 1.0 + e_final_rounds(p_u, mu, prm)
    ps_all1 = (1.0 - p_u) * (1.0 - p_d)          # All-1 at the nominal DR
    ps_req = p_succ_ack_req(p_u_ackreq, p_d)
    a_req = (1.0 - (1.0 - ps_req) ** prm.M) / ps_req
    n_req = rounds * (1.0 - ps_all1) * a_req
    n_reg = (prm.W - 1) * prm.w
    return {"N_frag": (n_reg / (1.0 - p_u * (1.0 - p_d))
                       + n_reg * mu / (1.0 - p_u)
                       + prm.w_last / (1.0 - p_u)),
            "N_all1": rounds, "N_req": n_req,
            "n_a": (prm.W - 1) * r_reg * (1.0 - p_d) + rounds,
            "n_w": (prm.W - 1) * r_reg * p_d + n_req}


def _assemble_delay(c: dict, t_f: float, t_all1: float, prm: ModelParams) -> float:
    return (c["N_frag"] * t_f + c["N_all1"] * t_all1 + c["N_req"] * prm.T_req
            + c["n_a"] * prm.T_a + c["n_w"] * prm.T_w)


def e_delay_optimistic_corrected(p_u: float, p_d: float, prm: ModelParams,
                                 t_all1: float | None = None) -> float:
    """Eq. (23) applied to optimistic AoE."""
    return _assemble_delay(delay_counts_optimistic(p_u, p_d, prm),
                           prm.T_f, prm.T_f_all1 if t_all1 is None else t_all1,
                           prm)


def e_delay_pessimistic_corrected(p_u: float, p_d: float, prm: ModelParams,
                                  t_all1: float) -> float:
    """Eq. (23) applied to pessimistic AoE (All-1 at the nominal data rate)."""
    return _assemble_delay(delay_counts_pessimistic(p_u, p_d, prm),
                           prm.T_f, t_all1, prm)
