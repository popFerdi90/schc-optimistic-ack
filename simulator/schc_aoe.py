"""
Monte-Carlo simulator of SCHC Fragmentation/Reassembly in ACK-on-Error mode.

Three senders are implemented:

* ``pessimistic`` -- standard RFC 8724 AoE: a final ACK is mandatory after the
  All-1 fragment, and the sender may issue up to ``M`` ACK_REQ messages to
  force it.
* ``optimistic``  -- the approach proposed in Section 4.1: the receiver never
  emits a positive ACK, the All-1 fragment is repeated up to ``KMAX`` times and
  the absence of any ACK is interpreted as an implicit acknowledgement.
* ``aass``        -- the adaptive scheme of Section 4.3, simulated at the level
  of a stream of packets by :func:`simulate_stream`.

The receiver is modelled faithfully with respect to RFC 8724 on one point that
the analytical model of Section 4.2 leaves implicit: reassembly can only be
completed once an All-1 fragment has been received, because the All-1 fragment
carries the RCS.  If every All-1 attempt is lost, the receiver's inactivity
timer expires and it emits a SCHC Receiver-Abort, which is counted as downlink
traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

from . import lora_phy as phy
from .channel import make_channel, JammedChannel

Mode = Literal["pessimistic", "optimistic"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchcConfig:
    """Protocol and PHY configuration (Tables 5-7 of the paper)."""
    F: int = 25                 # regular fragments
    w: int = 7                  # regular window size
    w_last: int = 4             # last-window size
    W: int = 4                  # number of windows
    kmax: int = 3               # All-1 attempts (optimistic)
    M: int = 5                  # ACK_REQ cycles (pessimistic)
    max_rounds: int = 20        # safety bound on correction rounds

    # Spreading factors
    sf_frag: int = 7
    sf_all1_opt: int = 12       # optimistic AoE: All-1 hardened at SF12
    sf_all1_pess: int = 7       # pessimistic AoE: ADR-managed, same DR
    sf_ack: int = 12            # ACKs served on RX2 (SF12, 869.525 MHz)
    sf_ackreq: int = 7

    # Application payload sizes (bytes of FRMPayload)
    b_frag: int = 51
    b_all1: int = 5             # SCHC header + RCS (CRC32)
    b_ack: int = 4
    b_ackreq: int = 3
    b_abort: int = 2

    duty_cycle: float = phy.DUTY_CYCLE_G1
    enforce_duty_cycle: bool = True

    def __post_init__(self):
        assert self.w * (self.W - 1) + self.w_last == self.F, \
            "window decomposition inconsistent with F"


@dataclass
class Timing:
    """Pre-computed Time-on-Air and receive-window durations."""
    toa_frag: float
    toa_all1_opt: float
    toa_all1_pess: float
    toa_ack: float
    toa_ackreq: float
    toa_abort: float
    listen_rx1_frag: float
    listen_rx2: float

    @classmethod
    def from_config(cls, cfg: SchcConfig) -> "Timing":
        return cls(
            toa_frag=phy.toa_uplink(cfg.b_frag, cfg.sf_frag),
            toa_all1_opt=phy.toa_uplink(cfg.b_all1, cfg.sf_all1_opt),
            toa_all1_pess=phy.toa_uplink(cfg.b_all1, cfg.sf_all1_pess),
            toa_ack=phy.toa_downlink(cfg.b_ack, cfg.sf_ack),
            toa_ackreq=phy.toa_uplink(cfg.b_ackreq, cfg.sf_ackreq),
            toa_abort=phy.toa_downlink(cfg.b_abort, cfg.sf_ack),
            listen_rx1_frag=phy.RX_SYMB_TIMEOUT * phy.symbol_time(cfg.sf_frag),
            listen_rx2=phy.RX_SYMB_TIMEOUT * phy.symbol_time(cfg.sf_ack),
        )


@dataclass
class RunResult:
    """Per-packet outcome."""
    n_ack_tx: int = 0            # ACK frames emitted by the receiver
    n_abort_tx: int = 0          # SCHC Receiver-Abort frames
    n_frag_tx: int = 0           # regular fragment transmissions
    n_all1_tx: int = 0           # All-1 transmissions
    n_ackreq_tx: int = 0         # ACK_REQ transmissions
    delay_model: float = 0.0     # delay restricted to the scope of Eq. (9)/(12)
    delay_protocol: float = 0.0  # delay incl. mandatory Class A receive windows
    delay_realistic: float = 0.0 # delay including EU868 duty-cycle back-off
    energy_j: float = 0.0        # node radio energy
    toa_up_s: float = 0.0        # cumulated uplink airtime
    toa_down_s: float = 0.0      # cumulated downlink airtime
    delivered: bool = False      # receiver completed reassembly
    declared_success: bool = False
    false_ack: bool = False
    aborted_by_sender: bool = False
    n_lost_observed: int = 0     # fragment losses reported back to the sender
    n_frag_observed: int = 0     # fragments covered by received feedback

    @property
    def n_downlink_tx(self) -> int:
        return self.n_ack_tx + self.n_abort_tx


# ---------------------------------------------------------------------------
# Sender/receiver state machine
# ---------------------------------------------------------------------------

class _PacketSim:
    """One SCHC Packet transfer over a given set of channels.

    Two distinct state vectors are maintained, because the whole question
    addressed by the paper is what the *sender* may legitimately infer:

    ``received``      ground truth at the receiver;
    ``known_missing`` what the sender has actually been told, through a bitmap.

    A sender only ever retransmits fragments listed in ``known_missing``; an
    ACK that is lost therefore leaves the two views desynchronised, which is
    the mechanism behind both the deferred repairs of pessimistic AoE and the
    false ACKs of optimistic AoE.
    """

    def __init__(self, cfg: SchcConfig, tim: Timing, energy: phy.EnergyModel,
                 ch_up, ch_all1, ch_down, ch_ackreq):
        self.cfg, self.tim, self.em = cfg, tim, energy
        self.ch_up, self.ch_all1 = ch_up, ch_all1
        self.ch_down, self.ch_ackreq = ch_down, ch_ackreq
        self.r = RunResult()
        self.received = np.zeros(cfg.F, dtype=bool)
        self.known_missing = np.zeros(cfg.F, dtype=bool)
        self.all1_received = False

    # -- physical events ---------------------------------------------------

    def _uplink(self, toa: float, expect_downlink: bool) -> None:
        """Account for one uplink and its mandatory Class A receive windows."""
        r, em, tim, cfg = self.r, self.em, self.tim, self.cfg
        r.toa_up_s += toa
        r.energy_j += em.e_tx(toa)
        r.delay_protocol += toa
        r.delay_model += toa
        if cfg.enforce_duty_cycle:
            r.delay_realistic += toa + phy.time_off(toa, cfg.duty_cycle)
        else:
            r.delay_realistic += toa

    def _rx_windows_idle(self, blocking: bool = False) -> None:
        self.r.energy_j += self.em.e_rx_window_idle(self.cfg.sf_frag,
                                                    self.cfg.sf_ack)
        self.r.delay_protocol += phy.RX2_DELAY + self.tim.listen_rx2
        if blocking:
            self.r.delay_model += phy.RX2_DELAY + self.tim.listen_rx2

    def _rx_windows_ack(self) -> None:
        self.r.energy_j += self.em.e_rx_window_ack(self.tim.toa_ack,
                                                   self.cfg.sf_frag)
        self.r.delay_protocol += phy.RX2_DELAY + self.tim.toa_ack
        self.r.delay_model += phy.RX2_DELAY + self.tim.toa_ack

    def _send_fragment(self, idx: int) -> None:
        self.r.n_frag_tx += 1
        self._uplink(self.tim.toa_frag, expect_downlink=False)
        if not self.ch_up.lost():
            self.received[idx] = True

    def _emit_ack(self) -> bool:
        """Receiver emits an ACK; returns True if the sender receives it."""
        self.r.n_ack_tx += 1
        self.r.toa_down_s += self.tim.toa_ack
        return not self.ch_down.lost()

    def _emit_abort(self) -> None:
        self.r.n_abort_tx += 1
        self.r.toa_down_s += self.tim.toa_abort

    # -- helpers -----------------------------------------------------------

    def _window_slice(self, win: int) -> slice:
        cfg = self.cfg
        start = win * cfg.w
        stop = start + (cfg.w if win < cfg.W - 1 else cfg.w_last)
        return slice(start, stop)

    def _observe(self, sl: slice) -> None:
        """The sender learns the reception status of a window from a bitmap."""
        self.r.n_frag_observed += sl.stop - sl.start
        n_lost = int((~self.received[sl]).sum())
        self.r.n_lost_observed += n_lost
        self.known_missing[sl] = ~self.received[sl]

    def _retransmit_known_missing(self, sl: slice | None = None) -> None:
        """Retransmit exactly the fragments the sender has been told are missing."""
        sl = sl or slice(0, self.cfg.F)
        for idx in range(sl.start, sl.stop):
            if self.known_missing[idx]:
                self.known_missing[idx] = False
                self._send_fragment(idx)
                self._rx_windows_idle()

    def _complete(self) -> bool:
        return bool(self.received.all()) and self.all1_received

    # -- pessimistic AoE ---------------------------------------------------

    def run_pessimistic(self) -> RunResult:
        """Standard RFC 8724 AoE with a mandatory final ACK.

        Regular windows are acknowledged only on error.  When such an ACK is
        lost the sender has no way of knowing it, so it simply proceeds to the
        next window; the outstanding fragments are then repaired during the
        final phase, where the receiver reports every incomplete window in a
        single Compound ACK (RFC 9441).  ACK_REQ messages are used only in the
        final phase, which is what Eq. (2)-(4) model.
        """
        cfg, tim, r = self.cfg, self.tim, self.r

        # --- phase 1: regular windows --------------------------------------
        for win in range(cfg.W - 1):
            sl = self._window_slice(win)
            for idx in range(sl.start, sl.stop):
                self._send_fragment(idx)
                self._rx_windows_idle()
            for _ in range(cfg.max_rounds):
                if self.received[sl].all():
                    break                      # silent window, move on
                if not self._emit_ack():        # error ACK lost: repair deferred
                    self._rx_windows_idle(blocking=True)
                    break
                self._rx_windows_ack()
                self._observe(sl)
                self._retransmit_known_missing(sl)

        # --- phase 2: last window, All-1 and mandatory final ACK -----------
        sl = self._window_slice(cfg.W - 1)
        for idx in range(sl.start, sl.stop):
            self._send_fragment(idx)
            self._rx_windows_idle()

        for _ in range(cfg.max_rounds):
            r.n_all1_tx += 1
            self._uplink(tim.toa_all1_pess, expect_downlink=True)
            if not self.ch_all1.lost():
                self.all1_received = True
                got = self._emit_ack()
                self._rx_windows_ack() if got else self._rx_windows_idle(blocking=True)
            else:
                got = False
                self._rx_windows_idle(blocking=True)

            if not got:
                got = self._force_ack()
                if not got:
                    r.aborted_by_sender = True
                    return self._finish()

            # the final (Compound) ACK carries the status of every window
            self._observe(slice(0, cfg.F))
            if self._complete():
                r.declared_success = True
                return self._finish()
            self._retransmit_known_missing()

        r.aborted_by_sender = True
        return self._finish()

    def _force_ack(self) -> bool:
        """Obtain the final ACK using up to ``M`` ACK_REQ cycles (Eq. (2))."""
        cfg, tim, r = self.cfg, self.tim, self.r
        for _ in range(cfg.M):
            r.n_ackreq_tx += 1
            self._uplink(tim.toa_ackreq, expect_downlink=True)
            if self.ch_ackreq.lost():
                self._rx_windows_idle(blocking=True)
                continue
            if self._emit_ack():
                self._rx_windows_ack()
                return True
            self._rx_windows_idle(blocking=True)
        return False

    # -- optimistic AoE ----------------------------------------------------

    def run_optimistic(self) -> RunResult:
        """Optimistic AoE (Section 4.1).

        Consistently with Eq. (7) and Eq. (12), the sender pushes the whole
        SCHC Packet without stopping at window boundaries; every repair is
        driven by the Compound ACK returned after an All-1 transmission.  The
        receiver never emits a positive ACK, so the sender concludes success
        from the absence of any feedback after ``KMAX`` All-1 attempts.
        """
        cfg, tim, r = self.cfg, self.tim, self.r

        for idx in range(cfg.F):
            self._send_fragment(idx)
            self._rx_windows_idle()

        for _ in range(cfg.max_rounds):
            got_ack = False
            for _k in range(cfg.kmax):
                r.n_all1_tx += 1
                self._uplink(tim.toa_all1_opt, expect_downlink=True)
                if self.ch_all1.lost():
                    self._rx_windows_idle(blocking=True)
                    continue
                self.all1_received = True
                if self.received.all():
                    # "no news is good news": the receiver stays silent
                    self._rx_windows_idle(blocking=True)
                    continue
                if self._emit_ack():
                    self._rx_windows_ack()
                    got_ack = True
                    break
                self._rx_windows_idle(blocking=True)

            if not got_ack:
                # KMAX attempts without feedback -> implicit acknowledgement
                r.declared_success = True
                if not self._complete():
                    # the receiver's inactivity timer fires; the Receiver-Abort
                    # still costs a downlink, but the sender no longer listens
                    self._emit_abort()
                return self._finish()

            self._observe(slice(0, cfg.F))
            self._retransmit_known_missing()

        r.aborted_by_sender = True
        return self._finish()

    # -- termination -------------------------------------------------------

    def _finish(self) -> RunResult:
        r = self.r
        r.delivered = self._complete()
        r.false_ack = r.declared_success and not r.delivered
        if self.cfg.enforce_duty_cycle:
            r.delay_realistic = max(r.delay_realistic, r.delay_protocol)
        else:
            r.delay_realistic = r.delay_protocol
        return r


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _channels(p_u: float, p_d: float, p_all1: float, rng, channel: str,
              burst_len: float, jam: dict | None):
    ch_up = make_channel(channel, p_u, rng, burst_len)
    ch_all1 = make_channel(channel, p_all1, rng, burst_len)
    ch_down = make_channel(channel, p_d, rng, burst_len)
    ch_ackreq = make_channel(channel, p_all1, rng, burst_len)
    if jam:
        if jam.get("all1", 0.0):
            ch_all1 = JammedChannel(ch_all1, jam["all1"], rng)
        if jam.get("ack", 0.0):
            ch_down = JammedChannel(ch_down, jam["ack"], rng)
        if jam.get("frag", 0.0):
            ch_up = JammedChannel(ch_up, jam["frag"], rng)
    return ch_up, ch_all1, ch_down, ch_ackreq


def simulate_packet(mode: Mode, p_u: float, cfg: SchcConfig, tim: Timing,
                    rng: np.random.Generator, alpha: float = 0.5,
                    channel: str = "iid", burst_len: float = 3.0,
                    energy: phy.EnergyModel = phy.DEFAULT_ENERGY,
                    jam: dict | None = None) -> RunResult:
    p_d = alpha * p_u
    # The All-1 fragment only benefits from the reduced loss probability when it
    # is actually hardened, i.e. transmitted at a higher spreading factor than
    # the regular fragments.  Sending it at the regular data rate leaves it
    # exposed to the same p_u as any other fragment.
    if mode == "optimistic":
        p_all1 = alpha * p_u if cfg.sf_all1_opt > cfg.sf_frag else p_u
    else:
        p_all1 = alpha * p_u if cfg.sf_all1_pess > cfg.sf_frag else p_u
    chans = _channels(p_u, p_d, p_all1, rng, channel, burst_len, jam)
    sim = _PacketSim(cfg, tim, energy, *chans)
    return sim.run_optimistic() if mode == "optimistic" else sim.run_pessimistic()


def simulate_batch(mode: Mode, p_u: float, n_runs: int, cfg: SchcConfig,
                   seed: int, alpha: float = 0.5, channel: str = "iid",
                   burst_len: float = 3.0, jam: dict | None = None,
                   tim: Timing | None = None) -> dict:
    """Run ``n_runs`` independent packet transfers and return raw per-run arrays."""
    tim = tim or Timing.from_config(cfg)
    rng = np.random.default_rng(seed)
    keys = ("n_ack_tx", "n_downlink_tx", "n_frag_tx", "n_all1_tx",
            "n_ackreq_tx", "delay_model", "delay_protocol", "delay_realistic",
            "energy_j", "toa_up_s", "toa_down_s")
    out = {k: np.empty(n_runs) for k in keys}
    delivered = np.zeros(n_runs, dtype=bool)
    false_ack = np.zeros(n_runs, dtype=bool)
    aborted = np.zeros(n_runs, dtype=bool)

    for i in range(n_runs):
        r = simulate_packet(mode, p_u, cfg, tim, rng, alpha, channel,
                            burst_len, jam=jam)
        out["n_ack_tx"][i] = r.n_ack_tx
        out["n_downlink_tx"][i] = r.n_downlink_tx
        out["n_frag_tx"][i] = r.n_frag_tx
        out["n_all1_tx"][i] = r.n_all1_tx
        out["n_ackreq_tx"][i] = r.n_ackreq_tx
        out["delay_model"][i] = r.delay_model
        out["delay_protocol"][i] = r.delay_protocol
        out["delay_realistic"][i] = r.delay_realistic
        out["energy_j"][i] = r.energy_j
        out["toa_up_s"][i] = r.toa_up_s
        out["toa_down_s"][i] = r.toa_down_s
        delivered[i] = r.delivered
        false_ack[i] = r.false_ack
        aborted[i] = r.aborted_by_sender

    out["delivered"] = delivered.astype(float)
    out["false_ack"] = false_ack.astype(float)
    out["aborted"] = aborted.astype(float)
    return out
