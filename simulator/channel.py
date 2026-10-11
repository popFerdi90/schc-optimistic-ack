"""
Channel models for SCHC fragment and ACK losses.

Two erasure models are provided:

* :class:`IIDChannel` -- the memoryless model assumed in Section 4.2.1 of the
  paper, where every frame is erased independently with probability ``p``.
* :class:`GilbertElliottChannel` -- a two-state Markov (Gilbert-Elliott) model
  producing correlated, bursty erasures, used in Section 5 to test the
  robustness of the conclusions drawn under the i.i.d. assumption.
* :class:`TraceChannel` -- replays a measured reception trace, so that the
  campaign can be re-run against field data without touching any other module.

The Gilbert-Elliott chain is calibrated so that its stationary erasure rate
equals a target mean ``p_mean`` and its mean sojourn time in the BAD state
equals ``burst_len`` frames.  With ``burst_len = 1`` and the default state
erasure rates the model degenerates towards an i.i.d. channel, which makes the
two models directly comparable at equal average loss.
"""
from __future__ import annotations

import numpy as np

# Erasure probability inside each state of the Gilbert-Elliott chain.
P_GOOD_DEFAULT = 0.01
P_BAD_DEFAULT = 0.95


class IIDChannel:
    """Memoryless erasure channel."""

    __slots__ = ("p", "rng")

    def __init__(self, p: float, rng: np.random.Generator):
        self.p = float(p)
        self.rng = rng

    def lost(self) -> bool:
        return self.rng.random() < self.p

    def reset(self) -> None:          # no memory to reset
        pass

    @property
    def mean_loss(self) -> float:
        return self.p


class GilbertElliottChannel:
    """Two-state Markov erasure channel (Gilbert-Elliott).

    States: 0 = GOOD (erasure probability ``p_good``),
            1 = BAD  (erasure probability ``p_bad``).

    Transition probabilities are derived from the target stationary erasure
    rate and the target mean burst length::

        delta = P(BAD -> GOOD)  = 1 / burst_len
        pi_bad = (p_mean - p_good) / (p_bad - p_good)
        gamma = P(GOOD -> BAD)  = delta * pi_bad / (1 - pi_bad)

    The chain advances one step per transmitted frame.
    """

    __slots__ = ("p_good", "p_bad", "gamma", "delta", "pi_bad", "state",
                 "rng", "_p_mean")

    def __init__(self, p_mean: float, rng: np.random.Generator,
                 burst_len: float = 3.0,
                 p_good: float = P_GOOD_DEFAULT,
                 p_bad: float = P_BAD_DEFAULT):
        p_mean = float(p_mean)
        if not (p_good < p_bad):
            raise ValueError("p_good must be smaller than p_bad")
        # Clamp the target mean into the reachable interval (p_good, p_bad).
        eps = 1e-9
        p_eff = min(max(p_mean, p_good + eps), p_bad - eps)

        self._p_mean = p_mean
        self.p_good = p_good
        self.p_bad = p_bad
        self.delta = 1.0 / burst_len
        self.pi_bad = (p_eff - p_good) / (p_bad - p_good)
        self.gamma = self.delta * self.pi_bad / (1.0 - self.pi_bad)
        self.rng = rng
        # Start in the stationary distribution so that no warm-up is needed.
        self.state = 1 if rng.random() < self.pi_bad else 0

    def reset(self) -> None:
        """Redraw the initial state from the stationary distribution."""
        self.state = 1 if self.rng.random() < self.pi_bad else 0

    def lost(self) -> bool:
        p_err = self.p_bad if self.state else self.p_good
        erased = self.rng.random() < p_err
        # advance the chain
        if self.state:
            if self.rng.random() < self.delta:
                self.state = 0
        else:
            if self.rng.random() < self.gamma:
                self.state = 1
        return erased

    @property
    def mean_loss(self) -> float:
        return self.pi_bad * self.p_bad + (1.0 - self.pi_bad) * self.p_good


class JammedChannel:
    """Wraps a channel with an active adversary that erases a target frame class.

    The adversary erases the frame with probability ``p_jam`` independently of
    the natural channel state; the frame is lost if either the natural channel
    erases it or the adversary jams it.
    """

    __slots__ = ("base", "p_jam", "rng")

    def __init__(self, base, p_jam: float, rng: np.random.Generator):
        self.base = base
        self.p_jam = float(p_jam)
        self.rng = rng

    def lost(self) -> bool:
        natural = self.base.lost()
        return natural or (self.rng.random() < self.p_jam)

    def reset(self) -> None:
        self.base.reset()

    @property
    def mean_loss(self) -> float:
        return 1.0 - (1.0 - self.base.mean_loss) * (1.0 - self.p_jam)


class TraceChannel:
    """Erasure channel replayed from a measured reception trace.

    ``trace`` is a sequence of 0/1 flags, one per transmitted frame, where 1
    marks a frame that was *not* received.  The trace is replayed cyclically
    and each simulated packet starts at an independent random offset, so that
    a finite trace yields uncorrelated replications while preserving the
    empirical burst structure.

    This is the only class that has to be provided in order to run the whole
    campaign against measured data instead of a parametric loss model: the
    fragmentation engine consumes channels exclusively through ``lost()``,
    ``reset()`` and ``mean_loss``.  See ``example_trace.py``.
    """

    __slots__ = ("trace", "rng", "i")

    def __init__(self, trace, rng: np.random.Generator):
        self.trace = np.asarray(trace, dtype=bool)
        if self.trace.size == 0:
            raise ValueError("empty trace")
        self.rng = rng
        self.i = int(self.rng.integers(self.trace.size))

    def lost(self) -> bool:
        v = bool(self.trace[self.i])
        self.i = (self.i + 1) % self.trace.size
        return v

    def reset(self) -> None:
        self.i = int(self.rng.integers(self.trace.size))

    @property
    def mean_loss(self) -> float:
        return float(self.trace.mean())


_TRACE_CACHE: dict[str, np.ndarray] = {}


def load_trace(path: str) -> np.ndarray:
    """Load a reception trace: one 0/1 flag per frame, 1 = frame lost.

    Accepted formats are one value per line, or a single comma-separated line.
    Blank lines and lines starting with ``#`` are ignored.  Results are cached
    so that a campaign reading the same trace thousands of times pays the
    parsing cost once.
    """
    arr = _TRACE_CACHE.get(path)
    if arr is None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = [ln.split("#", 1)[0].strip() for ln in fh]
        toks: list[str] = []
        for ln in raw:
            if ln:
                toks.extend(t for t in ln.replace(",", " ").split() if t)
        if not toks:
            raise ValueError(f"trace file {path!r} contains no sample")
        arr = np.array([int(float(t)) != 0 for t in toks], dtype=bool)
        _TRACE_CACHE[path] = arr
    return arr


def make_channel(kind: str, p_mean: float, rng: np.random.Generator,
                 burst_len: float = 3.0):
    """Build an erasure channel.

    ``kind`` is ``"iid"``, ``"ge"``, or ``"trace:<path>"``.  In the trace case
    the measured process supersedes the parametric loss rates: ``p_mean`` and
    hence the asymmetry factor alpha are ignored, every frame class replaying
    the same empirical trace from an independent offset.
    """
    if kind == "iid":
        return IIDChannel(p_mean, rng)
    if kind == "ge":
        return GilbertElliottChannel(p_mean, rng, burst_len=burst_len)
    if kind.startswith("trace:"):
        return TraceChannel(load_trace(kind[len("trace:"):]), rng)
    raise ValueError(f"unknown channel model: {kind!r}")
