"""Confidence intervals and paired significance tests."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Mean and half-width of the Student confidence interval."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = x.std(ddof=1) / np.sqrt(n)
    h = float(sps.t.ppf(0.5 + conf / 2.0, n - 1) * se)
    return m, m - h, m + h


def proportion_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because several probabilities of
    interest (false-ACK rate at low loss) are close to zero.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = sps.norm.ppf(0.5 + conf / 2.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired comparison of two approaches on common random numbers.

    Reports both the paired Student test and the non-parametric Wilcoxon
    signed-rank test, since most of the metrics involved (ACK counts, delays)
    are strongly non-Gaussian.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    res = {"mean_diff": float(d.mean()), "n": int(d.size)}
    if np.allclose(d, 0.0):
        res.update(t_p=1.0, wilcoxon_p=1.0, cohen_d=0.0)
        return res
    t_stat, t_p = sps.ttest_rel(a, b)
    try:
        w_stat, w_p = sps.wilcoxon(a, b, zero_method="wilcox",
                                   alternative="two-sided")
    except ValueError:
        w_p = float("nan")
    sd = d.std(ddof=1)
    res.update(t_p=float(t_p), wilcoxon_p=float(w_p),
               cohen_d=float(d.mean() / sd) if sd > 0 else 0.0)
    return res


def relative_error(analytic: float, simulated: float) -> float:
    """Signed relative deviation of the model from the simulation, in percent."""
    if simulated == 0.0:
        return float("nan")
    return 100.0 * (analytic - simulated) / simulated
