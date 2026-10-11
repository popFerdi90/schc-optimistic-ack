"""Fast checks of the reproduction package (a few seconds, standard library only).

    python -m unittest discover -s tests -v        # or: make test

They do not re-run the campaign (that is `make verify`); they check that the
building blocks are deterministic and that the shipped results, figures and
manuscript are consistent with one another.
"""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulator import experiments as E                      # noqa: E402
from simulator.schc_aoe import simulate_batch               # noqa: E402

EXPECTED_RESULTS = ("validation", "comparison", "gilbert_elliott", "alpha",
                    "energy_budget", "aass", "adversary", "probing")


class Seeds(unittest.TestCase):
    def test_seed_is_stable_across_processes(self):
        # Hard-coded on purpose: a change here means the campaign no longer
        # reproduces the published numbers. (hash() would differ per process.)
        self.assertEqual(E._seed("validation", 0), 821886110)
        self.assertEqual(E._seed("step-v2", 128), 2088644061)
        self.assertEqual(E._seed("comparison", 3), 240379071)

    def test_seed_depends_on_name_and_index(self):
        self.assertNotEqual(E._seed("alpha", 0), E._seed("alpha", 1))
        self.assertNotEqual(E._seed("alpha", 0), E._seed("adversary", 0))


class Simulation(unittest.TestCase):
    def test_same_seed_same_result(self):
        a = simulate_batch("optimistic", 0.3, 60, E.CFG, seed=123)
        b = simulate_batch("optimistic", 0.3, 60, E.CFG, seed=123)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)

    def test_different_seed_differs(self):
        a = simulate_batch("optimistic", 0.3, 60, E.CFG, seed=1)
        b = simulate_batch("optimistic", 0.3, 60, E.CFG, seed=2)
        self.assertFalse(np.array_equal(a["n_downlink_tx"], b["n_downlink_tx"]))

    def test_optimistic_saves_downlink(self):
        pess = simulate_batch("pessimistic", 0.2, 200, E.CFG, seed=7)
        opt = simulate_batch("optimistic", 0.2, 200, E.CFG, seed=7)
        self.assertLess(opt["n_downlink_tx"].mean(), pess["n_downlink_tx"].mean())


class RepresentativeStream(unittest.TestCase):
    """Rule that picks the stream drawn in Fig. 8a (see experiments.py)."""

    def test_closest_delay_among_streams_near_the_mean_switch_count(self):
        sw = [2, 2, 4, 2, 4]            # mean 2.8, tolerance 1.1: only the 2s qualify
        ad = [124, 38, 40, 36, 38.2]    # mean 55.24
        # eligible: indices 0, 1, 3 with distances 68.76, 17.24, 19.24
        self.assertEqual(E._representative_rep(sw, ad), 1)

    def test_outlier_in_switch_count_is_excluded(self):
        sw = [3] * 9 + [12]             # mean 3.9; the stream with 12 changes is out
        ad = [10, 20, 30, 40, 60, 60, 70, 80, 90, 49.5]   # mean 50.95
        # index 9 has the delay closest to the mean but is excluded; index 4 wins
        self.assertEqual(E._representative_rep(sw, ad), 4)

    def test_ties_go_to_lowest_index(self):
        self.assertEqual(E._representative_rep([5, 5, 5], [7, 7, 7]), 0)


class ShippedResults(unittest.TestCase):
    def test_all_json_present_and_loadable(self):
        for name in EXPECTED_RESULTS:
            path = ROOT / "results" / f"{name}.json"
            self.assertTrue(path.exists(), path)
            json.loads(path.read_text())

    def test_fig8a_trace_matches_caption(self):
        """The numbers quoted in the Fig. 8 caption come from results/aass.json."""
        step = json.loads((ROOT / "results" / "aass.json").read_text())["step"]
        tex = (ROOT / "sn-article.tex").read_text()
        for tag, label in (("v1", "AASS"), ("v2", "AASS-H")):
            s = step[tag]
            self.assertEqual(len(s["phat"]), 900)
            self.assertEqual(len(s["ctrl_optimistic"]), 900)
            self.assertEqual(s["n_rep"], 200)
            sw, ad = int(s["trace_switches"]), int(s["trace_delay"])
            pat = rf"{label}: \${sw}\$ mode changes, \${ad}\$ packets" if tag == "v1" \
                else rf"{label}: \${sw}\$ changes, \${ad}\$ packets"
            self.assertRegex(tex, pat)

    def test_figures_exist(self):
        for f in ("fig9_validation", "fig10_comparison", "fig11_robustness",
                  "fig13_aass", "fig14_adversary"):
            self.assertTrue((ROOT / "figs" / f"{f}.pdf").exists(), f)

    def test_every_figure_used_by_the_manuscript_exists(self):
        tex = (ROOT / "sn-article.tex").read_text()
        for rel in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex):
            self.assertTrue((ROOT / rel).exists(), rel)


class Manuscript(unittest.TestCase):
    def test_audit_tables_passes(self):
        r = subprocess.run([sys.executable, str(ROOT / "audit_tables.py")],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("ALL CELLS MATCH THEIR SOURCE", r.stdout)


if __name__ == "__main__":
    unittest.main()
