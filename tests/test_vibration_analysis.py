"""Unit tests for vibration_analysis.py.

Run from the project folder:  python -m unittest discover -s tests -v

The script executes the whole pipeline when imported, so these tests load only
its imports, config assignments and function definitions (via ast) and never
run the plotting/analysis steps.
"""
import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT = Path(__file__).resolve().parent.parent / "vibration_analysis.py"
CONFIG_NAMES = {"CFG", "CASES", "COLORS", "MAKERS"}


def load_namespace():
    """Exec only imports, config assignments and defs from the script."""
    tree = ast.parse(SCRIPT.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in CONFIG_NAMES:
                keep.append(node)
            elif (isinstance(tgt, ast.Subscript)
                  and getattr(tgt.value, "id", "") == "CFG"):
                keep.append(node)
    ns = {}
    exec(compile(ast.Module(keep, []), str(SCRIPT), "exec"), ns)
    return ns


NS = load_namespace()
np = NS["np"]
CFG = NS["CFG"]
FS, N, DF = CFG["fs"], CFG["N"], CFG["df"]
T = NS["time_axis"]()


class TestSpectrum(unittest.TestCase):
    def test_bin_aligned_tone_amplitude(self):
        f, a = NS["amplitude_spectrum"](2.0 * np.sin(2 * np.pi * 50 * T), FS)
        self.assertAlmostEqual(a[int(50 / DF)], 2.0, places=10)
        self.assertAlmostEqual(f[1] - f[0], DF)

    def test_dc_and_nyquist_not_doubled(self):
        _, a = NS["amplitude_spectrum"](np.full(N, 3.0), FS)
        self.assertAlmostEqual(a[0], 3.0, places=10)
        nyq = np.cos(np.pi * np.arange(N))
        _, a = NS["amplitude_spectrum"](nyq, FS)
        self.assertAlmostEqual(a[-1], 1.0, places=10)

    def test_hann_window_amplitude_corrected(self):
        x = 1.5 * np.sin(2 * np.pi * 100 * T)
        _, a = NS["amplitude_spectrum"](x, FS, window="hann")
        self.assertAlmostEqual(a.max(), 1.5, places=6)

    def test_off_bin_tone_loses_amplitude_with_rect(self):
        x = np.sin(2 * np.pi * CFG["leak_freq"] * T)
        _, a = NS["amplitude_spectrum"](x, FS)
        self.assertLess(a.max(), 0.8)

    def test_direct_dft_matches_fft(self):
        x = np.random.RandomState(0).randn(256)
        err = np.max(np.abs(NS["dft_direct"](x) - np.fft.fft(x)))
        self.assertLess(err, 1e-9)

    def test_parseval(self):
        x = np.random.RandomState(1).randn(N)
        e_t = np.sum(x ** 2)
        e_f = np.sum(np.abs(np.fft.fft(x)) ** 2) / N
        self.assertAlmostEqual(e_t / e_f, 1.0, places=10)


class TestSignalModel(unittest.TestCase):
    def test_noise_free_spectrum_matches_injected_components(self):
        for case in NS["CASES"]:
            x = NS["MAKERS"][case](noise=False)
            _, a = NS["amplitude_spectrum"](x, FS)
            for fk, ak, _ in NS["injected_components"](case):
                self.assertAlmostEqual(a[int(round(fk / DF))], ak, places=6,
                                       msg=f"{case} at {fk} Hz")

    def test_noise_reproducible_and_has_requested_std(self):
        np.random.seed(5)
        a = NS["make_healthy"]()
        np.random.seed(5)
        b = NS["make_healthy"]()
        self.assertTrue(np.array_equal(a, b))
        noise = a - NS["make_healthy"](noise=False)
        self.assertAlmostEqual(noise.std(), CFG["sigma"], delta=0.01)

    def test_am_sidebands_are_a_m_over_two(self):
        comps = {lab: amp for _, amp, lab in
                 NS["injected_components"]("Bearing fault")}
        self.assertAlmostEqual(comps["resonance + f_fault"],
                               CFG["a_res"] * CFG["mod_depth"] / 2)

    def test_label_peak(self):
        self.assertEqual(NS["label_peak"](100.5, "Healthy")[0], "2x")
        self.assertEqual(NS["label_peak"](77.0, "Healthy")[0], "unassigned")


class TestIndicators(unittest.TestCase):
    def test_snr_known_value(self):
        ref = np.sin(2 * np.pi * 50 * T)
        self.assertAlmostEqual(NS["snr_db"](ref, 1.1 * ref), 20.0, places=6)

    def test_time_stats_square_wave(self):
        x = np.tile([1.0, -1.0], 500)
        rms, peak, crest, kurt = NS["time_stats"](x)
        self.assertAlmostEqual(rms, 1.0)
        self.assertAlmostEqual(crest, 1.0)
        self.assertAlmostEqual(kurt, 1.0)


class TestFilters(unittest.TestCase):
    def test_filter_gains(self):
        lp, bp = NS["design_filters"]()
        sig = NS["signal"]
        _, h = sig.sosfreqz(lp, worN=[50, 1500], fs=FS)
        self.assertAlmostEqual(abs(h[0]), 1.0, places=3)
        self.assertLess(20 * np.log10(abs(h[1])), -40)
        _, h = sig.sosfreqz(bp, worN=[1200, 100], fs=FS)
        self.assertAlmostEqual(abs(h[0]), 1.0, places=2)
        self.assertLess(20 * np.log10(abs(h[1])), -60)


class TestClassifier(unittest.TestCase):
    def test_clean_signals_classified_correctly(self):
        for case in NS["CASES"]:
            _, a = NS["amplitude_spectrum"](NS["MAKERS"][case](noise=False), FS)
            self.assertEqual(NS["classify"](a), case)

    def test_flags_detect_combined_faults(self):
        x = NS["make_combined"](imbalance=True, misalignment=True, noise=False)
        _, a = NS["amplitude_spectrum"](x, FS)
        self.assertEqual(set(NS["detect_flags"](a)),
                         {"Imbalance", "Misalignment"})


class TestExtensions(unittest.TestCase):
    def test_bearing_frequency_identities(self):
        bf = NS["bearing_frequencies"](50.0, 9, 7.94, 39.04, 0.0)
        self.assertAlmostEqual(bf["BPFO"] + bf["BPFI"], 9 * 50.0, places=9)
        self.assertAlmostEqual(bf["BPFO"] / 50.0, 3.5848, places=3)
        self.assertAlmostEqual(bf["BPFI"] / 50.0, 5.4152, places=3)

    def test_stft_of_constant_tone_is_flat(self):
        x = np.sin(2 * np.pi * 50 * T)
        _, f, a = NS["stft_amplitude"](x, FS, 500, 100)
        row = a[int(round(50 / (FS / 500)))]
        self.assertTrue(np.allclose(row, 1.0, atol=1e-6))

    def test_order_tracking_recovers_sharp_orders(self):
        x, revs = NS["make_variable_speed"](noise=False)
        xu, spr = NS["order_track"](x, revs, T)
        orders, a = NS["amplitude_spectrum"](xu, spr)
        d = orders[1]
        for k, ak in [(1, 1.0), (2, CFG["mis_a2"]), (3, CFG["mis_a3"])]:
            self.assertAlmostEqual(a[int(round(k / d))], ak, delta=0.02)

    def test_severity_linearity(self):
        cfg = dict(CFG, imb_a1=3.0)
        _, a = NS["amplitude_spectrum"](NS["make_imbalance"](cfg, noise=False), FS)
        self.assertAlmostEqual(a[int(CFG["f_r"] / DF)], 3.0, places=6)

    def test_lowpass_candidates_dc_gain(self):
        """Unity DC gain, except even-order Chebyshev I (-1 dB by design)."""
        for name, kc in NS["lowpass_candidates"]().items():
            _, h = NS["filter_response"](kc, [0.0, 10.0], FS)
            expected = 10 ** (-1 / 20) if name.startswith("Chebyshev") else 1.0
            self.assertAlmostEqual(abs(h[0]), expected, delta=0.01, msg=name)


if __name__ == "__main__":
    unittest.main()
