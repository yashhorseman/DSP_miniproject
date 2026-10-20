# Viva Notes: FFT-Based Vibration Fault Detection (simulation)

Setup to remember: **fs = 5000 Hz, T = 2 s, N = 10000, resolution fs/N = 0.5 Hz, shaft f_r = 50 Hz (3000 RPM), noise sigma = 0.25, seed 42.** Simulation only, no real machine.

## Concepts used

| Concept | One-line explanation | Where in project |
|---|---|---|
| Discrete-time signal | x[n] = x(n/fs), a sequence of samples | `time_axis`, `make_*` |
| Sampling / Nyquist | Need fs > 2 f_max. Here f_max (1287 Hz) < 2500 Hz = fs/2 | fs = 5000 Hz |
| Noise | Zero-mean white Gaussian, `np.random.normal`, std sigma | `add_noise` |
| DFT | X[k] = sum x[n] e^(-j2pi kn/N); frequency of bin k is k fs/N | `np.fft.rfft` |
| FFT | Fast algorithm for the DFT, O(N log N) instead of O(N^2) | `rfft` |
| Single-sided amplitude spectrum | 2/N * abs(X[k]); DC/Nyquist not doubled | `amplitude_spectrum` |
| Resolution | delta f = fs/N = 1/T = 0.5 Hz | demo `07` |
| Leakage | Tone not on a bin spreads over many bins | demo `06` |
| Window (Hann) | Tapers the record: lower leakage, wider main lobe | demo `06` |
| Butterworth filter | Maximally flat passband; order sets roll-off | `signal.butter` |
| SOS | Cascade of 2nd-order sections; numerically stable at high order | `output='sos'` |
| Zero-phase filtering | `sosfiltfilt` runs forward and backward: no phase shift, response is abs(H)^2 | filtering step |
| Peak detection | `find_peaks` with height and prominence thresholds | peak table |
| Envelope | Magnitude of the analytic signal (Hilbert transform) | bearing demo |
| Time-domain indicators | RMS, peak, crest factor = peak/RMS, kurtosis | stats table |

## Why these parameters

- **N = 10000 and fs = 5000** make the bin spacing 0.5 Hz, so 25, 50, 100, 150, 1113, 1200 and 1287 Hz all lie exactly on bins. Rectangular window then gives no leakage for those tones.
- **Low-pass 400 Hz:** signal content is at 25-150 Hz; 400 Hz keeps it with margin and removes noise above.
- **Band-pass 1000-1400 Hz:** centred on the 1200 Hz resonance; width 400 Hz is more than the 174 Hz sideband span (2 x 87).
- **Bearing sidebands:** A(1 + m sin(wf t)) sin(wr t) = A sin(wr t) + (Am/2)[cos((wr - wf)t) - cos((wr + wf)t)], so sidebands of A m / 2 = 0.2 at 1200 +/- 87 Hz.

## Key results to quote

- Every injected component detected at the exact frequency; max amplitude error 0.006 in one run (tolerance 0.02). Over 50 runs the mean error is at most 0.0008 and the std is about 0.0035, matching theory sigma*sqrt(2/N).
- Imbalance 1x = 2.5056 (injected 2.5). Misalignment 2x = 1.1953, 3x = 0.8056, 0.5x = 0.1491. Bearing: 1200 Hz 0.5018, sidebands 0.1946 and 0.1997 (injected 0.2).
- Filtering noise reduction 8.4-8.5 dB (theory for 400 Hz of 2500 Hz: 7.96 dB). SNR before/after: imbalance 17.05 -> 25.52 dB, misalignment 14.02 -> 22.41 dB, bearing -5.70 -> 12.67 dB.
- Envelope spectrum peak at 87.0 Hz (0.379); raw spectrum at 87 Hz only 0.0067.
- Off-bin leakage demo: 50.3 Hz tone reads 0.7556 with rectangular, 0.9010 with Hann (true value 1).
- Toy classifier 100 % on simulated data (not a real-world claim). Weak peaks lost at sigma of about 2-3.

## Likely viva questions

**1. What is the difference between the DFT and the FFT?**
They give the same result. The DFT is the definition (O(N^2) operations); the FFT is an efficient algorithm (O(N log N)) that exploits symmetry in the complex exponentials.

**2. Why did you use `rfft`?**
The signal is real, so the spectrum is conjugate-symmetric. `rfft` returns only the N/2 + 1 non-negative frequencies, which is all we need and is faster.

**3. Why multiply by 2/N?**
A real sinusoid of amplitude A splits into two bins of height A N/2 (at +f and -f). Taking only positive frequencies, we double the magnitude and divide by N to read A directly. DC and Nyquist have no mirror image, so they are not doubled.

**4. Why did your peaks fall exactly on bins and what if they did not?**
Bin spacing is 0.5 Hz and every frequency is a multiple of it, so an integer number of cycles fits in the record. Otherwise the tone leaks into neighbouring bins and the peak amplitude is underestimated (our 50.3 Hz demo read 0.7556 instead of 1). A window such as Hann reduces leakage but widens the peak.

**5. What limits frequency resolution?**
delta f = fs/N = 1/T. Only a longer record improves it; zero-padding only interpolates. Our demo: tones 3 Hz apart are resolved at N = 10000 but not at N = 500 or 1000.

**6. What is the Nyquist frequency here and what if the sampling were too low?**
2500 Hz. Components above it would alias to lower frequencies and look like false peaks. The highest component we use is 1287 Hz, so there is no aliasing.

**7. Why Butterworth?**
Flat passband with no ripple, simple design, and the passband amplitudes we care about (peak heights) are not distorted.

**8. Why `sos` and `sosfiltfilt`?**
`sos` avoids the numerical problems of high-order transfer-function coefficients. `sosfiltfilt` filters forward and backward, so the phase is zero and the waveform is not delayed, but the magnitude response is squared (steeper) and the filter cannot be used in real time.

**9. How did you measure noise reduction?**
SNR against the known clean signal before and after filtering, and the noise rms before and after. Reduction was 8.4-8.5 dB; for white noise the theoretical value is 10 log10(2500/400) = 7.96 dB.

**10. Why does a bearing fault show up as sidebands?**
It is amplitude modulation: a carrier at the structural resonance (1200 Hz) multiplied by (1 + m sin(2 pi f_fault t)). Multiplication in time becomes shifting in frequency, giving components at f_res +/- f_fault.

**11. Why is envelope analysis needed for real bearings?**
Real bearing defects produce short impulses that excite a resonance. The fault rate is in the modulation of the resonance band, not in a clean spectral line at the fault rate. Band-pass around the resonance, take the envelope, then FFT it. We showed this: 87 Hz absent in the raw spectrum (0.0067) but dominant in the envelope spectrum (0.379).

**12. How do the faults differ in the spectrum?**
Imbalance: large 1x. Misalignment: large 2x and 3x (and some 0.5x here). Bearing: high-frequency resonance with +/- f_fault sidebands. These are the textbook signatures *as modelled here*; real machines show overlaps.

**13. Why not just use RMS or kurtosis?**
They are single numbers that say something changed but not what. Over 50 runs, kurtosis was healthy 1.90, imbalance 1.53, misalignment 2.98, bearing 2.30: repeatable (std about 0.01) but not ordered by fault type, and only meaningful against a healthy baseline. The spectrum separates faults by frequency.

**14. What does the toy classifier prove?**
Very little: its thresholds are midpoints of amplitudes we defined, so 100 % accuracy on simulated data is expected. It only shows the spectral features are separable in this model.

**15. What are the limitations of FFT analysis?**
Needs stationary signals; variable speed smears peaks; leakage; fixed resolution/time trade-off (long record = good frequency resolution, poor time localisation); our fault models are simplified; simulation is not validation on real machinery.

**16. At what noise level do the peaks vanish?**
In our sweep (N = 10000) the weak peaks (0.5x at 0.15, sidebands at 0.2) fall below 95 % detection at sigma of about 2-3, versus 0.25 normally. Strong peaks survive far higher noise. Averaging or a longer record lowers the noise floor.

**17. Why a fixed random seed?**
`np.random.seed(42)` makes noise realisations reproducible so the report numbers can be regenerated exactly.

**18. What would you do next?**
Use real accelerometer data, add variable speed and order tracking, use STFT/wavelets for non-stationary signals, and use bearing defect frequency formulas (BPFO/BPFI) in a more realistic model.

## Extended studies: extra viva questions

**19. How do you compute the bearing fault frequencies?**
BPFO = (n/2) f_r (1 - (d/D) cos phi) and BPFI = (n/2) f_r (1 + (d/D) cos phi), with n balls, ball diameter d, pitch diameter D and contact angle phi. For 9 balls, d = 7.94 mm, D = 39.04 mm at 50 Hz: BPFO = 179.24 Hz (3.5848 x f_r), BPFI = 270.76 Hz (5.4152 x f_r). Check: BPFO + BPFI = n f_r. The geometry is an assumed 6205-type value, not measured.

**20. BPFO is not on an FFT bin. What happens?**
179.24 Hz is 358.478 bins away from DC, so the line lies between bins. The rectangular window read the envelope line about 35 % low (0.1477 against 0.2257 for an aligned frequency) and Hann about 15 % low (0.1920 against 0.2251). It is scalloping loss; the position is still found to within a bin.

**21. Why did an impulsive model raise the kurtosis when the AM model did not?**
Kurtosis measures peakedness. A sparse train of short bursts has a few large values (kurtosis 6.93 above 500 Hz against 2.99 for healthy noise), but a steady amplitude-modulated carrier looks like a sinusoid (2.78). A narrow 400 Hz band-pass smears the 0.5 ms bursts into each other and brought the kurtosis back to 2.69, so the band choice matters (that is what a kurtogram optimises).

**22. Does Welch averaging help detect weak peaks?**
It lowers the scatter of the noise floor (coefficient of variation 0.52 to 0.12-0.17) and beats a single FFT of the same segment length (at sigma = 2 it found the 1287 Hz sideband in 96-98 % of runs against 8 %). But the full-length FFT was still better for stationary tones (0.82 against 0.46 or 0.14 at sigma = 3), because a long record integrates a tone coherently and the noise per bin falls with bin width. So: record longer for weak stationary lines; use Welch for smooth floor estimates, non-stationary records and unaligned tones.

**23. What does an STFT show that an FFT does not?**
When the frequency content changes. With imbalance starting at 1 s, one FFT gave a blended 1x amplitude (1.746, about the average of 1.0 and 2.5); the STFT showed 1.0 before and 2.5 after, with onset estimated at 1.01-1.02 s. A longer window sharpens frequency resolution but slows the transition (10-90 % rise time 0.02, 0.04 and 0.10 s for 0.04, 0.10 and 0.20 s windows).

**24. What does a drifting speed do to the spectrum and how is it fixed?**
Each harmonic k sweeps from k f0 to k f1, so the peak spreads over many bins and drops (for 50 to 56 Hz: 1x from 1.0 to 0.356, 3x from 0.8 to 0.163). Order tracking resamples at equal shaft-angle steps using the speed (tachometer) signal; the harmonics become fixed orders again (1.200 for 2x, injected 1.2). 106 whole revolutions make the 0.5x order fall on a bin.

**25. Why Butterworth and not Chebyshev, Bessel or FIR?**
The harmonic amplitudes are what we measure, and Butterworth leaves them unchanged (0.00 dB at 50, 100, 150 Hz). Chebyshev I rolls off faster but its 1 dB ripple changes amplitudes by up to 0.80 dB (about 9 %). Bessel has the flattest delay but a poor roll-off (-15 dB at 800 Hz). FIR has linear phase and -64 dB at 800 Hz but needs 101 taps and a 10 ms delay. We filter offline with forward-backward filtering, so delay does not matter.

**26. Why is the severity sweep useful?**
It is the basis of trending in predictive maintenance: the measured 1x amplitude follows the injected severity linearly (slope 1.0001, scatter 0.003-0.004), so a rising 1x amplitude gives a quantitative warning before failure.

**27. What goes wrong with a single-label classifier for combined faults?**
It can output only one label by rule priority: imbalance plus misalignment was always reported as Misalignment, and any combination with a bearing fault as Bearing fault. Multi-label rules (each rule tested independently) reported both faults in all 50 runs of each combination, but, again, their thresholds come from the simulated model.

**28. How did you check the FFT itself?**
A direct DFT matrix product agrees with `np.fft.fft` to 9.88e-11 (relative about 1e-13); Parseval's theorem (energy in time equals energy in frequency) holds to about 1e-16; run time scales as N^2 for the direct DFT (slope 1.98) and about N log N for the FFT (slope 1.19), about 4700 times faster at N = 2048 on our machine.

## Quick formulas

- Bin frequency: f_k = k fs / N. Resolution: fs / N.
- Single-sided amplitude: A_k = 2 abs(X[k]) / N (windowed: 2 abs(X[k]) / sum(w)).
- SNR [dB] = 10 log10(P_signal / P_noise). Crest factor = peak / RMS.
- White-noise power reduction by a filter of bandwidth B out of fs/2: 10 log10((fs/2) / B).
- Per-bin amplitude noise std (rect window): about sigma * sqrt(2/N) = 0.0035 here.
