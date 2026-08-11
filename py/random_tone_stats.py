"""Random-tone SNDR statistics over the (a1, a2) grid.

Loop physics imported from noise_shp_sanity.run_loop. Each trial uses a
single sine at a RANDOM in-band frequency and phase (drawn once, seeded,
shared by every grid point so configurations see identical stimuli).
Frequencies are random *integer* FFT bins so every trial stays coherent and
the bin-sum SNDR stays exact.

Per (a1, a2): TRIALS runs at fixed AMP -> mean / min / std of SNDR.
Deliverables (in results/):
  random_tone_mean.npy / _min.npy / _std.npy, random_tone_maps.png,
  top-10 table on stdout, and random_tone_trace.png: input, v1, v2 sampled
  at each clock edge for 5 periods of the trial-0 tone at the best (a1, a2).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from noise_shp_sanity import (run_loop, draw_board, board_coeffs, afe_gain,
                              design_coeffs)

WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))  # each worker holds its own
                                                     # copy of the 8 stimuli (~210 MB)

# --- fixed configuration ---
FS = 10e6
N = 2**16
VREF = 2.5
OSR = 128
BITS = 3
SLICES = 50
T_PROP, T_OLD, T_BREAK = 36e-9, 10e-9, 12e-9
LADDER_TOL = 0.001
FB_TOL = 0.001
MODE = 1                # element selection: 0 = static, 1 = DWA, 2 = rotate
SEED = 1

AMP = 1              # drive AT THE JACK: tone Vpp = AMP * 2.5 V (AMP = 1
                        # swings 0 to 2.5 V). The AFE divider (0.435) scales it
                        # to the node. 1.29 = the design point: 3.2 Vpp jack ->
                        # 0.70 V node peak. Stability floor: rho1 = 0.435 * AMP.
TRIALS = 16              # random tones per grid point
BIN_LO, BIN_HI = 5, 254 # random tone bin range: lobe +/-2 stays inside band bins 3..256

LEVELS = 2**BITS
DAC_STEP = VREF / (LEVELS - 1)
DT = 1 / FS / SLICES
BREAK_START = int(round((T_PROP + T_OLD) / DT))
BREAK_END = int(round((T_PROP + T_OLD + T_BREAK) / DT))
assert BREAK_END < SLICES

THRESHOLDS, W1, W2 = draw_board(np.random.default_rng(SEED), VREF, LEVELS,
                                LADDER_TOL, FB_TOL)

WINDOW = np.hanning(N)
BAND_EDGE = N // (2 * OSR)                # bin 256 = 39.06 kHz
INBAND = np.arange(3, BAND_EDGE + 1)

BOARD_A1, BOARD_A2, B1, B2 = board_coeffs(FS)
DESIGN_A1, DESIGN_A2 = design_coeffs(FS)


def make_tone(bin_k, phase):
    """Unit-peak sine at FFT bin `bin_k` on the fine grid."""
    t = np.arange(N * SLICES) * DT
    return np.sin(2 * np.pi * (FS * bin_k / N) * t + phase)


def sndr_of(output, bin_k):
    yf = np.fft.rfft(output * WINDOW)
    power = np.abs(yf[:N // 2]) ** 2
    sig = np.arange(bin_k - 2, bin_k + 3)
    noise = INBAND[~np.isin(INBAND, sig)]
    return 10 * np.log10(power[sig].sum() / power[noise].sum())


_worker = {}


def _init_worker(trial_bins, trial_phases, xscale):
    """Runs once in each worker process: build its private copy of the stimuli."""
    _worker['stimuli'] = [make_tone(b, p) for b, p in zip(trial_bins, trial_phases)]
    _worker['bins'] = trial_bins
    _worker['xscale'] = xscale


def _sweep_row(args):
    """One a1 row of the grid, all a2 values, all trials. Runs in a worker."""
    i, a1, a_vals = args
    stimuli = _worker['stimuli']
    bins = _worker['bins']
    xscale = _worker['xscale']
    n_a = len(a_vals)
    trials = len(bins)
    mean_row = np.full(n_a, np.nan)
    min_row = np.full(n_a, np.nan)
    std_row = np.full(n_a, np.nan)
    vals = np.empty(trials)
    for j, a2 in enumerate(a_vals):
        for tr in range(trials):
            out, _, _, div = run_loop(stimuli[tr], xscale, THRESHOLDS,
                                      a1, a2, B1, B2, SLICES,
                                      BREAK_START, BREAK_END, DAC_STEP,
                                      LEVELS, N, W1, W2, MODE)
            vals[tr] = np.nan if div >= 0 else sndr_of(out, bins[tr])
        if np.all(np.isfinite(vals)):
            mean_row[j] = vals.mean()
            min_row[j] = vals.min()
            std_row[j] = vals.std()
    return i, mean_row, min_row, std_row


def _bar(done, total, t0, label):
    el = time.time() - t0
    eta = el / done * (total - done) if done else 0
    filled = int(30 * done / total)
    print(f"\r[{'=' * filled}{' ' * (30 - filled)}] {label}  {done}/{total}  "
          f"elapsed {el / 60:5.1f} min  eta {eta / 60:5.1f} min ",
          end="", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = here / "results"
    results.mkdir(exist_ok=True)

    # the shared trial set: random coherent bins + random phases, seeded
    trial_rng = np.random.default_rng(SEED + 1)
    trial_bins = np.sort(trial_rng.choice(np.arange(BIN_LO, BIN_HI + 1),
                                          TRIALS, replace=False))
    trial_phases = trial_rng.uniform(0, 2 * np.pi, TRIALS)
    stimuli = [make_tone(b, p) for b, p in zip(trial_bins, trial_phases)]
    xscale = (AMP * VREF / 2) * afe_gain()  # jack Vpp = AMP*VREF, then the divider
    print("trial tones (kHz):",
          np.array2string(trial_bins * FS / N / 1e3, precision=2))

    a_vals = np.round(np.arange(0.20, 2.0001, 0.01), 3)   # 361 points/axis
    n_a = len(a_vals)
    sndr_mean = np.full((n_a, n_a), np.nan)
    sndr_min = np.full((n_a, n_a), np.nan)
    sndr_std = np.full((n_a, n_a), np.nan)

    # warm-up (compile) then calibrate the estimate
    run_loop(stimuli[0], xscale, THRESHOLDS, 0.5, 0.6, B1, B2, SLICES,
             BREAK_START, BREAK_END, DAC_STEP, LEVELS, N, W1, W2, MODE)
    t_cal = time.time()
    out, _, _, div = run_loop(stimuli[0], xscale, THRESHOLDS, 0.5, 0.6, B1, B2,
                              SLICES, BREAK_START, BREAK_END, DAC_STEP,
                              LEVELS, N, W1, W2, MODE)
    sndr_of(out, trial_bins[0])
    per_run = time.time() - t_cal
    total = n_a * n_a * TRIALS
    print(f"grid {n_a}x{n_a}, {TRIALS} trials each = {total} runs on {WORKERS} workers "
          f"-- estimated {total * per_run / 60 / WORKERS:.0f} min if all points were stable")

    t0 = time.time()
    rows_done = 0
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                             initargs=(trial_bins, trial_phases, xscale)) as pool:
        futures = [pool.submit(_sweep_row, (i, a1, a_vals))
                   for i, a1 in enumerate(a_vals)]
        for fut in as_completed(futures):
            i, mean_row, min_row, std_row = fut.result()
            sndr_mean[i] = mean_row
            sndr_min[i] = min_row
            sndr_std[i] = std_row
            rows_done += 1
            _bar(rows_done * n_a * TRIALS, total, t0, f"stats x{WORKERS}")
            if rows_done % 5 == 0:
                np.save(results / "random_tone_mean_partial.npy", sndr_mean)
    print()
    np.save(results / "random_tone_mean.npy", sndr_mean)
    np.save(results / "random_tone_min.npy", sndr_min)
    np.save(results / "random_tone_std.npy", sndr_std)

    # top 10 configurations by mean SNDR
    flat = np.argsort(np.nan_to_num(sndr_mean, nan=-1e9).ravel())[::-1][:10]
    print("\ntop 10 by mean SNDR over random tones:")
    print(f"{'rank':>4} {'rho1':>6} {'rho2':>6} {'a1':>6} {'a2':>6} "
          f"{'mean':>7} {'min':>7} {'std':>6}")
    for r, fi in enumerate(flat, 1):
        i, j = divmod(fi, n_a)
        print(f"{r:>4} {a_vals[i] / B1:>6.3f} {a_vals[j] / B2:>6.3f} "
              f"{a_vals[i]:>6.3f} {a_vals[j]:>6.3f} "
              f"{sndr_mean[i, j]:>7.1f} {sndr_min[i, j]:>7.1f} {sndr_std[i, j]:>6.2f}")

    # maps figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    ext = [a_vals[0] / B2, a_vals[-1] / B2, a_vals[0] / B1, a_vals[-1] / B1]
    for ax, data, ttl, cm in ((axes[0], sndr_mean, 'mean SNDR (dB)', 'magma'),
                              (axes[1], sndr_min, 'worst-trial SNDR (dB)', 'magma'),
                              (axes[2], sndr_std, 'std across trials (dB)', 'viridis')):
        cmap = plt.get_cmap(cm).copy()
        cmap.set_bad('black')
        im = ax.imshow(np.ma.masked_invalid(data), origin='lower',
                       aspect='auto', extent=ext, cmap=cmap)
        fig.colorbar(im, ax=ax, label=ttl)
        ax.set_xlabel('rho2 = R19 / (Rbank2/7)')
        ax.set_ylabel('rho1 = R11 / (Rbank1/7)')
        ax.plot(BOARD_A2 / B2, BOARD_A1 / B1, 'x', color='0.75',
                markersize=9, label='as-built (16.9k/13k)')
        ax.plot(DESIGN_A2 / B2, DESIGN_A1 / B1, '*', color='white',
                markersize=14, label='design (14.0k/7.32k)')
        ax.legend(loc='upper right', fontsize=8)
        # coefficient values on the secondary axes (a = rho * b at this fs / C)
        sx = ax.secondary_xaxis('top', functions=(lambda r: r * B2, lambda a: a / B2))
        sx.set_xlabel('a2')
        sy = ax.secondary_yaxis('right', functions=(lambda r: r * B1, lambda a: a / B1))
        sy.set_ylabel('a1')
    fig.savefig(results / "random_tone_maps.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {results / 'random_tone_maps.png'}")

    # --- node-voltage trace at the best configuration: 5 input periods ---
    bi, bj = divmod(np.nanargmax(sndr_mean), n_a)
    b_a1, b_a2 = a_vals[bi], a_vals[bj]
    f0 = FS * trial_bins[0] / N
    n_5 = int(np.ceil(5 * N / trial_bins[0]))      # clock cycles in 5 periods
    x5 = stimuli[0][:n_5 * SLICES]
    _, v1_t, v2_t, _ = run_loop(x5, xscale, THRESHOLDS, b_a1, b_a2, B1, B2,
                                SLICES, BREAK_START, BREAK_END, DAC_STEP,
                                LEVELS, n_5, W1, W2, MODE)
    t_us = np.arange(n_5) / FS * 1e6
    x_edges = xscale * stimuli[0][::SLICES][:n_5]  # input at each clock edge

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_us, x_edges, label='input', linewidth=1.5)
    ax.plot(t_us, v1_t, label='v1 (integrator 1)', linewidth=0.8)
    ax.plot(t_us, v2_t, label='v2 (integrator 2)', linewidth=0.8)
    ax.set_xlabel('time (us)')
    ax.set_ylabel('volts (centered frame)')
    ax.set_title(f'Node voltages at each clock edge -- best (a1={b_a1}, a2={b_a2}), '
                 f'tone {f0 / 1e3:.2f} kHz, 5 periods, amp={AMP}')
    ax.grid(True)
    ax.legend()
    fig.savefig(results / "random_tone_trace.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {results / 'random_tone_trace.png'}")
