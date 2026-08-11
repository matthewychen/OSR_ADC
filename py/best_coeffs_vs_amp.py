"""Where does the best (a1, a2) sit as drive level changes -- per test tone?

For each tone in TONES_HZ and each amp (jack Vpp / 2.5 V), sweep the
(a1, a2) grid with one tone per point, take the TOP_K cells by SNDR, and
record the average a1 and a2 of that group. Output: an animated GIF, one
frame per tone (2 s each), each frame the a1/a2-vs-amp line chart, plus
results/best_coeffs_vs_amp.npz holding every curve.

Record length per tone: the run is N clock periods = N/fs seconds, and a
tone is only measurable if it completes at least ~5 cycles in the record
(coherent bin >= 5, clear of the DC-leakage guard). resolve_tone() picks
the smallest N (up to 2^20) that satisfies that, so low tones cost more:
50 Hz runs a 16x longer record than 1 kHz and dominates the runtime.
Worker count shrinks for long records to bound stimulus memory.

Parallel over amps within each tone.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from noise_shp_sanity import (run_loop, draw_board, board_coeffs, afe_gain,
                              design_coeffs)

WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))
MEM_BUDGET = 4e9        # bytes across all workers' stimulus copies

# --- fixed configuration ---
FS = 10e6
VREF = 2.5
OSR = 128
BITS = 3
SLICES = 50
T_PROP, T_OLD, T_BREAK = 36e-9, 10e-9, 12e-9
LADDER_TOL = 0.001
FB_TOL = 0.001
MODE = 1
SEED = 1

TONES_HZ = [1000, 5000, 10000, 30000]
SECONDS_PER_FRAME = 2
AMPS = np.round(np.arange(0.10, 3.4001, 0.05), 2)
A_VALS = np.round(np.arange(0.24, 1.30001, 0.02), 2)
TOP_K = 5

LEVELS = 2**BITS
DAC_STEP = VREF / (LEVELS - 1)
DT_CLK = 1 / FS
BREAK_START = int(round((T_PROP + T_OLD) / (DT_CLK / SLICES)))
BREAK_END = int(round((T_PROP + T_OLD + T_BREAK) / (DT_CLK / SLICES)))
assert BREAK_END < SLICES

THRESHOLDS, W1, W2 = draw_board(np.random.default_rng(SEED), VREF, LEVELS,
                                LADDER_TOL, FB_TOL)
BOARD_A1, BOARD_A2, B1, B2 = board_coeffs(FS)


def resolve_tone(f_hz):
    """Smallest record putting this tone on a coherent bin >= 5.
    Returns (n_periods, bin_k, realized_hz)."""
    for p in (16, 17, 18, 19, 20):
        n = 2**p
        b = round(f_hz * n / FS)
        if b >= 5:
            return n, b, FS * b / n
    n = 2**20
    b = max(5, round(f_hz * n / FS))
    return n, b, FS * b / n


def workers_for(n):
    per_worker = n * SLICES * 8 * 1.6
    return max(1, min(WORKERS, int(MEM_BUDGET / per_worker)))


_worker = {}


def _init_worker(n, bin_k, phase):
    dt = DT_CLK / SLICES
    t = np.arange(n * SLICES) * dt
    _worker['x'] = np.sin(2 * np.pi * (FS * bin_k / n) * t + phase)
    _worker['n'] = n
    _worker['bin'] = bin_k
    _worker['window'] = np.hanning(n)
    _worker['inband'] = np.arange(3, n // (2 * OSR) + 1)


def _grid_at_amp(amp):
    """One full (a1, a2) grid at this drive level for the worker's tone.
    Returns (amp, avg_a1, avg_a2, avg_sndr, n_stable) over the TOP_K cells."""
    x = _worker['x']
    n = _worker['n']
    bin_k = _worker['bin']
    window = _worker['window']
    inband = _worker['inband']
    xscale = (amp * VREF / 2) * afe_gain()
    n_a = len(A_VALS)
    grid = np.full((n_a, n_a), np.nan)
    sig = np.arange(bin_k - 2, bin_k + 3)
    noise = inband[~np.isin(inband, sig)]
    for i in range(n_a):
        for j in range(n_a):
            out, _, _, div = run_loop(x, xscale, THRESHOLDS,
                                      A_VALS[i], A_VALS[j], B1, B2, SLICES,
                                      BREAK_START, BREAK_END, DAC_STEP,
                                      LEVELS, n, W1, W2, MODE)
            if div < 0:
                yf = np.fft.rfft(out * window)
                power = np.abs(yf[:n // 2]) ** 2
                grid[i, j] = 10 * np.log10(power[sig].sum() / power[noise].sum())
    n_stable = int(np.isfinite(grid).sum())
    if n_stable < TOP_K:
        return amp, np.nan, np.nan, np.nan, n_stable
    flat = np.argsort(np.nan_to_num(grid, nan=-1e9).ravel())[::-1][:TOP_K]
    ii, jj = np.unravel_index(flat, grid.shape)
    return (amp, float(A_VALS[ii].mean()), float(A_VALS[jj].mean()),
            float(grid[ii, jj].mean()), n_stable)


def _draw_frame(fig, curve, f_req, f_real, n):
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(curve[:, 0], curve[:, 1], 'o-', markersize=3, label='avg a1 of top 5')
    ax.plot(curve[:, 0], curve[:, 2], 's-', markersize=3, label='avg a2 of top 5')
    ax.axhline(BOARD_A1, color='C0', linestyle='--', alpha=0.4, label='as-built a1')
    ax.axhline(BOARD_A2, color='C1', linestyle='--', alpha=0.4, label='as-built a2')
    d_a1, d_a2 = design_coeffs(FS)
    ax.axhline(d_a1, color='C0', linestyle='-', alpha=0.6, label='design a1')
    ax.axhline(d_a2, color='C1', linestyle='-', alpha=0.6, label='design a2')
    ax.set_xlabel('amp (jack Vpp / 2.5 V)')
    ax.set_ylabel('coefficient value')
    ax.set_ylim(A_VALS[0], A_VALS[-1])
    ax.set_title(f'requested {f_req} Hz -> simulated {f_real:.1f} Hz '
                 f'(N = {n} clocks = {n / FS * 1e3:.1f} ms)')
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(curve[:, 0], curve[:, 3], ':', color='gray', label='avg SNDR (right)')
    ax2.set_ylabel('SNDR of top 5 (dB)')
    ax2.set_ylim(40, 105)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = here / "results"
    results.mkdir(exist_ok=True)

    plan = [resolve_tone(f) for f in TONES_HZ]
    base_cost = len(AMPS) * len(A_VALS) ** 2
    print(f"{'tone':>8} {'simulated':>10} {'bin':>5} {'N':>8} {'workers':>8} {'rel cost':>9}")
    for f_req, (n, b, f_real) in zip(TONES_HZ, plan):
        w = workers_for(n)
        rel = (n / 2**16) * (WORKERS / w)
        print(f"{f_req:>8} {f_real:>9.1f} {b:>5} {n:>8} {w:>8} {rel:>8.1f}x")

    phase_rng = np.random.default_rng(SEED + 1)
    curves = np.full((len(TONES_HZ), len(AMPS), 5), np.nan)
    t0 = time.time()
    for k, (f_req, (n, bin_k, f_real)) in enumerate(zip(TONES_HZ, plan)):
        phase = float(phase_rng.uniform(0, 2 * np.pi))
        w = workers_for(n)
        with ProcessPoolExecutor(max_workers=w, initializer=_init_worker,
                                 initargs=(n, bin_k, phase)) as pool:
            futures = {pool.submit(_grid_at_amp, amp): a
                       for a, amp in enumerate(AMPS)}
            done = 0
            for fut in as_completed(futures):
                curves[k, futures[fut]] = fut.result()
                done += 1
                print(f"\rtone {k + 1}/{len(TONES_HZ)} ({f_real:.0f} Hz): "
                      f"{done}/{len(AMPS)} drive levels, "
                      f"elapsed {(time.time() - t0) / 60:.1f} min ",
                      end="", flush=True)
        print()
        np.savez(results / "best_coeffs_vs_amp.npz",
                 tones_requested=np.array(TONES_HZ),
                 tones_realized=np.array([p[2] for p in plan]),
                 amps=AMPS, a_vals=A_VALS, curves=curves,
                 board=np.array([BOARD_A1, BOARD_A2, B1, B2]))

    fig = plt.figure(figsize=(10, 6))
    writer = PillowWriter(fps=1 / SECONDS_PER_FRAME)
    gif_path = results / "best_coeffs_vs_amp.gif"
    with writer.saving(fig, str(gif_path), dpi=100):
        for k, (f_req, (n, bin_k, f_real)) in enumerate(zip(TONES_HZ, plan)):
            _draw_frame(fig, curves[k], f_req, f_real, n)
            writer.grab_frame()
    plt.close(fig)
    print(f"saved {gif_path}")
