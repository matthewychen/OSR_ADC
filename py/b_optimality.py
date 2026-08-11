#finding the optimal values of b1, b2, corresponding to R11 R19
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from noise_shp_sanity import (run_loop, draw_board, board_coeffs, afe_gain,
                              design_coeffs)

WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))

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
MODE = 1
SEED = 1

B1_VALS = np.round(np.arange(0.20, 1.4001, 0.10), 2)   # coarse: expect flat
B2_VALS = np.round(np.arange(0.20, 1.6001, 0.02), 2)   # fine: expect a peak
AMP_LADDER = np.round(np.geomspace(0.15, 3.5, 18), 3)  # each b picks its best

# Banks under test: the chosen design point (bank1 = 14.0k -> a1 = 0.500,
# bank2 = 7.32k -> a2 = 0.956). To test the as-built board instead, set
# these to BOARD_A1, BOARD_A2 further down.
A1_TEST, A2_TEST = design_coeffs(FS)

LEVELS = 2**BITS
DAC_STEP = VREF / (LEVELS - 1)
DT = 1 / FS / SLICES
BREAK_START = int(round((T_PROP + T_OLD) / DT))
BREAK_END = int(round((T_PROP + T_OLD + T_BREAK) / DT))
assert BREAK_END < SLICES

THRESHOLDS, W1, W2 = draw_board(np.random.default_rng(SEED), VREF, LEVELS,
                                LADDER_TOL, FB_TOL)
WINDOW = np.hanning(N)
INBAND = np.arange(3, N // (2 * OSR) + 1)
BOARD_A1, BOARD_A2, B1, B2 = board_coeffs(FS)

_tone_rng = np.random.default_rng(SEED + 1)
TONE_BIN = int(_tone_rng.choice(np.arange(5, 255)))
TONE_PHASE = float(_tone_rng.uniform(0, 2 * np.pi))


def sndr_of(output, bin_k):
    yf = np.fft.rfft(output * WINDOW)
    power = np.abs(yf[:N // 2]) ** 2
    sig = np.arange(bin_k - 2, bin_k + 3)
    noise = INBAND[~np.isin(INBAND, sig)]
    return 10 * np.log10(power[sig].sum() / power[noise].sum())


_worker = {}


def _init_worker():
    t = np.arange(N * SLICES) * DT
    _worker['x'] = np.sin(2 * np.pi * (FS * TONE_BIN / N) * t + TONE_PHASE)


def _peak_over_amps(args):
    """Best SNDR this (b1, b2) can reach, searching the drive ladder.
    Runs in a worker. Returns (index, peak_sndr, best_amp, n_stable)."""
    idx, b1, b2 = args
    x = _worker['x']
    peak = -np.inf
    best_amp = np.nan
    n_stable = 0
    for amp in AMP_LADDER:
        xscale = (amp * VREF / 2) * afe_gain()
        out, _, _, div = run_loop(x, xscale, THRESHOLDS,
                                  A1_TEST, A2_TEST, b1, b2, SLICES,
                                  BREAK_START, BREAK_END, DAC_STEP,
                                  LEVELS, N, W1, W2, MODE)
        if div < 0:
            n_stable += 1
            s = sndr_of(out, TONE_BIN)
            if s > peak:
                peak = s
                best_amp = amp
    if not np.isfinite(peak):
        return idx, np.nan, np.nan, 0
    return idx, peak, best_amp, n_stable


def _sweep(tag, b_pairs, pool):
    res = np.full((len(b_pairs), 4), np.nan)
    futures = {pool.submit(_peak_over_amps, (k, b1, b2)): k
               for k, (b1, b2) in enumerate(b_pairs)}
    t0 = time.time()
    done = 0
    for fut in as_completed(futures):
        idx, peak, best_amp, ns = fut.result()
        res[idx] = (b_pairs[idx][0] if tag == 'b1' else b_pairs[idx][1],
                    peak, best_amp, ns)
        done += 1
        print(f"\r{tag}: {done}/{len(b_pairs)}  elapsed {time.time() - t0:4.0f} s ",
              end="", flush=True)
    print()
    return res


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = here / "results"
    results.mkdir(exist_ok=True)
    print(f"tone bin {TONE_BIN} = {FS * TONE_BIN / N / 1e3:.2f} kHz; "
          f"banks fixed at a1={A1_TEST:.3f}, a2={A2_TEST:.3f} "
          f"(board: {BOARD_A1:.3f}, {BOARD_A2:.3f}); "
          f"{len(AMP_LADDER)} drive levels per b value; {WORKERS} workers")

    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init_worker) as pool:
        res_b1 = _sweep('b1', [(b, B2) for b in B1_VALS], pool)
        res_b2 = _sweep('b2', [(B1, b) for b in B2_VALS], pool)
    np.save(results / "b_optimality_b1.npy", res_b1)
    np.save(results / "b_optimality_b2.npy", res_b2)

    # verdicts
    spread_b1 = np.nanmax(res_b1[:, 1]) - np.nanmin(res_b1[:, 1])
    k2 = np.nanargmax(res_b2[:, 1])
    best_b2 = res_b2[k2, 0]
    at_board = res_b2[np.argmin(np.abs(res_b2[:, 0] - B2)), 1]
    print(f"\nb1: peak-SNDR spread across {B1_VALS[0]}..{B1_VALS[-1]} = "
          f"{spread_b1:.1f} dB (flat = R11 is not a loop knob; pick it for "
          f"noise and input range)")
    print(f"b2: board = {B2:.3f} ({at_board:.1f} dB), best = {best_b2:.3f} "
          f"({res_b2[k2, 1]:.1f} dB), available gain = "
          f"{res_b2[k2, 1] - at_board:+.1f} dB")
    print(f"    best b2 as R19 (C2 = 100 pF): "
          f"{1 / FS / (best_b2 * 100e-12):.0f} ohms (board: 1620)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.plot(res_b2[:, 0], res_b2[:, 1], 'o-', markersize=3, label='peak SNDR')
    ax.axvline(B2, color='r', linestyle='--', label=f'board b2 = {B2:.2f} (R19 = 1.62k)')
    ax.set_xlabel('b2 = T / (R19 * C2)')
    ax.set_ylabel('peak SNDR over drive ladder (dB)')
    ax.set_title(f'b2 sweep at a1={A1_TEST}, a2={A2_TEST} (candidate banks)')
    ax.grid(True)
    ax.legend()

    ax = axes[1]
    ax.plot(res_b1[:, 0], res_b1[:, 1], 'o-', markersize=4, label='peak SNDR')
    ax.axvline(B1, color='r', linestyle='--', label=f'board b1 = {B1:.2f} (R11 = 1.43k)')
    ax.set_xlabel('b1 = T / (R11 * C1)')
    ax.set_ylabel('peak SNDR over drive ladder (dB)')
    ax.set_title('b1 should be flat (pure scale); best drive moves as 1/b1')
    ax.grid(True)
    ax2 = ax.twinx()
    ax2.plot(res_b1[:, 0], res_b1[:, 2], ':', color='gray', label='best amp (right)')
    ax2.set_ylabel('best amp (jack Vpp / 2.5 V)')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.savefig(results / "b_optimality.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {results / 'b_optimality.png'}")
