#Overnight regression

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from noise_shp_sanity import (run_loop, draw_board, board_coeffs, afe_gain,
                              design_coeffs)

# --- fixed configuration (loop physics lives in noise_shp_sanity.run_loop) ---
FS = 10e6
N = 2**16
FIN_BIN, FIN2_BIN = 73, 173
VREF = 2.5
OSR = 128
BITS = 3
SLICES = 50
T_PROP, T_OLD, T_BREAK = 36e-9, 10e-9, 12e-9
LADDER_TOL = 0.001
FB_TOL = 0.001
MODE = 1                # element selection: 0 = static, 1 = DWA, 2 = rotate
SEED = 1

LEVELS = 2**BITS
DAC_STEP = VREF / (LEVELS - 1)
DT = 1 / FS / SLICES
BREAK_START = int(round((T_PROP + T_OLD) / DT))
BREAK_END = int(round((T_PROP + T_OLD + T_BREAK) / DT))
assert BREAK_END < SLICES

THRESHOLDS, W1, W2 = draw_board(np.random.default_rng(SEED), VREF, LEVELS,
                                LADDER_TOL, FB_TOL)

# unit-amplitude two-tone, built once; scaled per run inside the loop
_t = np.arange(N * SLICES) * DT
X_UNIT = (np.sin(2 * np.pi * (FS * FIN_BIN / N) * _t)
          + np.sin(2 * np.pi * (FS * FIN2_BIN / N) * _t))
X_PEAK = np.max(np.abs(X_UNIT))
del _t

WINDOW = np.hanning(N)
_band_edge = N // (2 * OSR)
_inband = np.arange(3, _band_edge + 1)
SIG_BINS = np.concatenate([np.arange(FIN_BIN - 2, FIN_BIN + 3),
                           np.arange(FIN2_BIN - 2, FIN2_BIN + 3)])
NOISE_BINS = _inband[~np.isin(_inband, SIG_BINS)]

# board referral at this fs: a = feedback banks (R_unit/7), b = input/interstage
BOARD_A1, BOARD_A2, B1, B2 = board_coeffs(FS)
DESIGN_A1, DESIGN_A2 = design_coeffs(FS)


def sndr_point(amp, a1, a2):
    """SNDR in dB, NaN if the loop diverges. `amp` is defined AT THE JACK:
    stimulus Vpp = amp * VREF (amp = 1 swings 0 to 2.5 V); the AFE divider
    scales it to the node before the loop sees it."""
    xscale = (amp * VREF / 2) * afe_gain() / X_PEAK
    output, _, _, diverged = run_loop(X_UNIT, xscale, THRESHOLDS, a1, a2,
                                      B1, B2, SLICES, BREAK_START, BREAK_END,
                                      DAC_STEP, LEVELS, N, W1, W2, MODE)
    if diverged >= 0:
        return float('nan')
    yf = np.fft.fft(output * WINDOW)
    power = np.abs(yf[:N // 2]) ** 2
    return 10 * np.log10(power[SIG_BINS].sum() / power[NOISE_BINS].sum())


def msa_point(a1, a2, iters=12, lo=0.10, hi=3.20):
    """Bisect the stability boundary in amplitude.
    Returns (msa, sndr_at_msa). msa=0: unstable even at `lo`.
    msa=hi: never went unstable below the cap."""
    s = sndr_point(hi, a1, a2)
    if np.isfinite(s):
        return hi, s
    s_lo = sndr_point(lo, a1, a2)
    if not np.isfinite(s_lo):
        return 0.0, float('nan')
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = sndr_point(mid, a1, a2)
        if np.isfinite(s):
            lo, s_lo = mid, s
        else:
            hi = mid
    return lo, s_lo


def _bar(done, total, t0, label):
    el = time.time() - t0
    eta = el / done * (total - done) if done else 0
    filled = int(30 * done / total)
    print(f"\r[{'=' * filled}{' ' * (30 - filled)}] {label}  {done}/{total}  "
          f"elapsed {el / 60:5.1f} min  eta {eta / 60:5.1f} min ",
          end="", flush=True)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = here / "results"    # all outputs land here, regardless of cwd
    results.mkdir(exist_ok=True)
    frames_per_graph = 3          # GIF playback: amplitude frames per second

    # ---------------- Phase 1: MSA map ----------------
    a_fine = np.round(np.arange(0.2, 2.0001, 0.005), 3)      # 141 points
    n_f = len(a_fine)
    msa = np.full((n_f, n_f), np.nan)
    msa_sndr = np.full((n_f, n_f), np.nan)

    sndr_point(0.10, 0.5, 0.6)                    # warm-up: numba compile happens here, not in the estimate
    t_cal = time.time()
    msa_point(0.5, 0.6)
    per_point = time.time() - t_cal
    print(f"Phase 1: MSA map, {n_f}x{n_f} points, bisection to 0.0002 amp "
          f"-- estimated {n_f * n_f * per_point / 60:.0f} min")
    t0 = time.time()
    for i, a1 in enumerate(a_fine):
        for j, a2 in enumerate(a_fine):
            msa[i, j], msa_sndr[i, j] = msa_point(a1, a2)
        _bar((i + 1) * n_f, n_f * n_f, t0, "MSA")
        if i % 5 == 4:
            np.save(results / "overnight_msa_partial.npy", msa)
            np.save(results / "overnight_msa_sndr_partial.npy", msa_sndr)
    print()
    np.save(results / "overnight_msa.npy", msa)
    np.save(results / "overnight_msa_sndr.npy", msa_sndr)

    # MSA figure: boundary amplitude + SNDR at the boundary
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ext = [a_fine[0] / B2, a_fine[-1] / B2, a_fine[0] / B1, a_fine[-1] / B1]
    im0 = axes[0].imshow(msa, origin='lower', aspect='auto', extent=ext,
                         cmap='viridis')
    fig.colorbar(im0, ax=axes[0], label='MSA (amp units: jack Vpp / 2.5 V)')
    axes[0].set_title('Maximum stable amplitude')
    im1 = axes[1].imshow(msa_sndr, origin='lower', aspect='auto', extent=ext,
                         cmap='magma')
    fig.colorbar(im1, ax=axes[1], label='SNDR at MSA (dB)')
    axes[1].set_title('SNDR at the stability edge')
    for ax in axes:
        ax.set_xlabel('rho2 = R19 / (Rbank2/7)')
        ax.set_ylabel('rho1 = R11 / (Rbank1/7)')
        ax.plot(BOARD_A2 / B2, BOARD_A1 / B1, 'x', color='0.75',
                markersize=9, label='as-built (16.9k/13k)')
        ax.plot(DESIGN_A2 / B2, DESIGN_A1 / B1, '*', color='white',
                markersize=14, label='design (14.0k/7.32k)')
        ax.legend(loc='upper right')
        # coefficient values on the secondary axes (a = rho * b at this fs / C)
        sx = ax.secondary_xaxis('top', functions=(lambda r: r * B2, lambda a: a / B2))
        sx.set_xlabel('a2')
        sy = ax.secondary_yaxis('right', functions=(lambda r: r * B1, lambda a: a / B1))
        sy.set_ylabel('a1')
    fig.savefig(results / "overnight_msa.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {results / 'overnight_msa.png'}")

    # ---------------- Phase 2: SNDR grid GIF ----------------
    a_vals = np.round(np.arange(0.30, 1.1001, 0.01), 2)       
    amps = np.round(np.arange(0.30, 2.501, 0.01), 2)          
    grid = np.full((len(amps), len(a_vals), len(a_vals)), np.nan)

    total = grid.size
    t_cal = time.time()
    sndr_point(0.40, 0.5, 0.6)
    per_run = time.time() - t_cal
    print(f"Phase 2: SNDR grid, {len(a_vals)}x{len(a_vals)} x {len(amps)} frames "
          f"-- estimated {total * per_run / 60:.0f} min")
    done = 0
    t0 = time.time()
    for ai, amp in enumerate(amps):
        for i, a1 in enumerate(a_vals):
            for j, a2 in enumerate(a_vals):
                grid[ai, i, j] = sndr_point(amp, a1, a2)
                done += 1
            _bar(done, total, t0, f"amp={amp:.2f}")
        np.save(results / "overnight_grid_partial.npy", grid)
    print()
    np.save(results / "overnight_grid.npy", grid)

    finite = grid[np.isfinite(grid)]
    vmin, vmax = finite.min(), finite.max()
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')

    half = 0.005
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.ma.masked_invalid(grid[0]), origin='lower', aspect='auto',
                   extent=[(a_vals[0] - half) / B2, (a_vals[-1] + half) / B2,
                           (a_vals[0] - half) / B1, (a_vals[-1] + half) / B1],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='SNDR (dB)   [black = unstable]')
    ax.set_xlabel('rho2 = R19 / (Rbank2/7)')
    ax.set_ylabel('rho1 = R11 / (Rbank1/7)')
    ax.plot(BOARD_A2 / B2, BOARD_A1 / B1, 'x', color='0.75', markersize=9)
    ax.plot(DESIGN_A2 / B2, DESIGN_A1 / B1, '*', color='white', markersize=14)
    # coefficient values on the secondary axes (a = rho * b at this fs / C)
    sx = ax.secondary_xaxis('top', functions=(lambda r: r * B2, lambda a: a / B2))
    sx.set_xlabel('a2')
    sy = ax.secondary_yaxis('right', functions=(lambda r: r * B1, lambda a: a / B1))
    sy.set_ylabel('a1')
    title = ax.set_title(f'SNDR map, amp = {amps[0]:.2f}')

    def update(fr):
        im.set_array(np.ma.masked_invalid(grid[fr]))
        title.set_text(f'SNDR map, amp = {amps[fr]:.2f}')
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(amps), interval=500)
    gif_path = results / 'overnight_grid.gif'
    ani.save(str(gif_path), writer='pillow', fps=frames_per_graph)
    plt.close()
    print(f"saved {gif_path}")
