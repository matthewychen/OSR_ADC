"""(a1, a2, amp) -> SNDR, swept over a grid and animated over amplitude.

Loop physics imported from noise_shp_sanity.run_loop (the one canonical
model: mid-rise 7-comparator quantizer with mismatched ladder taps, 7-element
DAC with per-element mismatch and static/DWA/rotate selection, BBM DAC
timeline 36+10+12 ns). Unstable points return NaN and render black.

Deviations from the parent file, for sweep runtime:
  - slices_per_period defaults to 50 (parent uses more). Cross-checked: same
    SNDR within 0.4 dB, and the BBM edges land on-grid either way.
  - board draw is seeded so every grid point sees the same board.
  - the two-tone stimulus is built once per amplitude frame, not per point.

Grid: a1, a2 in 0.40..1.10 step 0.05 (15 x 15), one GIF frame per amplitude
0.10..0.60 step 0.01 (51 frames) = 11475 runs, numba-compiled inner loop.
After the sweep, prints the best (amp, a1, a2) over the whole volume.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from noise_shp_sanity import run_loop, draw_board, board_coeffs, afe_gain

WORKERS = min(8, max(1, (os.cpu_count() or 4) - 2))


def two_tone(amp, *, fs=10e6, N=2**16, fin_bin=73, fin2_bin=173,
             vref=2.5, slices_per_period=50):
    """Stimulus on the fine grid, node-referred.
    amp is defined AT THE JACK: the sum's Vpp = amp * vref (amp = 1 swings
    0 to 2.5 V). The AFE divider (afe_gain, 0.435) scales it to the node."""
    dt = 1 / fs / slices_per_period
    t = np.arange(N * slices_per_period) * dt
    x1 = np.sin(2 * np.pi * (fs * fin_bin / N) * t)
    x2 = np.sin(2 * np.pi * (fs * fin2_bin / N) * t)
    node_peak = (amp * vref / 2) * afe_gain()
    return (x1 + x2) * (node_peak / np.max(np.abs(x1 + x2)))


def run_sndr(amp, a1, a2, *,
             fs=10e6, N=2**16, fin_bin=73, fin2_bin=173,
             vref=2.5, OSR=128, bits=3,
             slices_per_period=50,
             t_prop=36e-9, t_old=10e-9, t_break=12e-9,
             ladder_tol=0.001, fb_tol=0.001, mode=1, seed=1, x=None,
             b1=None, b2=None):
    """One modulator run. Returns SNDR in dB, or NaN if the loop diverges.
    mode: 0 = static element selection, 1 = DWA, 2 = rotate.
    b1/b2 default to the board's input/interstage referrals, so `amp` is a
    PHYSICAL drive: peak volts at the input node = amp * vref."""
    levels = 2**bits
    dac_step = vref / (levels - 1)
    if b1 is None or b2 is None:
        _, _, bb1, bb2 = board_coeffs(fs)
        b1 = bb1 if b1 is None else b1
        b2 = bb2 if b2 is None else b2

    thresholds, w1, w2 = draw_board(np.random.default_rng(seed), vref, levels,
                                    ladder_tol, fb_tol)

    dt = 1 / fs / slices_per_period
    break_start = int(round((t_prop + t_old) / dt))
    break_end = int(round((t_prop + t_old + t_break) / dt))
    assert break_end < slices_per_period, "DAC not settled before the next decision"

    if x is None:
        x = two_tone(amp, fs=fs, N=N, fin_bin=fin_bin, fin2_bin=fin2_bin,
                     vref=vref, slices_per_period=slices_per_period)

    output, _, _, diverged_at = run_loop(x, 1.0, thresholds, a1, a2, b1, b2,
                                         slices_per_period, break_start,
                                         break_end, dac_step, levels, N,
                                         w1, w2, mode)
    if diverged_at >= 0:
        return float('nan')

    window = np.hanning(N)
    yf = np.fft.fft(output * window)
    power = np.abs(yf[:N // 2]) ** 2
    band_edge_bin = N // (2 * OSR)
    inband = np.arange(3, band_edge_bin + 1)
    sig_bins = np.concatenate([np.arange(fin_bin - 2, fin_bin + 3),
                               np.arange(fin2_bin - 2, fin2_bin + 3)])
    noise_bins = inband[~np.isin(inband, sig_bins)]
    return 10 * np.log10(power[sig_bins].sum() / power[noise_bins].sum())


def _frame_at_amp(args):
    """One whole (a1, a2) grid at one drive level. Runs in a worker process;
    builds its own stimulus for this amp."""
    ai, amp, a_vals = args
    x = two_tone(amp)
    n_a = len(a_vals)
    frame = np.full((n_a, n_a), np.nan)
    for i in range(n_a):
        for j in range(n_a):
            frame[i, j] = run_sndr(amp, a_vals[i], a_vals[j], x=x)
    return ai, frame


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    results = here / "results"                             # all outputs land here, regardless of cwd
    results.mkdir(exist_ok=True)
    frames_per_graph = 3          # GIF playback: amplitude frames shown per second (keep <= 25)
    a_vals = np.round(np.arange(0.40, 1.101, 0.005), 3)   # 141 points, both axes
    amps = np.round(np.arange(0.50, 3.001, 0.05), 2)      # 51 frames; same physical
                                                          # range as the old 0.10-0.60
                                                          # node-referred sweep (x4.6)
    grid = np.full((len(amps), len(a_vals), len(a_vals)), np.nan)

    print(f"{len(a_vals)}x{len(a_vals)} grid x {len(amps)} frames = {grid.size} "
          f"runs on {WORKERS} workers")
    frames_done = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_frame_at_amp, (ai, amp, a_vals)): ai
                   for ai, amp in enumerate(amps)}
        for fut in as_completed(futures):
            ai, frame = fut.result()
            grid[ai] = frame
            frames_done += 1
            el = time.time() - t0
            eta = el / frames_done * (len(amps) - frames_done)
            filled = int(30 * frames_done / len(amps))
            print(f"\r[{'=' * filled}{' ' * (30 - filled)}] "
                  f"{frames_done}/{len(amps)} frames  "
                  f"elapsed {el / 60:4.1f} min  eta {eta / 60:4.1f} min ",
                  end="", flush=True)
            if frames_done % 5 == 0:
                np.save(results / "sndr_grid_partial.npy", grid)
    print()
    np.save(results / "sndr_grid.npy", grid)

    # best operating point over the whole (amp, a1, a2) volume
    bi = np.unravel_index(np.nanargmax(grid), grid.shape)
    _, _, bb1, bb2 = board_coeffs()
    print(f"best: SNDR = {grid[bi]:.1f} dB at amp = {amps[bi[0]]:.2f}, "
          f"a1 = {a_vals[bi[1]]:.3f}, a2 = {a_vals[bi[2]]:.3f} "
          f"(rho1 = {a_vals[bi[1]] / bb1:.3f}, rho2 = {a_vals[bi[2]] / bb2:.3f})")

    # --- GIF: one frame per amplitude, colour = SNDR, black = unstable ---
    finite = grid[np.isfinite(grid)]
    vmin, vmax = finite.min(), finite.max()
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')

    half = 0.025
    fig, ax = plt.subplots(figsize=(8, 6))
    _, _, B1, B2 = board_coeffs()
    im = ax.imshow(np.ma.masked_invalid(grid[0]), origin='lower', aspect='auto',
                   extent=[(a_vals[0] - half) / B2, (a_vals[-1] + half) / B2,
                           (a_vals[0] - half) / B1, (a_vals[-1] + half) / B1],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label='SNDR (dB)   [black = unstable]')
    ax.set_xlabel('rho2 = R19 / (Rbank2/7)')
    ax.set_ylabel('rho1 = R11 / (Rbank1/7)')
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

    ani = animation.FuncAnimation(fig, update, frames=len(amps), interval=300)
    gif_path = results / 'sndr_grid.gif'
    ani.save(str(gif_path), writer='pillow', fps=frames_per_graph)
    plt.close()
    print(f"saved {gif_path}")
