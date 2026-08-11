"""Simulation core for the HOERNERV 2nd-order continuous-time delta-sigma ADC.

Every sweep script in this folder imports its physics from here (run_loop).

Voltage frame: stored value = board value - 1.25 V (the summing-node bias).
Clock period (100 ns): t = 0 comparators latch (the only sampling event);
t = 46 ns changed switches open (break-before-make); t = 58 ns they land;
the value holds into the next period (excess loop delay). Integration is
continuous, approximated by slices_per_period equal slices.

Coefficients are T/RC charge gains: b1 = T/(R11*C1), a1 = T/((Rbank1/7)*C1),
b2 = T/(R19*C2), a2 = T/((Rbank2/7)*C2). Feedback carries a minus sign; the
board must match it with an odd inversion count, comparator -> switch rail.

Idealized: op-amps and comparators perfect, no thermal noise, no jitter.
Divergence declared at |v| >= 100 V.
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

fs = 10e6               # clock rate, Hz
N = 2**16               # clock periods per run (6.55 ms at 10 MHz)
fin_bin = 73            # test tone 1: integer cycles per record -> coherent
fin = fs * fin_bin / N  # 11.139 kHz
fin2_bin = 173          # test tone 2; both prime, so harmonics/IMD land on distinct bins
fin2 = fs * fin2_bin / N  # 26.398 kHz
amp = 1.29              # jack drive: sine Vpp = amp * vref. 1.29 = design
                        # point (3.2 Vpp jack -> 0.70 V node peak via afe_gain).
vref = 2.5              # reference voltage: ladder top, DAC high rail
OSR = 128               # oversampling ratio -> audio band edge = fs/(2*OSR)
bits = 3
levels = 2**bits        # 8 DAC levels from 7 comparators / 7 elements (mid-rise)
step = vref / levels    # 312.5 mV: ladder tap spacing (8 equal segments)
dac_step = vref / (levels - 1)  # 357 mV: DAC change when one element switches rail
ladder_tol = 0.001      # ladder resistor tolerance, uniform +/-
fb_tol = 0.001          # feedback element tolerance, uniform +/-
ref_tol = 3e-4          # virtual ground = ladder mid-tap: within 0.03%
                        # (0.375 mV) of the rail midpoint for 0.1% parts. 0 = off.
dwa_mode = 1            # element selection: 0 static, 1 DWA (data-weighted averaging), 2 rotate


def draw_board(rng, vref=2.5, levels=8, ladder_tol=0.001, fb_tol=0.001):
    """Draw one board: (thresholds, w1, w2) = the 7 node-referred ladder
    taps (ascending) and the per-element weights of the two feedback
    banks (1.0 = nominal; banks drawn independently)."""
    r_ladder = 1 + ladder_tol * (2 * rng.random(levels) - 1)
    thresholds = vref * np.cumsum(r_ladder[:-1]) / np.sum(r_ladder) - vref / 2
    w1 = 1 + fb_tol * (2 * rng.random(levels - 1) - 1)
    w2 = 1 + fb_tol * (2 * rng.random(levels - 1) - 1)
    return thresholds, w1, w2


rng = np.random.default_rng()
thresholds, w1, w2 = draw_board(rng, vref, levels, ladder_tol, fb_tol)
vg_err = ref_tol * (vref / 2) * (2 * rng.random() - 1)   # this run's reference error, volts

# board values, from the schematic (order rev 2026-08-09)
R_FB1_UNIT = 14.0e3     # feedback bank 1, 7 equal elements (R12-R18)
R_FB2_UNIT = 7.32e3     # feedback bank 2, 7 equal elements (R21-R27)
R_IN = 1.5e3            # R11: signal input into integrator 1
R_INTER = 1.5e3         # R19: integrator 1 output into integrator 2
C_INT1 = 100e-12        # C25
C_INT2 = 100e-12        # C28

# AFE input stage, from the schematic: jack -> R3/R4 -> biased node -> buffer
R_SRC = 13e3            # R3 and R4, one series resistor per jack channel
R_BIAS_TOP = 10e3       # R5: node up to +2.5 V
R_BIAS_BOT = 10e3       # R6: node down to ground


def afe_gain():
    """Jack-to-node voltage gain (0.435): R3||R4 = 6.5k into R5||R6 = 5k."""
    r_series = R_SRC / 2
    r_load = R_BIAS_TOP * R_BIAS_BOT / (R_BIAS_TOP + R_BIAS_BOT)
    return r_load / (r_load + r_series)


def board_coeffs(fs=10e6):
    """The four T/RC gains the board realizes: (a1, a2, b1, b2)."""
    T = 1 / fs
    a1 = T / ((R_FB1_UNIT / 7) * C_INT1)
    a2 = T / ((R_FB2_UNIT / 7) * C_INT2)
    b1 = T / (R_IN * C_INT1)
    b2 = T / (R_INTER * C_INT2)
    return a1, a2, b1, b2


# bank values adopted in the order rev (sweep scripts import design_coeffs
# for their DESIGN markers; board_coeffs now realizes the same numbers)
R_FB1_DESIGN = 14.0e3
R_FB2_DESIGN = 7.32e3


def design_coeffs(fs=10e6):
    """(a1, a2) the ordered banks realize: (0.500, 0.956) at 10 MHz."""
    T = 1 / fs
    return (T / ((R_FB1_DESIGN / 7) * C_INT1),
            T / ((R_FB2_DESIGN / 7) * C_INT2))


def coeff_ratios(a1, a2, b1, b2):
    """(rho1, rho2) = (a1/b1, a2/b2): pure resistor ratios. rho1 sets the
    input range (stability floor: rho1 > afe_gain() * amp); rho2 is the
    same budget for stage 2."""
    return a1 / b1, a2 / b2


def size_resistors(rho1, rho2, a1, a2, fs=10e6, c1=C_INT1, c2=C_INT2):
    """Map an operating point (rho1, rho2, a1, a2) to resistor values.
    Returns (R11, R19, R_bank1_unit, R_bank2_unit) in ohms."""
    T = 1 / fs
    r_bank1 = 7 * T / (a1 * c1)
    r_bank2 = 7 * T / (a2 * c2)
    r11 = rho1 * r_bank1 / 7
    r19 = rho2 * r_bank2 / 7
    return r11, r19, r_bank1, r_bank2


slices_per_period = 100
dt = 1 / fs / slices_per_period

t_prop = 36e-9          # decision -> new data at the switch control pins
t_old = 10e-9           # switch still on the old rail after the data arrives
t_break = 12e-9         # changed switches float (neither rail)

break_start = int(round((t_prop + t_old) / dt))
break_end = int(round((t_prop + t_old + t_break) / dt))
assert break_end < slices_per_period, "DAC not settled before the next decision: model needs cross-period events"

# Operating point for this file's standalone run: the ordered board.
# board_coeffs and design_coeffs now agree (14.0k / 7.32k -> a1 = 0.500,
# a2 = 0.956; R11 = R19 = 1.5k -> b1 = b2 = 0.667, rho1 = 0.750, rho2 = 1.434).
a1, a2 = design_coeffs(fs)
b1 = board_coeffs(fs)[2]
b2 = board_coeffs(fs)[3]


@njit(cache=True)
def run_loop(x, xscale, thresholds, a1, a2, b1, b2, slices_per_period,
             break_start, break_end, dac_step, levels, n_samp, w1, w2, mode,
             vg_err=0.0):
    """Run the modulator for n_samp clock periods.

    x: input on the fine grid (slices_per_period samples per clock),
    scaled by xscale. mode: 0 static, 1 DWA, 2 rotate. vg_err (volts):
    shifts every ladder tap by -err and offsets both DAC banks by ~-err.

    Returns (output, v1_trace, v2_trace, diverged_at). output[n] is the
    ideal volts of code n (the FPGA's number; the mismatched analog value
    drives the loop). diverged_at: -1 if stable, else the failing period.
    """
    n_elem = levels - 1
    per_element = dac_step / 2.0    # one element's contribution about midscale
    w_total1 = np.sum(w1)
    w_total2 = np.sum(w2)

    v1 = 0.0
    v2 = 0.0

    element_sign = np.full(n_elem, -1.0)    # +1: on the 2.5 V rail, -1: on 0 V
    chosen_sign = np.empty(n_elem)
    rotation = 0

    dac1_wire = 0.0     # value driving integrator 1 right now.
    dac2_wire = 0.0     # start-up: wires read midscale until the first
                        # decision's events, even though the element
                        # bookkeeping starts all-low; the mismatch lasts
                        # only the first 46 ns and touches nothing after
    dac1_landed = 0.0   # value once the changed switches land (from 58 ns)
    dac2_landed = 0.0
    dac1_open = 0.0     # value while the changed switches float (46-58 ns)
    dac2_open = 0.0

    output = np.zeros(n_samp)
    v1_trace = np.zeros(n_samp)
    v2_trace = np.zeros(n_samp)

    for n in range(n_samp):
        for s in range(slices_per_period):

            if s == 0:
                v1_trace[n] = v1
                v2_trace[n] = v2
                n_high = np.searchsorted(thresholds, v2 + vg_err)
                output[n] = (n_high - n_elem / 2.0) * dac_step

                for e in range(n_elem):
                    chosen_sign[e] = -1.0
                if mode == 0:
                    for e in range(n_high):
                        chosen_sign[e] = 1.0
                else:
                    for e in range(n_high):
                        chosen_sign[(rotation + e) % n_elem] = 1.0
                    if mode == 1:
                        rotation = (rotation + n_high) % n_elem
                    else:
                        rotation = (rotation + 1) % n_elem

                dac1_landed = 0.0
                dac2_landed = 0.0
                dac1_open = 0.0
                dac2_open = 0.0
                w1_open = 0.0
                w2_open = 0.0
                for e in range(n_elem):
                    dac1_landed += chosen_sign[e] * w1[e]
                    dac2_landed += chosen_sign[e] * w2[e]
                    if chosen_sign[e] == element_sign[e]:
                        dac1_open += chosen_sign[e] * w1[e]
                        dac2_open += chosen_sign[e] * w2[e]
                        w1_open += w1[e]
                        w2_open += w2[e]
                    element_sign[e] = chosen_sign[e]
                # conducting elements each span (1.25 -/+ vg_err) of rail:
                # the sign part scales per_element, the vg_err part adds an
                # offset of -vg_err * (conducting weight / 7)
                dac1_landed = dac1_landed * per_element - vg_err * w_total1 / n_elem
                dac2_landed = dac2_landed * per_element - vg_err * w_total2 / n_elem
                dac1_open = dac1_open * per_element - vg_err * w1_open / n_elem
                dac2_open = dac2_open * per_element - vg_err * w2_open / n_elem

            if s == break_start:
                dac1_wire = dac1_open
                dac2_wire = dac2_open
            if s == break_end:
                dac1_wire = dac1_landed
                dac2_wire = dac2_landed

            v1 = v1 + (b1 * xscale * x[n * slices_per_period + s]
                       - a1 * dac1_wire) / slices_per_period
            v2 = v2 + (b2 * v1 - a2 * dac2_wire) / slices_per_period

        if not (abs(v1) < 100.0 and abs(v2) < 100.0):
            return output, v1_trace, v2_trace, n
    return output, v1_trace, v2_trace, -1

if __name__ == "__main__":
    t = np.arange(N * slices_per_period) * dt
    tone1 = np.sin(2 * np.pi * fin * t)
    tone2 = np.sin(2 * np.pi * fin2 * t)
    jack_peak = amp * vref / 2                  # Vpp = amp * vref
    node_peak = jack_peak * afe_gain()          # through the input divider
    x = (tone1 + tone2) * (node_peak / np.max(np.abs(tone1 + tone2)))

    output, v1_trace, v2_trace, diverged_at = run_loop(
        x, 1.0, thresholds, a1, a2, b1, b2, slices_per_period,
        break_start, break_end, dac_step, levels, N, w1, w2, dwa_mode, vg_err)
    print(f"reference this run: 1.25 V {vg_err * 1e3:+.1f} mV "
          f"-> output DC = {output.mean() * 1e3:+.1f} mV")
    if diverged_at >= 0:
        print(f"UNSTABLE: integrator state left +/-100 V at sample {diverged_at} "
              f"({diverged_at / fs * 1e3:.2f} ms) -- spectrum below is post-divergence garbage")

    window = np.hanning(N)
    yf = np.fft.fft(output * window)
    amplitude_db = 20 * np.log10((np.abs(yf[:N // 2]) / (N / 4)) + 1e-20)
    freqs = np.fft.fftfreq(N, 1 / fs)[:N // 2]

    power = np.abs(yf[:N // 2]) ** 2
    band_edge_bin = N // (2 * OSR)
    inband = np.arange(3, band_edge_bin + 1)        # bins 0-2 hold windowed DC
    sig_bins = np.concatenate([np.arange(fin_bin - 2, fin_bin + 3),
                               np.arange(fin2_bin - 2, fin2_bin + 3)])
    noise_bins = inband[~np.isin(inband, sig_bins)]
    p_sig = power[sig_bins].sum()
    p_nd = power[noise_bins].sum()
    sndr = 10 * np.log10(p_sig / p_nd)
    enob = (sndr - 1.76) / 6.02
    sfdr = 10 * np.log10(power[sig_bins].max() / power[noise_bins].max())
    mode_name = ('static', 'DWA', 'rotate')[dwa_mode]
    print(f"SNDR = {sndr:.1f} dB   ENOB = {enob:.2f} bits   SFDR = {sfdr:.1f} dB   "
          f"({mode_name}, tones {fin / 1e3:.3f} + {fin2 / 1e3:.3f} kHz, band {fs / (2 * OSR) / 1e3:.1f} kHz)")

    print(f"ladder taps with +/-{ladder_tol:.1%} resistor mismatch:")
    for j in range(levels - 1):
        nominal = (j + 1 - levels / 2) * step
        print(f"  comp {j+1}: {thresholds[j]:+.6f} V   nominal {nominal:+.6f} V   "
              f"err {(thresholds[j] - nominal) * 1e3:+.3f} mV")
    plt.figure(figsize=(10, 5))
    plt.semilogx(freqs, amplitude_db)
    plt.title(f"3-Bit 2nd-Order CT Modulator (OSR={OSR}, {slices_per_period} slices/period, "
              f"BBM DAC {(t_prop + t_old) * 1e9:.0f}+{t_break * 1e9:.0f} ns, {mode_name}), fs={fs / 1e6:.0f} MHz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid(True, which="both")
    plt.axvline(fs / (2 * OSR), color='r', linestyle='--', label='Signal Bandwidth')
    plt.axvline(fin, color='g', linestyle=':', label='Input Tones')
    plt.axvline(fin2, color='g', linestyle=':')
    plt.ylim([-160, 20])
    plt.legend()
    plt.savefig("dsm_ct.png", dpi=110, bbox_inches="tight")
    plt.show()