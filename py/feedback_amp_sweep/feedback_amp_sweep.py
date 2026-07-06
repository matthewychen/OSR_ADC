import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Major Assumptions
# 1. infinite opamp slew rate
# 2. no thermal/Johnson noise


fs = 20e6               # Sampling frequency (20 MHz)
N = 2**14               # Number of samples (16384)

# Coherent sampling: Choose an odd prime number of cycles
M = 11                  
fin = M * fs / N        # Coherent input frequency (~13.43 kHz)

vref = 2.5              # Reference voltage
OSR = 128               # Oversampling ratio
bits = 3                
levels = 2**bits - 1    # 7 levels, 3bit flash ADC
resistor_mismatch_coeff = 0.002

t = np.arange(N) / fs
window = np.hanning(N)

# A 7-level mid-tread DAC uses 6 unit resistor segments
num_resistors = levels - 1
resistors = np.random.uniform(1 - resistor_mismatch_coeff, 1 + resistor_mismatch_coeff, num_resistors)

# Define the amplitude sweep range from 0.1 to 1.0 in 0.05 steps (19 frames)
amps = np.linspace(0.1, 1.0, 19)
heatmaps = []

print("--- Starting Modulator Simulation Sweep ---")

for amp_idx, amp in enumerate(amps):
    print(f"\nComputing frame {amp_idx + 1}/{len(amps)} (Amplitude = {amp:.2f})")
    
    # Recalculate input stimulus for the current amplitude
    x = amp * vref * np.sin(2 * np.pi * fin * t)
    heatmap = np.zeros((20, 20))
    
    for a1_idx in range(1, 21):     
        progress = (a1_idx / 20) * 100
        sys.stdout.write(f"\rProgress: [{'=' * int(progress // 5):<20}] {progress:.1f}%")
        sys.stdout.flush()
        
        for a2_idx in range(1, 21):
            fb1 = (1/19) * a1_idx
            fb2 = (1/19) * a2_idx

            v1, v2, fb_volt = 0.0, 0.0, 0.0
            output = np.zeros(N)
            unstable = False

            for n in range(N):
                # 1. Integrator Stage
                v1 += fb1 * (x[n] - fb_volt)
                v2 += fb2 * (v1 - fb_volt)
                
                # 2. Quantizer (Symmetric 7-level Mid-Tread)
                q_index = np.round(v2 / (vref / 3))
                q_index = np.clip(q_index, -3, 3)
                
                # 3. Thermometer DAC Logic
                num_on = int(q_index + 3) 
                
                # Physical feedback calculation
                pos_side = np.sum(resistors[:num_on]) * vref
                neg_side = np.sum(resistors[num_on:]) * (-vref)
                fb_volt = (pos_side + neg_side) / float(num_resistors)
                
                # 4. Store the digital bitstream index
                output[n] = q_index
                
                # 5. Stability Check
                if abs(v1) > 100 or abs(v2) > 100:
                    unstable = True
                    break

            if unstable:
                # Penalize unstable points
                heatmap[a1_idx-1, a2_idx-1] = -10.0
                continue

            # Compute FFT on stable outputs
            windowed_output = output * window
            yf = np.fft.fft(windowed_output)

            # Calculate Power Spectral Density
            psd = 20 * np.log10((np.abs(yf[:N//2]) / (np.sum(window) / 2)) + 1e-20)
            
            # Calculate in-band SQNR
            bw_limit = int(N / (2 * OSR))
            
            # Find peak inside signal bandwidth, ignoring DC
            snrpeak = np.argmax(psd[2:bw_limit]) + 2
            
            sig_power = 0
            noise_power = 0
            
            for k in range(bw_limit):
                bin_power = np.abs(yf[k])**2
                if abs(k - snrpeak) <= 3:
                    sig_power += bin_power
                else:
                    noise_power += bin_power
            
            if sig_power > 0 and noise_power > 0:
                snqr = 10 * np.log10(sig_power / noise_power)
            else:
                snqr = -10.0
                
            heatmap[a1_idx-1, a2_idx-1] = snqr
            
    heatmaps.append(heatmap)

print("\n\nAll simulations complete. Generating GIF animation...")

# ---------------------------------------------------------
# Set up the Matplotlib Animation Plot
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Identify global scale limits to keep color coding consistent
all_vals = np.array(heatmaps)
vmin = all_vals.min()
vmax = all_vals.max()

# Create initial frame
im = ax.imshow(heatmaps[0], aspect='auto', origin='lower', 
                extent=[1/19, 20/19, 1/19, 20/19], cmap='magma', 
                vmin=vmin, vmax=vmax)

fig.colorbar(im, ax=ax, label='SQNR (dB)')
ax.set_xlabel('a2 Coefficient (Integrator 2)')
ax.set_ylabel('a1 Coefficient (Integrator 1)')
title_text = ax.set_title(f'2nd-Order Modulator Stability Map (Amp = {amps[0]:.2f})')

# Reference marker and legend
ax.plot(0.5, 0.65, 'wx', label='Standard Design Point')
ax.legend(loc='upper right')

# Update function for animation
def update(frame):
    im.set_array(heatmaps[frame])
    title_text.set_text(f'2nd-Order Modulator Stability Map (Amp = {amps[frame]:.2f})')
    return [im, title_text]

# Compile the animation (interval=333ms corresponds to 3 FPS)
ani = animation.FuncAnimation(fig, update, frames=len(amps), interval=333, blit=True)

# Save output
gif_filename = 'modulator_stability_sweep.gif'
ani.save(gif_filename, writer='pillow', fps=3)
plt.close()

print(f"Animation saved as '{gif_filename}'")