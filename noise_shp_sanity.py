import numpy as np
import matplotlib.pyplot as plt

fs = 20e6               # sampling freq (20 MHz)
N = 2**16               # num of samples
fin = 11.3e3            # input sine frequency
amp = 0.5               # input amplitude (% of reference voltage); this is voltage to sweep
vref = 2.5              # reference voltage (used in Flash ADC comparisons and in feedback loop stable voltage source with low sag)
OSR = 128               # oversampling ratio
bits = 3                # n-bit flash ADC
levels = 2**bits - 1    # 7 levels (thermometer code)

t = np.arange(N) / fs
x = amp * vref * np.sin(2 * np.pi * fin * t) #stimulus sin wave

a1 = 0.5
a2 = 0.5

# for a1 in range(0.5, 0.5):
#     for a2 in range(0.5, 0.5):
#success=False

v1 = 0.0           # Integrator 1 state
v2 = 0.0           # Integrator 2 state
feedback = 0.0     # Initial feedback is 0
output = np.zeros(N)

for n in range(N):
    # Integrator 1: Error = Input - Feedback
    v1 = v1 + a1 * (x[n] - feedback)
    
    # Integrator 2: Error = Output of Int1 - Feedback
    v2 = v2 + a2 * (v1 - feedback)
    # Normalize v2 relative to vref
    normalized_v2 = v2 / vref
    
    # Round to the nearest of the 8 levels
    # (levels/2) scales the range -1 to 1 into -3.5 to 3.5
    q_index = np.round(normalized_v2 * (levels / 2))
    
    # Clip the result to stay within the 3-bit range
    q_index = np.clip(q_index, -(levels/2), (levels/2))
    
    # 4. Convert back to a voltage for the feedback loop
    feedback = (q_index / (levels / 2)) * vref
    
    # Store the result for the next stage
    output[n] = feedback
    
    window = np.hanning(N)
    windowed_output = output * window

    # 2. Compute FFT
    yf = np.fft.fft(windowed_output)

    # 3. Calculate Power Spectral Density in dB
    # We normalize by (N/4) to account for the window and the double-sided FFT
    psd = 20 * np.log10((np.abs(yf[:N//2]) / (N / 4)) + 1e-20)

    # 4. Frequency axis
    freqs = np.fft.fftfreq(N, 1/fs)[:N//2]
    
plt.figure(figsize=(10, 5))
plt.semilogx(freqs, psd)
plt.title(f"3-Bit 2nd-Order Modulator (OSR={OSR})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid(True, which="both")
# Draw a line at the signal bandwidth
plt.axvline(fs / (2 * OSR), color='r', linestyle='--', label='Signal Bandwidth')
plt.ylim([-160, 20])
plt.show()