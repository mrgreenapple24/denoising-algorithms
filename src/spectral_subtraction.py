import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.ndimage as nd   
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD AUDIO FILES
# ==============================
mixed_input, sr1 = sf.read("synthetic_input.wav")
reference_noise, sr2 = sf.read("synthetic_noise.wav")
clean_output, sr3 = sf.read("clean_input.wav")

if not (sr1 == sr2 == sr3):
    raise ValueError("Sampling rates must match")

sr = sr1

# Convert to mono if needed
def to_mono(x):
    return x[:, 0] if x.ndim > 1 else x

mixed_input = to_mono(mixed_input)
reference_noise = to_mono(reference_noise)
clean_output = to_mono(clean_output)

mixed_input = mixed_input / np.max(np.abs(mixed_input))
reference_noise = reference_noise / np.max(np.abs(reference_noise))
clean_output = clean_output / np.max(np.abs(clean_output))

# Match lengths
min_len = min(len(mixed_input), len(reference_noise), len(clean_output))
mixed_input = mixed_input[:min_len]
reference_noise = reference_noise[:min_len]
clean_output = clean_output[:min_len]

# ==============================
# 2. SPECTRAL SUBTRACTION
# ==============================

# STFT parameters
nperseg = 1536
noverlap = 768
window = 'hann'

# STFT of noisy input
f, t_frames, Zxx = signal.stft(
    mixed_input,
    fs=sr,
    window=window,
    nperseg=nperseg,
    noverlap=noverlap
)

# STFT of noise-only signal
_, _, Nxx = signal.stft(
    reference_noise,
    fs=sr,
    window=window,
    nperseg=nperseg,
    noverlap=noverlap
)

# Estimate noise spectrum (average across time)
noise_mag = np.mean(np.abs(Nxx[:, :10]), axis=1, keepdims=True)

# Separate magnitude and phase
Y_mag = np.abs(Zxx)
Y_phase = np.angle(Zxx)

# Spectral subtraction parameters
alpha = 2.0 # over-subtraction
beta = 0.03  # spectral floor

# Apply spectral subtraction
S_mag = Y_mag - alpha * noise_mag
S_mag = np.maximum(S_mag, beta * noise_mag)

# Reconstruct signal
S_hat = S_mag * np.exp(1j * Y_phase)

_, spectral_output = signal.istft(
    S_hat,
    fs=sr,
    window=window,
    nperseg=nperseg,
    noverlap=noverlap
)

spectral_output = spectral_output[:min_len]

# ==============================
# 3. METRICS
# ==============================

def mse(x, y):
    return np.mean((x - y) ** 2)

def snr(clean, test):
    noise = clean - test
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-12))

input_snr = snr(clean_output, mixed_input)
output_snr = snr(clean_output, spectral_output)
snr_improvement = output_snr - input_snr
output_mse = mse(clean_output, spectral_output)

print("\n--- Spectral Subtraction Results ---")
print(f"Input SNR        : {input_snr:.2f} dB")
print(f"Output SNR       : {output_snr:.2f} dB")
print(f"SNR Improvement  : {snr_improvement:.2f} dB")
print(f"MSE              : {output_mse:.6f}")

# ==============================
# 4. PLOTTING (MATCH LMS STYLE)
# ==============================

time = np.arange(min_len) / sr

# Zoom window (same idea as LMS plot)
start_time = 1.0
end_time = 1.062

start_idx = int(start_time * sr)
end_idx = int(end_time * sr)

plt.figure(figsize=(16, 9))

plt.subplot(4, 1, 1)
plt.plot(time[start_idx:end_idx], clean_output[start_idx:end_idx], color='green')
plt.title("Clean Signal (Expected Output)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 2)
plt.plot(time[start_idx:end_idx], mixed_input[start_idx:end_idx], color='red')
plt.title("Clean + Noise Audio")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 3)
plt.plot(time[start_idx:end_idx], reference_noise[start_idx:end_idx], color='gray')
plt.title("Noise Only")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(4, 1, 4)
plt.plot(time[start_idx:end_idx], spectral_output[start_idx:end_idx], color='blue')
plt.title("Spectral Subtraction Output")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# ==============================
# 5. SAVE OUTPUT (IMPORTANT)
# ==============================

sf.write("./spectral_subtraction_output.wav", spectral_output, sr)