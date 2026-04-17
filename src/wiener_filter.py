import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def apply_wiener(primary_signal, reference_noise,
                 fs=16000, nperseg=1024,
                 alpha=2.0, show_plot=True, ideal_signal=None):
    """Applies Wiener filtering to reduce noise from a signal"""

    # ----------------------------------
    # Make both signals equal in length
    # ----------------------------------
    n_samples = min(len(primary_signal), len(reference_noise))
    primary_signal = primary_signal[:n_samples]
    reference_noise = reference_noise[:n_samples]

    # ----------------------------------
    # Transform signals into frequency domain
    # ----------------------------------
    f, t_stft, Zxx = signal.stft(primary_signal, fs=fs, nperseg=nperseg)
    _, _, Zxx_noise = signal.stft(reference_noise, fs=fs, nperseg=nperseg)

    # ----------------------------------
    # Estimate power of signal and noise
    # ----------------------------------
    noise_psd = np.mean(np.abs(Zxx_noise) ** 2, axis=1, keepdims=True)
    signal_psd = np.abs(Zxx) ** 2

    # ----------------------------------
    # Compute Wiener gain
    # Reduces components where noise dominates
    # ----------------------------------
    H = signal_psd / (signal_psd + alpha * noise_psd + 1e-10)

    # Prevent complete suppression
    H = np.maximum(H, 0.05)

    # Apply filtering in frequency domain
    Zxx_filtered = H * Zxx

    # ----------------------------------
    # Convert back to time domain
    # ----------------------------------
    _, clean_voice = signal.istft(Zxx_filtered, fs=fs)
    clean_voice = clean_voice[:n_samples]

    # Normalize output
    clean_voice = clean_voice / (np.max(np.abs(clean_voice)) + 1e-10)

    # ----------------------------------
    # Plot signals for visualization
    # ----------------------------------
    if show_plot:
        plot_samples = min(1000, n_samples)

        # Skip initial region to avoid edge effects
        start_idx = min(int(7.0 * fs), n_samples - plot_samples)
        end_idx = start_idx + plot_samples

        t = np.linspace(start_idx/fs, end_idx/fs, plot_samples, endpoint=False)

        rows = 4 if ideal_signal is not None else 3
        plt.figure(figsize=(12, 10))

        curr_row = 1

        # Ideal clean signal (if available)
        if ideal_signal is not None:
            plt.subplot(rows, 1, curr_row)
            plt.title("Clean sine wave (Expected Output)")
            plt.plot(t, ideal_signal[start_idx:end_idx], color='green', linewidth=2)
            plt.grid(True)
            curr_row += 1

        # Noisy input
        plt.subplot(rows, 1, curr_row)
        plt.title("Input Signal (Signal + Noise)")
        plt.plot(t, primary_signal[start_idx:end_idx], color='red')
        plt.grid(True)
        curr_row += 1

        # Noise reference
        plt.subplot(rows, 1, curr_row)
        plt.title("Noise Signal")
        plt.plot(t, reference_noise[start_idx:end_idx], color='gray')
        plt.grid(True)
        curr_row += 1

        # Filtered output
        plt.subplot(rows, 1, curr_row)
        plt.title("Filtered Output (Wiener)")
        plt.plot(t, clean_voice[start_idx:end_idx], color='blue', linewidth=2)
        plt.xlabel("Time (seconds)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # ----------------------------------
        # Print basic filter behavior stats
        # ----------------------------------
        print(f"\n--- Wiener Filter Stats ---")
        print(f"Max Gain Value: {np.max(H)}")
        print(f"Min Gain Value: {np.min(H)}")
        print(f"Mean Gain Value: {np.mean(H)}")

    return clean_voice