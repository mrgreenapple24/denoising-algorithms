import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def apply_fxlms(primary_signal, reference_noise, mu, filter_order, fs, show_plot=True, ideal_signal=None):
    
    N = len(primary_signal)

    # Secondary path (simplified model)
    S = np.array([1.0])
    Shat = np.array([1.0])

    # Initialization
    w = np.zeros(filter_order)
    x_buf = np.zeros(filter_order)

    y = np.zeros(N)
    e = np.zeros(N)

    # REAL-TIME PLOT
    if show_plot:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_title("Real-Time Error Convergence (FxLMS)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Error")

    # FxLMS LOOP
    for n in range(N):

        x_buf = np.roll(x_buf, 1)
        x_buf[0] = reference_noise[n]

        x_filt = lfilter(Shat, [1], x_buf)
        x_filt_vec = x_filt[:filter_order]

        y[n] = np.dot(w, x_buf)
        y_sec = y[n]

        e[n] = primary_signal[n] - y_sec

        w = w + mu * e[n] * x_filt_vec

        # Real-time update
        if show_plot and n % 100 == 0:
            line.set_data(np.arange(n+1), e[:n+1])
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)

    if show_plot:
        plt.ioff()

    # SKIP CONVERGENCE
    start = int(0.3 * N)

    primary_play = primary_signal[start:]
    output_play = e[start:]

    # Normalize
    primary_play = primary_play / np.max(np.abs(primary_play))
    output_play = output_play / np.max(np.abs(output_play))

    # TIME DOMAIN
    if show_plot:
        plt.figure()

        plt.subplot(2,1,1)
        plt.plot(primary_play)
        plt.title("Before ANC (Noisy Signal)")

        plt.subplot(2,1,2)
        plt.plot(output_play)
        plt.title("After ANC (Cleaner Signal)")

        # FREQUENCY DOMAIN
        plt.figure()

        D = np.abs(np.fft.fft(primary_play))
        E = np.abs(np.fft.fft(output_play))
        f = np.linspace(0, fs, len(D))

        plt.subplot(2,1,1)
        plt.plot(f, D)
        plt.title("Spectrum Before ANC")

        plt.subplot(2,1,2)
        plt.plot(f, E)
        plt.title("Spectrum After ANC")

    return e