import numpy as np
import os
import soundfile as sf
from lms import apply_lms

def run_synthetic_mode():
    print("\nGenerating Synthetic Data")
    fs = 16000      #set sampling rate of sine wave to 16kHz (nyquist theorem and ease of computation)
    duration = 5.0  #to create 2s sine wave clip
    t = np.linspace(0, duration, int(fs * duration), endpoint=False) 

    # 1. Create synthetic test data (Voice sine wave + Random Noise)
    clean_voice_ideal = np.sin(2 * np.pi * 400 * t) #creates 400Hz sound wave
    ref_noise = np.random.normal(0, 0.5, len(t)) #generates random numbers (white noise)
    
    # Simulate noise hitting primary mic with a small delay
    primary_mic = clean_voice_ideal + (np.roll(ref_noise, 5) * 0.8) #add noise to original signal 
    #includes delay of 5 samples and 0.8 times the amplitude
    #simulates real world delays/damping. If added plainly, it would cancel out perfectly and only most recent weight would be 1
    #it is also linked to filter order as noise sample must be within the filter order
    
    # Normalize before processing
    primary_mic = primary_mic / np.max(np.abs(primary_mic)) #make max point = 1 0 < mu < 2/(L*P_x) stability condition
    ref_noise = ref_noise / np.max(np.abs(ref_noise)) #as numbers can be huge and multiplication will lead to higher numbers

    print("Running LMS Algorithm...")
    # 2. Run the math engine from lms.py
    recovered_audio = apply_lms(
        primary_signal=primary_mic, 
        reference_noise=ref_noise, 
        mu=0.01, 
        filter_order=32, 
        fs=fs, 
        show_plot=True,
        ideal_signal=clean_voice_ideal
    )
    
    # 3. Create folder and save results
    folder_name = "LMS_Synthetic_Audio"
    os.makedirs(folder_name, exist_ok=True)
    
    print(f"Saving synthetic audio files to '{folder_name}'...")
    sf.write(os.path.join(folder_name, "clean_input.wav"), clean_voice_ideal, fs)
    sf.write(os.path.join(folder_name, "synthetic_input.wav"), primary_mic, fs)
    sf.write(os.path.join(folder_name, "synthetic_noise.wav"), ref_noise, fs)
    sf.write(os.path.join(folder_name, "synthetic_output.wav"), recovered_audio, fs)
    
    print("Synthetic test complete.")

def run_real_audio_mode():
    print("\n--- Loading Real Audio Files ---")
    primary_file = "LMS_Real_Audio/audio.wav" 
    noise_file = "LMS_Real_Audio/noise.wav"
    output_file = "LMS_Real_Audio/LMSop.wav"

    if not os.path.exists(primary_file) or not os.path.exists(noise_file):
        print(f"ERROR: Could not find '{primary_file}' or '{noise_file}'.")
        print("Please place them in the same folder as this script.")
        return

    # Read audio using soundfile 
    primary_mic, fs_primary = sf.read(primary_file)
    ref_noise, fs_noise = sf.read(noise_file)

    print(f"DEBUG: Primary Mic raw max value: {np.max(np.abs(primary_mic))}")
    print(f"DEBUG: Ref Noise raw max value:  {np.max(np.abs(ref_noise))}")
    print(f"DEBUG: Total samples in file: {len(primary_mic)}")

    # Force Mono if stereo
    if primary_mic.ndim > 1: primary_mic = primary_mic[:, 0]
    if ref_noise.ndim > 1: ref_noise = ref_noise[:, 0] 

    # Normalize to prevent math explosion (NaNs)
    primary_mic = primary_mic / np.max(np.abs(primary_mic))
    ref_noise = ref_noise / np.max(np.abs(ref_noise))

    print("Running LMS Algorithm...")
    # Real audio is complex: use lower mu and higher filter_order
    recovered_audio = apply_lms(
        primary_signal=primary_mic, 
        reference_noise=ref_noise, 
        mu=0.009,           
        filter_order=2048,   # High order for fan noise complexity
        fs=fs_primary, 
        show_plot=True,
        
    )
    
    # Safety check for NaNs (if mu was still too high) #not a number, represents missing, undefined, or unrepresentable numerical data. 
    if np.isnan(recovered_audio).any():
        print("CRITICAL ERROR: Output contains NaNs. Lower the 'mu' value.")
    else:
        print(f"Saving recovered audio as '{output_file}'...")
        sf.write(output_file, recovered_audio, fs_primary)
        print("Done!")

# MAIN STARTUP

if __name__ == "__main__":
    print("=======================================")
    print("     LMS Noise Cancellation Tester     ")
    print("=======================================")
    print("1. Synthetic Mode (Generates & Saves Sine Waves)")
    print("2. Real Audio Mode (Processes .wav Files)")
    
    choice = input("\nEnter 1 or 2 to select mode: ")
    
    if choice == '1':
        run_synthetic_mode()
    elif choice == '2':
        run_real_audio_mode()
    else:
        print("Invalid choice.")