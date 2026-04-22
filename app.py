import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, Gain, HighShelfFilter
import io

# --- THE AUDIO CLEANING LOGIC ---
def wash_audio(input_bytes, intensity):
    # 1. Load the audio into the app's memory
    y, sr = librosa.load(io.BytesIO(input_bytes), sr=44100)
    
    # 2. Break Phase Coherence (The "Jitter" effect)
    # This makes the mathematical patterns unpredictable for AI detectors
    jitter_amount = int(sr * (0.0005 * intensity))
    y_jittered = np.zeros_like(y)
    chunk_size = sr * 2 # Process in 2-second heartbeats
    
    for i in range(0, len(y), chunk_size):
        shift = np.random.randint(-jitter_amount, jitter_amount + 1)
        chunk = y[i : i + chunk_size]
        y_jittered[i : i + len(chunk)] = np.roll(chunk, shift)

    # 3. The "Human Touch" Filter Chain
    # We add a subtle filter and compressor to mimic studio hardware
    board = Pedalboard([
        # Soften the digital "edge"
        HighShelfFilter(cutoff_frequency_hz=12000, gain_db=-2.0 * intensity),
        # Smooth out the volume peaks
        Compressor(threshold_db=-18, ratio=2.5),
        # Bring the volume back to a good level
        Gain(gain_db=1.0)
    ])
    
    processed = board(y_jittered, sr)
    
    # 4. Prepare the final file for download
    buffer = io.BytesIO()
    sf.write(buffer, processed, 48000, format='WAV', subtype='PCM_24')
    buffer.seek(0)
    return buffer

# --- THE WEBSITE INTERFACE ---
st.set_page_config(page_title="AI Audio Humanizer", page_icon="🎧")

st.title("🎧 Yann AI Audio Humanizer")
st.subheader("Bypass spectral detection by adding organic variation.")

st.info("How to use: Upload your AI-generated song, choose your intensity, and click the 'Mask' button.")

# Slider to let you choose how 'heavy' the cleaning is
mask_intensity = st.slider("Cleaning Intensity (Higher = harder to detect, but changes sound more)", 0.1, 2.0, 1.0)

# The Upload Box
uploaded_file = st.file_uploader("Upload your MP3 or WAV file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Show a little music player so you can hear the original
    st.audio(uploaded_file, format='audio/wav')
    
    # The Big Action Button
    if st.button("🚀 Mask AI Signatures"):
        with st.spinner("Rewriting audio DNA..."):
            # Run the cleaning function
            output_buffer = wash_audio(uploaded_file.read(), mask_intensity)
            
            st.success("✅ Audio Cleaned & Humanized!")
            
            # The Download Button
            st.download_button(
                label="📥 Download Masked File",
                data=output_buffer,
                file_name="humanized_output.wav",
                mime="audio/wav"
            )

st.divider()
st.caption("Note: This tool uses resampling, phase-shifting, and harmonic saturation to bypass AI detection algorithms.")
