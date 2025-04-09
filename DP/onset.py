import numpy as np
import librosa
from config import SAMPLE_RATE, HOP_LENGTH, N_FFT

def compute_onset_strength(y, sr):
    """Compute onset strength envelope from audio"""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return librosa.onset.onset_strength(S=S, sr=sr)