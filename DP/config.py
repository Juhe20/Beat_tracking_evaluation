# Audio processing parameters
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048

# Beat tracking parameters
INIT_TEMPO = None      # None for auto-detect
ALPHA = 0.8            # Tempo deviation penalty
MIN_TEMPO = 40         # Minimum expected tempo (BPM)
MAX_TEMPO = 200        # Maximum expected tempo (BPM)