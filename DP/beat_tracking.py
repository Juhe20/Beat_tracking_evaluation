import numpy as np
import librosa
from config import HOP_LENGTH, SAMPLE_RATE, INIT_TEMPO, ALPHA, MIN_TEMPO, MAX_TEMPO


def track_beats(onset_env, sr):
    """Improved beat tracker with accurate tempo detection for both fast and slow tempos"""
    # Get tempo estimate using current librosa function
    try:
        # For librosa 0.10.0+
        tempo_candidates = librosa.feature.rhythm.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=HOP_LENGTH,
            aggregate=np.median
        )
    except AttributeError:
        # Fallback for older versions
        tempo_candidates = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=HOP_LENGTH,
            aggregate=np.median
        )

    # Initial tempo selection with bias toward slower tempos for jazz
    if INIT_TEMPO is None:
        # Consider both original and half-time tempos
        extended_candidates = []
        for t in tempo_candidates:
            extended_candidates.extend([t, t / 2])

        # Filter valid tempos with preference for slower ones
        valid_tempos = [t for t in extended_candidates if MIN_TEMPO <= t <= MAX_TEMPO]
        if not valid_tempos:
            tempo = 60.0  # Default fallback
        else:
            # Weight toward slower tempos
            weights = np.array([1.0 / (t + 0.1) for t in valid_tempos])
            tempo = np.average(valid_tempos, weights=weights)
    else:
        tempo = INIT_TEMPO

    # Convert tempo to interval in frames
    beat_period = round(60.0 * sr / (tempo * HOP_LENGTH))

    # Dynamic programming setup
    n = len(onset_env)
    C = np.zeros(n)  # Cumulative score
    P = np.zeros(n, dtype=int)  # Backpointers

    # Initialize first two frames
    C[0] = onset_env[0]
    C[1] = onset_env[1]

    # Main DP loop with tempo-adaptive window
    for i in range(2, n):
        # Larger window for slower tempos
        max_lookback = min(4 * beat_period, i) if tempo < 90 else min(2 * beat_period, i)
        candidates = []

        for delta in range(1, max_lookback + 1):
            penalty = ALPHA * (np.log2(delta / beat_period) ** 2)
            score = C[i - delta] + onset_env[i] - penalty
            candidates.append((score, delta))

        C[i], best_delta = max(candidates, key=lambda x: x[0])
        P[i] = i - best_delta

    # Backtrack to find beat sequence
    beats = []
    i = n - 1
    while i > 0:
        beats.append(i)
        i = P[i]

    beats = np.array(beats[::-1])

    # Calculate final tempo from detected beats
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP_LENGTH)
        ibi = np.diff(beat_times)  # Inter-beat intervals
        actual_tempo = 60.0 / np.median(ibi)

        # Verify if we should use half-time tempo
        if actual_tempo > MAX_TEMPO / 1.5 and len(beats) > 4:
            # Try checking every other beat
            alt_tempo = 60.0 / np.median(ibi[::2])
            if abs(alt_tempo - tempo) < 30:  # If more plausible
                actual_tempo = alt_tempo
                beats = beats[::2]  # Keep every other beat

        tempo = np.clip(actual_tempo, MIN_TEMPO, MAX_TEMPO)

    return beats, tempo


"""import numpy as np
import librosa
from config import HOP_LENGTH, SAMPLE_RATE, INIT_TEMPO, ALPHA, MIN_TEMPO, MAX_TEMPO


def track_beats(onset_env, sr):
    Dynamic programming beat tracker
    # Estimate tempo if not provided
    tempo = INIT_TEMPO or librosa.beat.tempo(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=HOP_LENGTH,
        min_tempo=MIN_TEMPO,
        max_tempo=MAX_TEMPO
    )[0]

    # Convert tempo to interval in frames
    beat_period = round(60.0 * sr / (tempo * HOP_LENGTH))

    # Dynamic programming setup
    n = len(onset_env)
    C = np.zeros(n)  # Cumulative score
    P = np.zeros(n, dtype=int)  # Backpointers

    # Initialize first two frames
    C[0] = onset_env[0]
    C[1] = onset_env[1]

    # Main DP loop
    for i in range(2, n):
        max_lookback = min(2 * beat_period, i)
        candidates = []

        for delta in range(1, max_lookback + 1):
            penalty = ALPHA * (np.log2(delta / beat_period) ** 2)
            score = C[i - delta] + onset_env[i] - penalty
            candidates.append((score, delta))

        C[i], best_delta = max(candidates, key=lambda x: x[0])
        P[i] = i - best_delta

    # Backtrack to find beat sequence
    beats = []
    i = n - 1
    while i > 0:
        beats.append(i)
        i = P[i]

    return np.array(beats[::-1])"""