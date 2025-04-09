import matplotlib.pyplot as plt
import librosa.display


def plot_results(y, sr, beats, onset_env, bpm):
    """Plot waveform with beats and onset envelope including BPM"""
    plt.figure(figsize=(14, 8))

    # Plot waveform with BPM in title
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for t in librosa.frames_to_time(beats, sr=sr, hop_length=512):
        plt.axvline(t, color='r', linestyle='--', alpha=0.8)
    plt.title(f'Detected Beats | Estimated Tempo: {bpm:.1f} BPM')

    # Plot onset envelope
    plt.subplot(2, 1, 2)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    plt.plot(times, onset_env, label='Onset Strength')
    for t in librosa.frames_to_time(beats, sr=sr, hop_length=512):
        plt.axvline(t, color='r', linestyle='--', alpha=0.5)
    plt.title('Onset Envelope with Beat Positions')

    plt.tight_layout()
    plt.show()

"""import matplotlib.pyplot as plt
import librosa.display


def plot_results(y, sr, beats, onset_env):
    Plot waveform with beats and onset envelope
    plt.figure(figsize=(14, 8))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    for t in librosa.frames_to_time(beats, sr=sr, hop_length=512):
        plt.axvline(t, color='r', linestyle='--', alpha=0.8)
    plt.title('Detected Beats')

    # Plot onset envelope
    plt.subplot(2, 1, 2)
    times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    plt.plot(times, onset_env, label='Onset Strength')
    for t in librosa.frames_to_time(beats, sr=sr, hop_length=512):
        plt.axvline(t, color='r', linestyle='--', alpha=0.5)
    plt.title('Onset Envelope with Beat Positions')

    plt.tight_layout()
    plt.show()"""