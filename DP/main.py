import argparse
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union
from Utils.audio import load_wav, validate_wav
from onset import compute_onset_strength
from beat_tracking import track_beats
from Utils.plot import plot_results
from config import HOP_LENGTH, SAMPLE_RATE


def analyze_audio_file(file_path: str, plot: bool = False) -> Dict[str, Union[np.ndarray, float]]:
    """
    Analyze a WAV file and return beat tracking results
    Args:
        file_path: Path to WAV file
        plot: Whether to show visualization plot
    Returns:
        Dictionary containing:
            - 'beat_times': Array of beat times in seconds
            - 'beat_frames': Array of beat positions in frames
            - 'tempo': Estimated tempo in BPM
            - 'onset_env': Onset strength envelope
    """
    # Validate and load WAV
    wav_path = validate_wav(file_path)
    y, sr = load_wav(wav_path)

    # Compute onset strength
    onset_env = compute_onset_strength(y, sr)

    # Track beats and get tempo
    beat_frames, tempo = track_beats(onset_env, sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # Output results
    print(f"\nAnalysis of {Path(wav_path).name}:")
    print(f"- Estimated tempo: {tempo:.1f} BPM")
    print(f"- Detected {len(beat_times)} beats")

    # Visualize if requested
    if plot:
        plot_results(y, sr, beat_frames, onset_env, tempo)

    return {
        'beat_times': beat_times,
        'beat_frames': beat_frames,
        'tempo': tempo,
        'onset_env': onset_env
    }


def process_directory(input_dir: str, plot: bool = False) -> Dict[str, list]:
    """
    Process all WAV files in a directory
    Args:
        input_dir: Directory containing WAV files
        plot: Whether to show plots for each file
    Returns:
        Dictionary of lists containing results for all files
    """
    input_path = Path(input_dir)
    results = {
        'filenames': [],
        'beat_times': [],
        'beat_frames': [],
        'tempos': []
    }

    for wav_file in input_path.glob('*.wav'):
        try:
            print(f"Processing {wav_file.name}...")
            result = analyze_audio_file(str(wav_file), plot)

            results['filenames'].append(wav_file.name)
            results['beat_times'].append(result['beat_times'])
            results['beat_frames'].append(result['beat_frames'])
            results['tempos'].append(result['tempo'])

        except Exception as e:
            print(f"Error processing {wav_file.name}: {e}")
            continue

    return results


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description='Beat tracking for WAV files')
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to WAV file or directory'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show visualization plots'
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)

    try:
        if input_path.is_file():
            # Single file processing
            analyze_audio_file(args.input_path, args.plot)
        elif input_path.is_dir():
            # Directory processing
            results = process_directory(args.input_path, args.plot)
            print("\nProcessing complete!")
        else:
            raise FileNotFoundError(f"Path not found: {args.input_path}")
    except Exception as e:
        print(f"Error: {e}")

"""import argparse
import librosa
import numpy as np
from pathlib import Path
from Utils.audio import load_wav, validate_wav
from onset import compute_onset_strength
from beat_tracking import track_beats
from Utils.plot import plot_results
from config import HOP_LENGTH, SAMPLE_RATE


def process_wav_file(input_wav):
    Complete beat tracking pipeline for WAV files
    # Validate and load WAV
    wav_path = validate_wav(input_wav)
    y, sr = load_wav(wav_path)

    # Compute onset strength
    onset_env = compute_onset_strength(y, sr)

    # Track beats and get tempo
    beat_frames, estimated_bpm = track_beats(onset_env, sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP_LENGTH)

    # Output results
    print(f"\nAnalysis of {Path(wav_path).name}:")
    print(f"- Estimated tempo: {estimated_bpm:.1f} BPM")
    print(f"- Detected {len(beat_times)} beats at times (seconds):")
    print(np.round(beat_times, 3))

    # Visualize
    plot_results(y, sr, beat_frames, onset_env, estimated_bpm)

    return beat_times, estimated_bpm


if __name__ == "__main__":
    wav_file = input("Enter WAV filename (from TestAudio folder): ")
    try:
        process_wav_file(f"TestAudio/{wav_file}")
    except Exception as e:
        print(f"Error processing file: {e}")"""