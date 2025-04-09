from main import analyze_audio_file
from pathlib import Path
import numpy as np
from typing import List


def process_and_return_beats(directory: str = "TestAudio") -> List[np.ndarray]:
    """Process all WAV files, print detailed results, and return beat arrays"""
    test_audio_dir = Path(directory)
    all_beats = []

    for wav_file in test_audio_dir.glob("*.wav"):
        try:
            print("\n" + "=" * 50)
            print(f"Processing {wav_file.name}")
            print("=" * 50)

            # Analyze file
            results = analyze_audio_file(str(wav_file))
            beats = results['beat_times']
            all_beats.append(beats)

            # Print detailed results
            print(f"\n• Estimated tempo: {results['tempo']:.1f} BPM")
            print(f"• Number of beats: {len(beats)}")
            print("\nFull beat sequence (seconds):")

            # Print beats in a readable format (5 per line)
            beat_str = np.array2string(beats.round(3),
                                       precision=3,
                                       suppress_small=True,
                                       max_line_width=80)
            print(beat_str.replace('[ ', '').replace('[', '').replace(']', ''))

            print(f"\nFirst 5 beats: {beats[:5].round(3)}")
            print(f"Average interval: {np.mean(np.diff(beats)):.3f} seconds")

        except Exception as e:
            print(f"\nError processing {wav_file.name}: {e}")
            continue

    return all_beats

#dette er hvordan man kan få beat listerne
#beat arrays er en array med alle listerne
if __name__ == "__main__":
    beat_arrays = process_and_return_beats()

    # Example of using the returned arrays
    print("\n=== Returned Beat Arrays Summary ===")
    for i, beats in enumerate(beat_arrays):
        print(f"\nFile {i + 1}:")
        print(f"Total beats: {len(beats)}")
        print(f"Beat range: {beats[0]:.3f}s to {beats[-1]:.3f}s")

    print(beat_arrays)