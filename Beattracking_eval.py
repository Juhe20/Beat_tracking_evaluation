import os
import librosa
import numpy as np
import mir_eval
from glob import glob

# === CONFIG ===
audio_dir = "Music_files/blues"
annotation_dir = "Annotated_beats/blues"

# === EVALUATION LOOP ===
for wav_path in sorted(glob(os.path.join(audio_dir, "blues.*.wav"))):
    # Extract the number part from 'blues_00000.wav'
    file_id = os.path.splitext(os.path.basename(wav_path))[0].split(".")[-1]

    # Build corresponding annotation filename: 'gtzan_blues_00000.beats'
    beat_filename = f"gtzan_blues_{file_id}.beats"
    beat_path = os.path.join(annotation_dir, beat_filename)

    # Skip if annotation doesn't exist
    if not os.path.exists(beat_path):
        print(f"Annotation not found for {beat_filename}, skipping.")
        continue

    # === Load audio and ground truth ===
    y, sr = librosa.load(wav_path, sr=None)
    gt_beats = np.loadtxt(beat_path, usecols=0)


    # === Detect onsets ===
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    est_onsets = librosa.frames_to_time(onset_frames, sr=sr)

    # === Evaluate ===
    f_measure = mir_eval.onset.f_measure(gt_beats, est_onsets)
    p_score = mir_eval.beat.p_score(gt_beats, est_onsets)
    cemgil = mir_eval.beat.cemgil(gt_beats, est_onsets)

    print(f"{beat_filename} | F: {f_measure} | P: {p_score} | Cemgil: {cemgil}")

