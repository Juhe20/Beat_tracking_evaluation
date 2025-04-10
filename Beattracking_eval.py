import os
import sys
import librosa
import numpy as np
import mir_eval
import matplotlib.pyplot as plt
from glob import glob
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'PLP', 'real_time_plp-main'))
from beatcli import capture_audio, time, stability


asyncio.run(capture_audio())
# === CONFIG ===
audio_dir = "Music_files/blues"
annotation_dir = "Annotated_beats/blues"

# === Storage for plotting ===
file_names = []
f_scores = []
p_scores = []
cemgil_scores = []

# === EVALUATION LOOP ===
for wav_path in sorted(glob(os.path.join(audio_dir, "blues.*.wav"))):
    file_id = os.path.splitext(os.path.basename(wav_path))[0].split(".")[-1]
    beat_filename = f"gtzan_blues_{file_id}.beats"
    beat_path = os.path.join(annotation_dir, beat_filename)

    if not os.path.exists(beat_path):
        print(f"Annotation not found for {beat_filename}, skipping.")
        continue

    y, sr = librosa.load(wav_path, sr=None)
    gt_beats = np.loadtxt(beat_path, usecols=0)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    est_onsets = librosa.frames_to_time(onset_frames, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    est_beats = librosa.frames_to_time(beat_frames, sr=sr)

    f_measure = mir_eval.onset.f_measure(gt_beats, est_onsets)[0]
    p_score = mir_eval.beat.p_score(gt_beats, est_onsets)
    cemgil_score = mir_eval.beat.cemgil(gt_beats, est_onsets)[0]  # Only the first value
    goto_score = mir_eval.beat.goto(gt_beats, est_beats)
    continuity_score = mir_eval.beat.continuity(gt_beats, est_beats)

    # print(f"{beat_filename} | F: {f_measure} | P: {p_score} | Cemgil: {cemgil_score} | Goto: {goto_score} | Continuity: {continuity_score}")

    # Store for plotting
    file_names.append(beat_filename)
    f_scores.append(f_measure)
    p_scores.append(p_score)
    cemgil_scores.append(cemgil_score)

    f_average = np.mean(f_scores)
    p_score_average = np.mean(p_scores)
    cemgil_score_average = np.mean(cemgil_scores)

# === Data for plotting ===
genre = ['blues']
scores = [f_average, p_score_average, cemgil_score_average]
labels = ['F-measure', 'P-score', 'Cemgil']

x = np.arange(len(genre))  # X-axis positions
width = 0.2  # Width of each bar

# Plot bars for each score, slightly shifted on x-axis
plt.figure(figsize=(8, 6))
plt.bar(x - width, f_average, width, label='F-measure', color='skyblue')
plt.bar(x, p_score_average, width, label='P-score', color='lightgreen')
plt.bar(x + width, cemgil_score_average, width, label='Cemgil', color='salmon')

plt.xticks(x, genre)
plt.ylim(0, 1.1)
plt.ylabel("Average Score")
plt.title("Average Evaluation Metrics for Genre: Blues")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()
