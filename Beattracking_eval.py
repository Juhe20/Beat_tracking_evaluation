import os
import sys
import librosa
import numpy as np
import mir_eval
import matplotlib.pyplot as plt
from glob import glob
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'PLP', 'real_time_plp-main'))
# noinspection PyUnresolvedReferences
from beatcli import capture_audio, beat_times_list, tempo

# --------- SETUP ---------
audio_dir = "Music_files/blues"
annotation_dir = "Annotated_beats/blues"

# --------- STORAGE LIST FOR PLOTTING ---------
file_names = []
f_scores = []
p_scores = []
cemgil_scores = []
goto_scores = []
continuity_scores = []

# --------- EVALUATE ON ALL METHODS ---------
def evaluate(gt_beats, est_beats):
    #Need to figure out if we can use beat for f_measure too or if it has to be onset
    f_measure = mir_eval.onset.f_measure(gt_beats, est_beats)[0]
    p_score = mir_eval.beat.p_score(gt_beats, est_beats)
    cemgil = mir_eval.beat.cemgil(gt_beats, est_beats)[0]
    goto = mir_eval.beat.goto(gt_beats, est_beats)
    #Rounding this one cause it had 4 values with 10 digits each
    continuity = tuple(round(float(x), 4) for x in mir_eval.beat.continuity(gt_beats, est_beats))

    f_scores.append(f_measure)
    p_scores.append(p_score)
    cemgil_scores.append(cemgil)
    goto_scores.append(goto)
    continuity_scores.append(continuity)

    return f_measure, p_score, cemgil, goto, continuity


#--------- LIBROSA EVALUATION ---------
def librosa_eval():
    #Load path for audio and ground truth files
    for wav_path in sorted(glob(os.path.join(audio_dir, "blues.*.wav"))):
        file_id = os.path.splitext(os.path.basename(wav_path))[0].split(".")[-1]
        beat_filename = f"gtzan_blues_{file_id}.beats"
        beat_path = os.path.join(annotation_dir, beat_filename)
        #Check to skip if it doesn't find a ground truth file
        if not os.path.exists(beat_path):
            print(f"Annotation not found for {beat_filename}, skipping.")
            continue
        #Use the loaded path for ground truth and
        y, sr = librosa.load(wav_path, sr=None)
        librosa_gt_beats = np.loadtxt(beat_path, usecols=0)

        #Use dynamic programming (Librosa) algorithm to find estimated beats
        librosa_beat_times = librosa.beat.beat_track(y=y, sr=sr)
        librosa_est_beats = librosa.frames_to_time(librosa_beat_times, sr=sr)
        f_measure, p_score, cemgil, goto, continuity = evaluate(librosa_gt_beats, librosa_est_beats)

        print(f"{beat_filename} | F: {f_measure} | P: {p_score} | Cemgil: {cemgil} | Goto: {goto} | Continuity: {continuity}")

        #Store for plotting
        file_names.append(beat_filename)

        #Take average of all audio files of a genre (not sure if necessary)
        f_average = np.mean(f_scores)
        p_score_average = np.mean(p_scores)
        cemgil_score_average = np.mean(cemgil_scores)
        goto_score_average = np.mean(goto_scores)
        continuity_score_average = np.mean(continuity_scores)


# --------- PLP EVALUATION ---------
def plp_eval():
    filename = "00000"
    plp_gt_beats_file = f"Annotated_beats/blues/gtzan_blues_{filename}.beats"
    plp_gt_beats = np.loadtxt(plp_gt_beats_file, usecols=0)
    plp_est_beats = np.array(asyncio.run(capture_audio()))

    #beat_times_list variable comes from the capture_audio method from beatcli.py (PLP script)
    f_measure, p_score, cemgil, goto, continuity = evaluate(plp_gt_beats, plp_est_beats)
    print(f"F: {f_measure} | P: {p_score} | Cemgil: {cemgil} | Goto: {goto} | Continuity: {continuity}")


# --------- PLOTTING ---------
#adding when everything works

# --------- MAIN (RUN EVALUATION HERE) ---------
if __name__ == '__main__':
    #librosa_eval()
    plp_eval()