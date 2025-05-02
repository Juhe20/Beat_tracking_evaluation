import os
import sys
import librosa
import numpy as np
import mir_eval
import matplotlib.pyplot as plt
from glob import glob
import asyncio
import sounddevice as sd
sys.path.append(os.path.join(os.path.dirname(__file__), 'PLP', 'real_time_plp-main'))
# noinspection PyUnresolvedReferences
from beatcli import capture_audio, beat_times_list, tempo
sys.path.append(os.path.join(os.path.dirname(__file__), 'beat_this-main'))
# noinspection PyUnresolvedReferences
from beatthis import beat_this

# --------- STORAGE LIST FOR PLOTTING ---------
f_scores = []
p_scores = []
cemgil_scores = []
goto_scores = []
continuity_scores = []
infogain_scores = []
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
average_results = {
    "librosa": {},
    "plp": {},
    "beat_this": {}
}

# --------- EVALUATE ON ALL METHODS ---------
def evaluate(gt_beats, est_beats):
    #Need to figure out if we can use beat for f_measure too or if it has to be onset
    f_measure = mir_eval.onset.f_measure(gt_beats, est_beats)[0]
    p_score = mir_eval.beat.p_score(gt_beats, est_beats)
    cemgil = mir_eval.beat.cemgil(gt_beats, est_beats)[0]
    goto = mir_eval.beat.goto(gt_beats, est_beats)
    infogain = mir_eval.beat.information_gain(gt_beats, est_beats)
    #Rounding this one because it had 4 values with 10 digits each
    continuity = tuple(round(float(x), 4) for x in mir_eval.beat.continuity(gt_beats, est_beats))
    infogain_scores.append(infogain)
    f_scores.append(f_measure)
    p_scores.append(p_score)
    cemgil_scores.append(cemgil)
    goto_scores.append(goto)

    continuity_scores.append(continuity)

    return f_measure, p_score, cemgil, goto, infogain, continuity

def calculate_averages():
    f_average = np.mean(f_scores)
    p_score_average = np.mean(p_scores)
    cemgil_score_average = np.mean(cemgil_scores)
    goto_score_average = np.mean(goto_scores)
    infogain_score_average = np.mean(infogain_scores)
    continuity_score_average = np.mean(continuity_scores)

    return f_average, p_score_average, cemgil_score_average, goto_score_average, infogain_score_average, continuity_score_average

#--------- LIBROSA EVALUATION ---------
def librosa_eval(audio_path, gt_path):
    y, sr = librosa.load(audio_path, sr=None)
    gt_beats = np.loadtxt(gt_path, usecols=0)

    # Use the Librosa algorithm to find estimated beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    est_beats = librosa.frames_to_time(beat_frames, sr=sr)

    f_measure, p_score, cemgil, goto, infogain, continuity = evaluate(gt_beats, est_beats)

    print(f"{audio_path} | F: {f_measure} | P: {p_score} | Cemgil: {cemgil} | Goto: {goto} | Continuity: {continuity}")

    #plot_beats_vs_ground_truth(gt_beats, est_beats, "Librosa estimated beats")

    f_average, p_score_average, cemgil_score_average, goto_score_average,infogain_score_average ,continuity_score_average = calculate_averages()
    return f_average, p_score_average, cemgil_score_average, goto_score_average,infogain_score_average, continuity_score_average


# --------- PLP EVALUATION ---------
def plp_eval(audio_path,gt_path):
    gt_beats = np.loadtxt(gt_path, usecols=0)
    #Run capture_audio method from beatcli.py for 30 seconds to get estimated beats
    est_beats = np.array(asyncio.run(capture_audio(audio_path)))
    f_measure, p_score, cemgil, goto, infogain, continuity = evaluate(gt_beats, est_beats)
    print(f"F: {f_measure} | P: {p_score} | Cemgil: {cemgil} | Goto: {goto} | Continuity: {continuity}")
    #plot_beats_vs_ground_truth(gt_beats, est_beats, "PLP estimated beats")
    #Take average of all audio files of a genre (not sure if necessary)
    f_average, p_score_average, cemgil_score_average, goto_score_average,infogain_score_average, continuity_score_average = calculate_averages()
    return f_average, p_score_average, cemgil_score_average, goto_score_average,infogain_score_average, continuity_score_average


# --------- BEAT THIS! EVALUATION ---------
def beat_this_eval(audio_path, gt_path):
    gt_beats = np.loadtxt(gt_path, usecols=0)
    est_beats = beat_this(audio_path)

    #Run beat_this method from beatthis.py to get estimated beats
    f_measure, p_score, cemgil, goto, infogain, continuity = evaluate(gt_beats, est_beats)
    print(f"File: {audio_path} | F: {f_measure} | P: {p_score} | Cemgil: {cemgil} | Goto: {goto} | Continuity: {continuity}")
    #plot_beats_vs_ground_truth(gt_beats, est_beats, "Beat_this! estimated beats")
    #Take average of all audio files of a genre (not sure if necessary)
    f_average, p_score_average, cemgil_score_average, goto_score_average, infogain_score_average, continuity_score_average = calculate_averages()
    return f_average, p_score_average, cemgil_score_average, goto_score_average,infogain_score_average, continuity_score_average


# --------- PLOTTING ---------
def plot_beats_vs_ground_truth(gt_beats, est_beats, label, color='tab:blue'):
    plt.figure(figsize=(12, 3))
    for beat in gt_beats:
        plt.axvline(beat, color='green', linestyle='--', label='Ground Truth' if beat == gt_beats[0] else "")
    for beat in est_beats:
        plt.axvline(beat, color=color, linestyle='-', label=label if beat == est_beats[0] else "")
    plt.legend()
    plt.title("Estimated beats vs ground truth beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Beats")
    plt.tight_layout()
    plt.show()


def plot_average_scores_per_genre(average_results, metric):
    plt.figure(figsize=(12, 6))

    # Get all genres from any algorithm
    algo = next(iter(average_results.values()))
    genres = list(algo.keys())
    algorithms = list(average_results.keys())

    for algorithm in algorithms:
        values = []
        for genre in genres:
            try:
                value = average_results[algorithm][genre][metric]
            except KeyError:
                value = 0  # or use np.nan if you prefer gaps
            values.append(value)
        plt.plot(genres, values, marker='o', label=algorithm)

    plt.title(f"Average {metric.replace('_', ' ').title()} per Genre")
    plt.xlabel("Genre")
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compare_algorithms_on_metric(average_results, metric):
    genres = list(next(iter(average_results.values())).keys())
    algorithms = list(average_results.keys())
    x = np.arange(len(genres))  # the label locations
    width = 0.25

    plt.figure(figsize=(14, 6))
    for i, algo in enumerate(algorithms):
        values = [average_results[algo][genre][metric] for genre in genres]
        plt.bar(x + i * width, values, width, label=algo)

    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Algorithm Comparison on {metric.replace('_', ' ').title()}")
    plt.xticks(x + width, genres, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_all_metrics(average_results):
    algorithms = list(average_results.keys())
    genres = list(next(iter(average_results.values())).keys())
    metrics = list(next(iter(average_results.values()))[genres[0]].keys())

    num_metrics = len(metrics)
    fig, axs = plt.subplots(nrows=1, ncols=num_metrics, figsize=(6 * num_metrics, 5), sharey=False)

    if num_metrics == 1:
        axs = [axs]  # Ensure it's iterable if only one metric

    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        for algorithm in algorithms:
            values = []
            for genre in genres:
                value = average_results[algorithm].get(genre, {}).get(metric, 0)
                values.append(value)
            ax.plot(genres, values, marker='o', label=algorithm)

        ax.legend()

        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_xlabel("Genre")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.set_xticklabels(genres, rotation=45)

    fig.suptitle("Average Scores per Genre for Each Metric", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    axs[0].legend(loc='upper left')
    plt.show()

# --------- Choose algorithm ---------
algorithms = {
    "librosa": librosa_eval,
    "beat_this": beat_this_eval,
    "plp": plp_eval
}
#Copy 1 of the algorithms strings above ↑
#Paste in selected_algorithm to evaluate that algorithm
#selected_algorithm = "librosa"

# --------- MAIN (RUN EVALUATION HERE) ---------
if __name__ == '__main__':
    for genre in genres:
        print(f"\nProcessing genre: {genre}")
        audio_dir = f"Music_files/{genre}"
        annotation_dir = f"Annotated_beats/{genre}"
        wav_paths = sorted(glob(os.path.join(audio_dir, f"{genre}.*.wav")))
        wav_paths = [p.replace("\\", "/") for p in wav_paths]

        #Clear all the average scores every time it switches genre
        f_scores.clear()
        p_scores.clear()
        cemgil_scores.clear()
        goto_scores.clear()
        infogain_scores.clear()
        continuity_scores.clear()

        for selected_algorithm, eval_func in algorithms.items():
            for wav_path in wav_paths[:2]:
                file_id = os.path.splitext(os.path.basename(wav_path))[0].split(".")[-1]
                beat_filename = f"gtzan_{genre}_{file_id}.beats"
                beat_path = os.path.join(annotation_dir, beat_filename)

                if not os.path.exists(beat_path):
                    print(f"Skipping {wav_path}, no beat file found.")
                    continue

                f_average, p_score_average, cemgil_score_average, goto_score_average, infogain_scores_average,continuity_score_average = eval_func(
                    wav_path, beat_path)

                if selected_algorithm not in average_results:
                    average_results[selected_algorithm] = {}

                average_results[selected_algorithm][genre] = {
                    "f_measure": f_average,
                    "p_score": p_score_average,
                    "cemgil": cemgil_score_average,
                    "goto": goto_score_average,
                    "infogain": infogain_scores_average,
                    "continuity": continuity_score_average
                }

            plot_all_metrics(average_results)





