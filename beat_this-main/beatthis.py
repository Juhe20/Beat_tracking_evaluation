from pathlib import Path
from beat_this.inference import File2Beats

audio_path = Path("C:/Users/Jonas/Documents/GitHub/Beat_tracking_evaluation/Music_files/blues/blues.00000.wav")

def beat_this():
    file2beats = File2Beats(checkpoint_path="final0", dbn=False)
    beats, downbeats = file2beats(str(audio_path))
    return beats
    #print("First 20 beats", beats[:20])
    #print("First 20 downbeats", downbeats[:20])