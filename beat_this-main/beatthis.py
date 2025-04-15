from pathlib import Path
from beat_this.inference import File2Beats

script_dir = Path(__file__).resolve().parent.parent



def beat_this(audio_path):
    file2beats = File2Beats(checkpoint_path="final0", dbn=False)
    #Finds beats and downbeats from a beat this method
    beats, downbeats = file2beats(str(audio_path))
    #Can return downbeats as well, but we aren't using them for now
    return beats