import librosa
import soundfile as sf
from pathlib import Path

def load_wav(file_path):
    """Load WAV file with consistent sample rate"""
    y, sr = sf.read(file_path)
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    return y, 22050

def validate_wav(file_path):
    """Check if file exists and is WAV format"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != '.wav':
        raise ValueError("Only WAV files are supported")
    return str(path)