from beat_this.inference import File2Beats

audio_path = "/Music_files/Blues/blue.00000.wav"

file2beats = File2Beats(checkpoint_path="final0", dbn=False)
beats, downbeats = file2beats(audio_path)

print("First 20 beats", beats[:20])
print("First 20 downbeats", downbeats[:20])