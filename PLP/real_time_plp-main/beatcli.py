"""
Module Name: beatcli.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: A Real-Time Beat Tracking Dashboard for the Terminal.
License:     MIT License (https://opensource.org/licenses/MIT)
"""

import argparse
import threading

import numpy as np
import sounddevice as sd
from pythonosc import udp_client
import soundfile as sf
from realtimeplp import RealTimeBeatTracker
import asyncio
import datetime
import librosa

#Loads audio file if false, uses microphone input if true
use_microphone = False
start_time = datetime.datetime.now()

# TODO: Add --learn mode for mapping OSC signals.
# Output all OSC channels in a row with 1 second delay between.

# Argparse
parser = argparse.ArgumentParser(description="Beat (C)ommand (L)ine (I)nterface.")
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
parser.add_argument(
    "--device",
    metavar="ID",
    type=int,
    help="(%(default)s) device id for sounddevice input",
)
parser.add_argument(
    "--channel",
    default=2,
    metavar="NUMBER",
    type=int,
    help="(%(default)s) channel number for sounddevice input",
)
parser.add_argument(
    "--samplerate",
    default=44100,
    metavar="FS",
    type=int,
    help="(%(default)s) samplerate for sounddevice",
)
parser.add_argument(
    "--blocksize",
    default=512,
    metavar="SAMPLES",
    type=int,
    help="(%(default)s) blocksize for sounddevice",
)
parser.add_argument(
    "--tempo",
    nargs=2,
    metavar=("LOW", "HIGH"),
    default=[30, 240],
    type=int,
    help="(%(default)s) tempo range in BPM",
)
parser.add_argument(
    "--lookahead",
    default=0,
    metavar="FRAMES",
    type=int,
    help="(%(default)s) number of frames (samplerate / blocksize) to lookahead"
    " in time and get the next beat earlier to compensate for latency",
)
parser.add_argument(
    "--kernel",
    default=6,
    metavar="SIZE",
    type=int,
    help="(%(default)s) kernel size in seconds",
)
parser.add_argument(
    "--ip", default="127.0.0.1", type=str, help="(%(default)s) ip address for OSC client"
)
parser.add_argument(
    "--port", default=5005, type=int, help="(%(default)s) port for OSC client"
)
parser.add_argument(
    "--file",
    type=str,
    default=None,
    help="Path to an audio file to process instead of using the microphone",
)
args = parser.parse_args()
# List devices
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
low, high = args.tempo
if high <= low:
    parser.error("HIGH must be greater than LOW")


# Sounddevice Settings
sd.default.blocksize = args.blocksize
sd.default.samplerate = args.samplerate
sd.default.latency = 0  # in seconds
sd.default.channels = args.channel  # number of channels (both in and out)
sd.default.device = (args.device, args.device)  # (in_device_id, out_device_id)
device = sd.query_devices(args.device, "input")

# OSC Client
client = udp_client.SimpleUDPClient(args.ip, args.port)
print(sd.query_devices())
beat = RealTimeBeatTracker.from_args(
    N=2 * args.blocksize,
    H=args.blocksize,
    samplerate=args.samplerate,
    N_time=args.kernel,
    Theta=np.arange(low, high + 1, 1),
    lookahead=args.lookahead,
)

stability = []
tempo = []
time = []
i = 0
beat_times_list = []
start_time = None

def callback(indata, _frames, _time, status):
    global start_time, beat_times_list
    #Timer stuff to count down from 30 seconds
    if start_time is None:
        start_time = datetime.datetime.now()
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()

    #Checks for beats until it reaches 30 seconds
    if elapsed_seconds < 30:
        beat_detected = beat.process(indata[:, args.channel - 1])
        #Append beat to list of beat times
        if beat_detected:
            beat_time = float(elapsed_seconds)
            beat_times_list.append(beat_time)
            print(
                f"time={beat_time:.3f} | tempo={beat.plp.current_tempo} | stability={beat.cs.beta_confidence:.3f}"
            )
    else:
        #After 30 seconds, stop the audio capture
        raise sd.CallbackStop


#Capture audio function to get mic input for beat times
async def capture_audio(audio_path=None):
    global beat_times_list, start_time
    print("Capturing audio for 30 seconds...")
    beat_times_list = []
    start_time = None

    if not use_microphone:
        # Offline mode: read from file
        data, sr = sf.read(audio_path)
        if sr != args.samplerate:
            print(f"Resampling from {sr} Hz to {args.samplerate} Hz...")
            if data.ndim == 1:
                data = librosa.resample(data, orig_sr=sr, target_sr=args.samplerate)
            else:
                data = librosa.resample(data.T, orig_sr=sr, target_sr=args.samplerate).T
            sr = args.samplerate

        # Process in chunks (simulate real-time)
        num_blocks = int(len(data) / args.blocksize)
        for i in range(num_blocks):
            block = data[i * args.blocksize:(i + 1) * args.blocksize]
            if block.shape[0] != args.blocksize:
                break
            beat_detected = beat.process(block[:, args.channel - 1] if data.ndim > 1 else block)
            elapsed_seconds = i * args.blocksize / args.samplerate
            if beat_detected:
                beat_times_list.append(float(elapsed_seconds))
                print(
                    f"time={elapsed_seconds:.3f} | tempo={beat.plp.current_tempo} | stability={beat.cs.beta_confidence:.3f}"
                )

        print("Audio file processing finished")
        return beat_times_list

    else:
        # Microphone input mode
        try:
            with sd.InputStream(callback=callback, samplerate=args.samplerate):
                await asyncio.sleep(30)
        except sd.CallbackStop:
            print("Microphone input finished")

        return beat_times_list





#async def main():
    #await capture_audio_periodically()

#if __name__ == "__main__":
    #asyncio.run(main())

