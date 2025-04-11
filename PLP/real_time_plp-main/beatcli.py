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

from realtimeplp import RealTimeBeatTracker
import asyncio
import datetime

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

# Global Variables
stability = []
tempo = []
time = []
i = 0
beat_times_list = []
start_time = None

def callback(indata, _frames, _time, status):
    global start_time, beat_times_list

    if start_time is None:
        start_time = datetime.datetime.now()

    elapsed_time = datetime.datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()

    if elapsed_seconds < 30:
        beat_detected = beat.process(indata[:, args.channel - 1])

        if beat_detected:
            beat_time = float(elapsed_seconds)
            beat_times_list.append(beat_time)

            print(
                f"time={beat_time:.3f} | tempo={beat.plp.current_tempo} | stability={beat.cs.beta_confidence:.3f}"
            )
    else:
        # After 30 seconds, stop the stream by raising an exception
        raise sd.CallbackStop


# Simulate a function that captures and processes audio
async def capture_audio():
    global beat_times_list, start_time

    print("Capturing audio for 30 seconds...")

    beat_times_list = []
    start_time = None

    loop = asyncio.get_event_loop()

    # Create the stream and start it
    try:
        with sd.InputStream(callback=callback, samplerate=args.samplerate):
            await asyncio.sleep(30)  # This won't actually block; stream will stop from callback
    except sd.CallbackStop:
        print("Recording finished after 30 seconds.")

    return beat_times_list

async def capture_audio_periodically():
    while True:
        await capture_audio()  # Capture and process audio
        asyncio.create_task(capture_audio())
        await asyncio.sleep(5)   # Wait 30 seconds before checking again



#async def main():
    #await capture_audio_periodically()

#if __name__ == "__main__":
    #asyncio.run(main())

