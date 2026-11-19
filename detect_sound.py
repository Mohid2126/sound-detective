import numpy as np
import soundfile as sf
from transformers import pipeline

print("Loading the sound AI...")

detector = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused",
)

sound_file = input("Type the sound file name, then press Return: ").strip()

audio, sample_rate = sf.read(sound_file, dtype="float32")

if audio.ndim == 2:
    audio = audio.mean(axis=1)

guesses = detector(
    audio,
    candidate_labels=[
        "a dog barking",
        "rain falling",
        "a doorbell ringing",
        "someone knocking",
    ],
    sampling_rate=sample_rate,
)

print("\nSound Detective's guesses:")
for guess in guesses:
    print(f"{guess['label']}: {guess['score']:.1%}")