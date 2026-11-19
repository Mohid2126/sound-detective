# Sound Detective

A local sound-learning lab built with Python and Streamlit. Record examples of two sounds, train a small classifier on your own recordings, then test it with a fresh recording.

## Features

- Record sounds directly in the browser
- Build a labeled example set for two sound classes
- Extract audio features (MFCC, spectral centroid, zero-crossing rate, and loudness)
- Train a support-vector machine classifier
- Compare a fresh recording and inspect its waveform

## Setup

Requires Python 3.10 or newer and a working microphone. From this folder, create and activate an environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address Streamlit prints. Name two sounds, record at least three examples of each (five or more varied examples is better), train, and then record a fresh test clip. Try to keep the microphone distance and room similar.

## How the learning works

The app summarizes each recording into numerical clues about its frequency texture and loudness. A support-vector machine learns a boundary between the two labeled sound groups. This is a small educational model: it only chooses between the two sounds you teach it, and confidence scores are relative to those choices. With few or similar examples, it can make mistakes.

Examples and the trained model live in Streamlit's session memory and clear when the page is refreshed. Recordings are processed by the local app and are not uploaded to a hosted service by this code.

## Project structure

- `app.py` — interactive Sound Detective app
- `requirements.txt` — Python packages

## License

MIT. See [LICENSE](LICENSE).
