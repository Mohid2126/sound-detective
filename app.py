from __future__ import annotations

from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Sound Detective · Sound Lab", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#f4f0e8; --mint:#b7f36b; }
.stApp { background:radial-gradient(ellipse at 82% 0%,#252447 0%,#11141c 42%,#0d1016 100%); color:var(--ink); }
html,body,[class*="css"] { font-family:'Space Grotesk',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1120px; padding-top:2.6rem; padding-bottom:4rem; }
.hero { border:1px solid #41405c; border-radius:26px; padding:34px 38px; margin:4px 0 26px; background:linear-gradient(120deg,rgba(48,48,79,.88),rgba(23,28,39,.82)); box-shadow:0 22px 70px #0005; position:relative; overflow:hidden; }
.hero:after { content:'〰'; position:absolute; right:25px; top:0; color:#b7f36b22; font-size:150px; line-height:1; }
.eyebrow,.section-kicker { color:var(--mint); font:500 11px 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { font-size:clamp(40px,6vw,68px); letter-spacing:-.06em; line-height:.98; margin:14px 0; color:var(--ink); }
.hero p { color:#c2c2cb; font-size:17px; max-width:650px; margin:0; }
[data-testid="stMetric"] { background:#171b24; border:1px solid #303744; padding:18px 20px; border-radius:18px; }
[data-testid="stMetricLabel"] { color:#b6b3c0; } [data-testid="stMetricValue"] { color:var(--mint); }
[data-testid="stAudioInput"] { background:#171b24; border:1px dashed #55566c; padding:14px; border-radius:15px; }
.stButton>button { border-radius:12px; min-height:46px; font-weight:600; }
.stButton>button[kind="primary"] { background:var(--mint); color:#15190e; border:0; }
[data-testid="stExpander"] { background:#171b24; border:1px solid #303744; border-radius:15px; }
[data-testid="stProgress"] > div > div { background:var(--mint); }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def read_recording(uploaded_audio):
    audio, sample_rate = sf.read(BytesIO(uploaded_audio.getvalue()), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sample_rate

def audio_features(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if len(audio) < int(sample_rate * 0.25):
        raise ValueError("That recording is very short. Record for at least one second.")
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
    sr = 22050
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    spectral = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    rms = librosa.feature.rms(y=audio)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [spectral.mean(), spectral.std(), zcr.mean(), rms.mean(), rms.std()]]).astype("float32")

def get_examples():
    return st.session_state.setdefault("examples", [])

def add_example(category: str, recording):
    try:
        audio, sr = read_recording(recording)
        features = audio_features(audio, sr)
        get_examples().append({"label": category, "features": features, "audio": recording.getvalue()})
        st.success(f"Added example {sum(x['label'] == category for x in get_examples())} for {category}.")
    except Exception as exc:
        st.error(f"I couldn't use that recording: {exc}")

st.markdown("""
<div class="hero"><div class="eyebrow">FIELD NOTES · SOUND LAB 01</div>
<h1>Sound<br>Detective<span style="color:#b7f36b">.</span></h1>
<p>Teach your Mac to recognize sounds from your world. Record a few clues, train your model, then see if it cracks a fresh case.</p></div>
""", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Build your case file</div>', unsafe_allow_html=True)
with st.expander("How does the sound learner work?"):
    st.write("Each recording becomes sound clues about its texture, brightness, and loudness. The model finds patterns in those clues. Use varied examples recorded in similar conditions; its scores compare only the two sounds you teach it.")
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### 🟢 Sound A")
    label_a = st.text_input("Name this sound", value="Clap", key="label_a").strip() or "Sound 1"
    rec_a = st.audio_input(f"Record an example of {label_a}", key="record_a")
    if rec_a and st.button(f"Add {label_a} example", key="add_a", use_container_width=True): add_example(label_a, rec_a)
with right:
    st.markdown("### 🟣 Sound B")
    label_b = st.text_input("Name this sound", value="Knock", key="label_b").strip() or "Sound 2"
    rec_b = st.audio_input(f"Record an example of {label_b}", key="record_b")
    if rec_b and st.button(f"Add {label_b} example", key="add_b", use_container_width=True): add_example(label_b, rec_b)
examples = get_examples()
labels = [label_a, label_b]
counts = {label: sum(item["label"] == label for item in examples) for label in labels}
st.markdown('<div class="section-kicker">02 / Gather evidence</div>', unsafe_allow_html=True)
st.subheader("Your case file")
count_cols = st.columns(2)
for col, label in zip(count_cols, labels):
    with col:
        st.metric(label, counts[label], help="Try for five or more different examples of each sound.")
        for item in [x for x in examples if x["label"] == label]: st.audio(item["audio"], format="audio/wav")
if st.button("🧠 Train my sound learner", type="primary", use_container_width=True):
    if label_a == label_b: st.error("Give the two sound types different names first.")
    elif min(counts.values()) < 3: st.warning("Record at least 3 examples of each sound before training. More varied examples usually help.")
    else:
        X = np.stack([item["features"] for item in examples if item["label"] in labels])
        y = np.array([item["label"] for item in examples if item["label"] in labels])
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced"))
        model.fit(X, y)
        st.session_state["trained_model"] = model
        st.session_state["trained_labels"] = labels.copy()
        st.success("Your learner is trained. Record a fresh sound below to test it.")
        if min(counts.values()) >= 5:
            scores = cross_val_score(make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")), X, y, cv=3)
            st.write(f"Practice-set check: **{scores.mean():.0%}** average across held-out folds.")
            st.caption("With few clips, this is only a rough learning check—not a guarantee on new sounds.")
if "trained_model" in st.session_state:
    st.divider()
    st.markdown('<div class="section-kicker">03 / Crack a new case</div>', unsafe_allow_html=True)
    st.subheader("Put it to the test")
    test_audio = st.audio_input("Record a new sound for the model to identify", key="test_audio")
    if test_audio and st.button("🔍 Make a guess", use_container_width=True):
        try:
            audio, sr = read_recording(test_audio)
            features = audio_features(audio, sr)
            model = st.session_state["trained_model"]
            probabilities = model.predict_proba([features])[0]
            ranking = sorted(zip(model.classes_, probabilities), key=lambda pair: pair[1], reverse=True)
            st.success(f"The leading clue points to: {ranking[0][0]}")
            st.write("### Confidence split")
            for label, score in ranking:
                st.write(f"**{label}** · {score:.0%}")
                st.progress(float(score))
            st.caption("Scores compare only the two sounds you taught it; they are not certainty about every possible sound.")
            st.write("Sound shape · waveform")
            step = max(1, len(audio) // 1200)
            st.line_chart(audio[::step])
        except Exception as exc: st.error(f"I couldn't analyze that recording: {exc}")
st.divider()
st.caption("LOCAL CASE FILE · Recordings stay on this Mac during this session. Refreshing the page clears the case file.")
