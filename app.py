from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Echo Worlds · Sound Lab", page_icon="🌌", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--ink:#f4fff9;--cyan:#62f2ce;--violet:#8ecfff;--muted:#a7c9c3}
.stApp{background:radial-gradient(ellipse at 78% 0%,#163b45,#0b202b 44%,#07131c 100%);color:var(--ink)}
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif}[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:38px;border:1px solid #62f2ce55;border-radius:28px;background:linear-gradient(115deg,#102d36ed,#163d42e8);box-shadow:0 20px 80px #0008,0 0 50px #62f2ce18;position:relative;overflow:hidden;margin-bottom:24px}
.hero:after{content:'✳';position:absolute;right:45px;top:-55px;font-size:240px;color:#62f2ce22;animation:spin 65s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.eyebrow,.kicker{font:500 11px 'DM Mono',monospace;letter-spacing:.17em;text-transform:uppercase;color:var(--cyan)}
.hero h1{font-size:clamp(44px,7vw,78px);letter-spacing:-.07em;line-height:.92;margin:17px 0;color:#f4f5ff}.hero p{font-size:17px;color:#c4c8dc;max-width:700px}
[data-testid="stMetric"]{background:#141a28;border:1px solid #62f2ce33;padding:16px;border-radius:16px}[data-testid="stMetricValue"]{color:var(--cyan)}
.stButton>button{border-radius:12px;min-height:45px;border:1px solid #62f2ce77;background:#172433;color:white;font-weight:600}.stButton>button:hover{border-color:var(--cyan);box-shadow:0 0 20px #62f2ce55}.stButton>button[kind="primary"]{background:linear-gradient(100deg,#62f2ce,#8ecfff);color:#07121c;border:0}
[data-testid="stExpander"],[data-testid="stAudioInput"]{background:#121927;border:1px solid #39445a;border-radius:15px} [data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#62f2ce,#8ecfff)}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


def read_recording(uploaded_audio):
    audio, sample_rate = sf.read(BytesIO(uploaded_audio.getvalue()), dtype="float32")
    if audio.ndim == 2: audio = audio.mean(axis=1)
    return audio, sample_rate


def analyze_audio(audio: np.ndarray, sample_rate: int):
    if len(audio) < int(sample_rate * .25): raise ValueError("That clip is too short. Please record at least one second.")
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
    sr = 22050
    rms = librosa.feature.rms(y=audio)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), [centroid.mean(), centroid.std(), zcr.mean(), rms.mean(), rms.std()]]).astype("float32")
    duration = len(audio) / sr
    energy = float(np.mean(rms))
    brightness = float(np.clip(np.mean(centroid) / 5000, 0, 1))
    texture = float(np.clip(np.mean(zcr) * 5, 0, 1))
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    return features, {"audio": audio, "duration": duration, "energy": energy, "brightness": brightness, "texture": texture, "tempo": float(np.asarray(tempo).reshape(-1)[0])}


def make_world(data, title="Your sound, imagined"):
    audio = data["audio"]
    step = max(1, len(audio)//1000)
    wave = np.clip(audio[::step], -1, 1)
    encoded = base64.b64encode(wave.astype("float32").tobytes()).decode("ascii")
    energy = min(1, data["energy"] * 5)
    bright = data["brightness"]
    texture = data["texture"]
    hue = 150 + bright*75
    html = f'''<div id="world"><canvas id="sky"></canvas><div class="label"><small>ECHO WORLD · {data['duration']:.1f} SECONDS</small><h2>{title}</h2><p>brightness {bright:.2f} &nbsp; · &nbsp; energy {energy:.2f} &nbsp; · &nbsp; texture {texture:.2f}</p></div><div class="hint">A BIOLUMINESCENT MAP OF YOUR SOUND</div></div>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#07151d;color:#edfff9;font-family:system-ui}}#world{{height:470px;position:relative;overflow:hidden;border-radius:24px;border:1px solid #62f2ce55;background:radial-gradient(ellipse at 50% 55%,hsl({hue} 68% 19%),#081b25 72%);box-shadow:inset 0 0 100px #0009}}canvas{{position:absolute;inset:0;width:100%;height:100%}}.label{{position:absolute;top:25px;left:30px;text-shadow:0 2px 18px #000}}small{{font:11px monospace;letter-spacing:.18em;color:#70f5d1}}h2{{font-size:30px;margin:10px 0 5px;letter-spacing:-.04em}}p{{color:#c1cce0;margin:0;font:12px monospace}}.hint{{position:absolute;bottom:18px;right:22px;color:#7f94ad;font:10px monospace;letter-spacing:.14em}}</style>
<script>
const c=document.getElementById('sky'),ctx=c.getContext('2d');let w,h,parts=[],t=0;function size(){{let r=c.getBoundingClientRect(),d=devicePixelRatio||1;w=r.width;h=r.height;c.width=w*d;c.height=h*d;ctx.setTransform(d,0,0,d,0,0);parts=Array.from({{length:Math.floor(w/5)}},()=>({{x:Math.random()*w,y:Math.random()*h,r:Math.random()*2+0.5,p:Math.random()*6.28,s:Math.random()*0.7+0.2}}))}}size();new ResizeObserver(size).observe(c);
const raw=atob('{encoded}'),view=new DataView(Uint8Array.from(raw,x=>x.charCodeAt(0)).buffer),wave=[];for(let i=0;i<view.byteLength;i+=4)wave.push(view.getFloat32(i,true));
function frame(){{t+=.016;ctx.clearRect(0,0,w,h);let centerY=h*.57;ctx.beginPath();for(let x=0;x<w;x++){{let idx=Math.floor(x/w*wave.length),v=wave[idx]||0,amp=({energy:.3f})*95+12,y=centerY+Math.sin(x*.012+t*1.2)*12+v*amp;if(x===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}}ctx.strokeStyle='hsla({hue:.0f},100%,72%,.85)';ctx.lineWidth=2;ctx.shadowColor='#62f2ce';ctx.shadowBlur=14;ctx.stroke();ctx.shadowBlur=0;for(let p of parts){{let pulse=.5+.5*Math.sin(t*p.s+p.p);ctx.beginPath();ctx.arc(p.x,p.y,p.r+pulse*({texture:.2f})*3,0,Math.PI*2);ctx.fillStyle=`hsla({hue:.0f},90%,${{65+pulse*25}}%,${{.18+pulse*.55}})`;ctx.fill()}}requestAnimationFrame(frame)}}frame();</script>'''
    world_path = Path("/tmp/echo_world.html")
    world_path.write_text(html, encoding="utf-8")
    st.iframe(world_path, height=490)


def examples(): return st.session_state.setdefault("examples", [])

def add_example(label, recording):
    try:
        audio, sr = read_recording(recording); features, _ = analyze_audio(audio, sr)
        examples().append({"label": label, "features": features, "audio": recording.getvalue()})
        st.success(f"Added example {sum(x['label']==label for x in examples())} for {label}.")
    except Exception as exc: st.error(f"I couldn't use that recording: {exc}")

st.markdown('''<div class="hero"><div class="eyebrow">BIOLUMINESCENT SOUND LAB</div><h1>Echo<br>Worlds<span style="color:#62f2ce">.</span></h1><p>Every sound has a shape, a pulse, and a color. Explore the world hidden inside a recording, then teach your own tiny AI to recognize sounds.</p></div>''', unsafe_allow_html=True)
st.markdown('<div class="kicker">01 / Enter a sound world</div>', unsafe_allow_html=True)
st.write("Record any sound right here. When you have a recording, create its world below.")
world_recording = st.audio_input("Record a sound for your world", key="world_recording")
if world_recording:
    try:
        audio, sr = read_recording(world_recording); _, data = analyze_audio(audio, sr)
        st.audio(world_recording)
        if st.button("🌌 Create my sound world", type="primary", use_container_width=True, key="create_world"):
            st.session_state["world_data"] = data
        if "world_data" in st.session_state:
            make_world(st.session_state["world_data"], "Your recorded sound")
            data = st.session_state["world_data"]
        cols = st.columns(4)
        cols[0].metric("Duration", f"{data['duration']:.1f}s")
        cols[1].metric("Energy", f"{data['energy']:.3f}")
        cols[2].metric("Brightness", f"{data['brightness']:.2f}")
        cols[3].metric("Pulse", f"{data['tempo']:.0f} BPM")
        st.caption("The moving signal follows the recording’s waveform. Color, glow, and particles respond to its measured brightness, loudness, and texture.")
    except Exception as exc: st.error(f"I couldn't use that recording: {exc}")
else:
    st.info("Tap the microphone, record a sound, then create its world. Try a clap, a hum, a knock, or rain against a window.")

st.divider()
st.markdown('<div class="kicker">02 / Teach the sound engine</div>', unsafe_allow_html=True)
with st.expander("How this learning lab works"):
    st.write("Each clip becomes numerical clues about its tone and texture. Record a few examples of two sounds, train a small model on your Mac, and see how it classifies a fresh clip. The visual world is independent of those labels, so you can explore any sound.")
a,b=st.columns(2,gap="large")
with a:
    st.markdown("### Cyan signal")
    la=st.text_input("Sound class A",value="Clap",key="la").strip() or "Sound A"
    ra=st.audio_input(f"Record {la}",key="ra")
    if ra and st.button(f"Add {la}",key="aa",use_container_width=True):add_example(la,ra)
with b:
    st.markdown("### Violet signal")
    lb=st.text_input("Sound class B",value="Knock",key="lb").strip() or "Sound B"
    rb=st.audio_input(f"Record {lb}",key="rb")
    if rb and st.button(f"Add {lb}",key="ab",use_container_width=True):add_example(lb,rb)
items=examples(); labels=[la,lb]; counts={x:sum(i["label"]==x for i in items) for x in labels}
st.markdown('<div class="kicker">03 / Grow your training set</div>',unsafe_allow_html=True)
cols=st.columns(2)
for col,label in zip(cols,labels):
    with col:
        st.metric(f"{label} examples",counts[label])
        for item in [i for i in items if i["label"]==label]: st.audio(item["audio"],format="audio/wav")
if st.button("✦ Train my sound engine",type="primary",use_container_width=True):
    if la==lb: st.error("Choose two different sound names.")
    elif min(counts.values())<3: st.warning("Add at least 3 examples of each sound first.")
    else:
        X=np.stack([i["features"] for i in items if i["label"] in labels]); y=np.array([i["label"] for i in items if i["label"] in labels])
        calibration_folds = min(3, min(counts.values()))
        calibration = CalibratedClassifierCV(SVC(kernel="rbf",class_weight="balanced"), cv=StratifiedKFold(n_splits=calibration_folds, shuffle=True, random_state=42), ensemble=False)
        model=make_pipeline(StandardScaler(),calibration); model.fit(X,y)
        st.session_state["model"]=model; st.session_state["model_labels"]=labels.copy(); st.success("Sound engine trained. Try a new recording below.")
        if min(counts.values())>=5:
            score=cross_val_score(make_pipeline(StandardScaler(),SVC(kernel="rbf",class_weight="balanced")),X,y,cv=3)
            st.write(f"Held-out practice score: **{score.mean():.0%}** (rough check with a small dataset).")
if "model" in st.session_state:
    st.markdown('<div class="kicker">04 / Discover and classify</div>',unsafe_allow_html=True)
    test=st.audio_input("Record a mystery sound",key="mystery")
    if test and st.button("Reveal its sound world + AI guess",use_container_width=True):
        try:
            audio,sr=read_recording(test); features,data=analyze_audio(audio,sr); model=st.session_state["model"]
            ranking=sorted(zip(model.classes_,model.predict_proba([features])[0]),key=lambda x:x[1],reverse=True)
            make_world(data,"Mystery signal")
            st.subheader(f"AI’s leading guess: {ranking[0][0]}")
            for label,score in ranking: st.write(f"**{label}** · {score:.0%}"); st.progress(float(score))
            st.caption("Scores compare only the two sounds in your training set; they are not certainty about every possible sound.")
        except Exception as exc: st.error(f"I couldn't analyze that recording: {exc}")
st.divider();st.caption("ECHO WORLDS · Your recordings stay in this browser session on this Mac. Refreshing clears your training examples.")
