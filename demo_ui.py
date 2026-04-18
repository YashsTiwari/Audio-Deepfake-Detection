"""
demo_ui.py — VoiceAnatomy: Forensic Voice Deepfake Detector
════════════════════════════════════════════════════════════
Run with: streamlit run demo_ui.py

Demo flow:
  1. Upload or record audio
  2. Click "Analyze"
  3. See verdict + per-layer scores + spectrogram + pitch contour
"""

import sys
import os
import time
import warnings
import tempfile
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
from pathlib import Path
from io import BytesIO

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title  = "VoiceAnatomy",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Paths ─────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "aasist"))

SR = 16000

# ── CSS styling ───────────────────────────────────────────────────
st.markdown("""
<style>
.verdict-fake {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    color: white; padding: 20px; border-radius: 12px;
    text-align: center; font-size: 2.5rem; font-weight: bold;
    margin: 10px 0;
}
.verdict-real {
    background: linear-gradient(135deg, #00cc44, #006622);
    color: white; padding: 20px; border-radius: 12px;
    text-align: center; font-size: 2.5rem; font-weight: bold;
    margin: 10px 0;
}
.signal-box {
    background: #1a1a2e; color: #eee;
    padding: 10px 14px; border-radius: 8px;
    margin: 4px 0; font-family: monospace;
    border-left: 4px solid #ff4444;
}
.metric-card {
    background: #16213e; padding: 12px;
    border-radius: 10px; text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Load detector (cached) ────────────────────────────────────────
@st.cache_resource
def load_detector():
    try:
        from detector import VoiceDetector
        det = VoiceDetector(verbose=False)
        return det, None
    except Exception as e:
        return None, str(e)


# ── Feature extraction for visualization ─────────────────────────
def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio, SR


def make_spectrogram_plot(audio: np.ndarray, sr: int,
                           title: str, color: str = "viridis") -> BytesIO:
    """Generate mel spectrogram as image buffer."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8),
                              facecolor='#0e0e1a')
    fig.suptitle(title, color='white', fontsize=13, fontweight='bold')

    # Mel spectrogram
    ax1 = axes[0]
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                       x_axis='time', y_axis='mel',
                                       ax=ax1, cmap=color)
    ax1.set_title("Mel Spectrogram", color='white', fontsize=10)
    ax1.tick_params(colors='white')
    ax1.yaxis.label.set_color('white')
    ax1.xaxis.label.set_color('white')
    plt.colorbar(img, ax=ax1, format='%+2.0f dB')

    # Pitch contour
    ax2 = axes[1]
    ax2.set_facecolor('#0e0e1a')
    f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
    t  = librosa.times_like(f0, sr=sr)
    f0_voiced = np.where(f0 > 60, f0, np.nan)
    ax2.plot(t, f0_voiced, color='#00ff88' if color == 'viridis' else '#ff6666',
             linewidth=1.2, alpha=0.9)
    ax2.set_ylabel("F0 (Hz)", color='white')
    ax2.set_xlabel("Time (s)", color='white')
    ax2.set_title("Pitch Contour (F0)", color='white', fontsize=10)
    ax2.tick_params(colors='white')
    ax2.set_ylim(0, 400)
    ax2.set_facecolor('#111122')
    ax2.spines[:].set_color('#333355')

    # RMS energy (noise floor visualization)
    ax3 = axes[2]
    ax3.set_facecolor('#0e0e1a')
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
    t_r = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
    ax3.fill_between(t_r, rms,
                     color='#4499ff' if color == 'viridis' else '#ff9944',
                     alpha=0.7)
    ax3.plot(t_r, rms,
             color='#88ccff' if color == 'viridis' else '#ffcc88',
             linewidth=0.8)
    ax3.set_ylabel("RMS Energy", color='white')
    ax3.set_xlabel("Time (s)", color='white')
    ax3.set_title("Energy Profile (noise floor in silences)", color='white', fontsize=10)
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#111122')
    ax3.spines[:].set_color('#333355')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                facecolor='#0e0e1a', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


def make_signal_bars(scores: dict) -> BytesIO:
    """Bar chart of layer scores."""
    layers = {
        "Hardcoded\nSignals": scores.get("hardcoded", 0.5),
        "XGBoost\nFeatures": scores.get("xgboost", 0.5),
        "AASIST\nNeural":    scores.get("aasist", 0.5),
        "Watermark\nCheck":  scores.get("watermark", 0.0),
    }

    fig, ax = plt.subplots(figsize=(8, 3), facecolor='#0e0e1a')
    ax.set_facecolor('#111122')

    colors = []
    for v in layers.values():
        if v > 0.7:   colors.append('#ff3333')
        elif v > 0.5: colors.append('#ff9900')
        else:         colors.append('#00cc44')

    bars = ax.barh(list(layers.keys()), list(layers.values()),
                   color=colors, alpha=0.85, height=0.5)

    # Add value labels
    for bar, val in zip(bars, layers.values()):
        ax.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                f'{val:.0%}', va='center', color='white',
                fontsize=11, fontweight='bold')

    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color='#555577', linestyle='--', linewidth=1)
    ax.set_xlabel("Fake probability →", color='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333355')
    ax.set_title("Per-Layer Detection Scores", color='white',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100,
                facecolor='#0e0e1a', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔬 VoiceAnatomy")
        st.markdown("**Forensic Voice Deepfake Detector** — "
                    "4-layer acoustic autopsy system")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Built in 24h · ASVspoof + Custom Analysis")

    st.divider()

    # Load detector
    detector, det_error = load_detector()
    if det_error:
        st.warning(f"Detector loading issue: {det_error}. "
                   f"Running in visualization mode.")

    # ── Input section ─────────────────────────────────────────────
    st.subheader("📤 Input Audio")

    input_method = st.radio("Source", ["Upload file", "Use example"],
                             horizontal=True)

    audio_path = None

    if input_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload WAV or MP3 (10-30 seconds recommended)",
            type=["wav", "mp3", "m4a", "flac"]
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(uploaded.read())
            tmp.flush()
            audio_path = tmp.name
            st.audio(uploaded)

    else:
        # Example files
        examples = {}
        for system in ["xtts", "yourtts", "chatterbox", "xtts_finetune"]:
            p = PROJECT_DIR / "output_voices" / system
            if p.exists():
                wavs = list(p.glob("*/sent1.wav"))
                if wavs:
                    examples[f"FAKE — {system} clone"] = str(wavs[0])

        for p in (PROJECT_DIR / "input_voices").glob("*_converted.wav"):
            examples[f"REAL — {p.stem.replace('_converted','')}"] = str(p)
            break  # just one real example

        if examples:
            choice = st.selectbox("Select example", list(examples.keys()))
            audio_path = examples[choice]
            audio_data, sr = load_audio(audio_path)
            buf = BytesIO()
            sf.write(buf, audio_data[:sr*15], sr, format='WAV')
            buf.seek(0)
            st.audio(buf, format="audio/wav")
        else:
            st.info("No example files found. Upload an audio file.")

    # ── Analyze button ────────────────────────────────────────────
    st.divider()

    if audio_path and st.button("🔍 Analyze Audio",
                                 type="primary", use_container_width=True):
        with st.spinner("Running forensic analysis..."):
            t0 = time.time()

            # Run detector
            if detector:
                result = detector.predict(audio_path)
            else:
                # Fallback — basic feature extraction only
                result = _fallback_predict(audio_path)

            elapsed = time.time() - t0

        # ── Results ───────────────────────────────────────────────
        st.divider()
        st.subheader("🧬 Forensic Autopsy Results")

        label      = result["label"]
        confidence = result["confidence"]
        verdict    = result.get("verdict", f"{label.upper()}")

        # Big verdict box
        css_class = "verdict-fake" if label == "fake" else "verdict-real"
        icon      = "🚨" if label == "fake" else "✅"
        st.markdown(
            f'<div class="{css_class}">'
            f'{icon} {verdict.upper()}<br>'
            f'<span style="font-size:1.2rem">'
            f'Confidence: {confidence:.1%} | Analysis: {elapsed:.1f}s'
            f'</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Per-layer scores ──────────────────────────────────────
        col1, col2 = st.columns([1, 2])

        with col1:
            scores = result.get("scores", {})
            m1, m2 = st.columns(2)
            m1.metric("Hardcoded", f"{scores.get('hardcoded', 0):.0%}")
            m2.metric("XGBoost",   f"{scores.get('xgboost', 0):.0%}")
            m3, m4 = st.columns(2)
            m3.metric("AASIST",    f"{scores.get('aasist', 0):.0%}")
            m4.metric("Watermark", "DETECTED" if scores.get('watermark', 0) > 0.5
                                   else "Not found")

        with col2:
            bar_buf = make_signal_bars(scores)
            st.image(bar_buf, use_container_width=True)

        # ── Triggered signals ──────────────────────────────────────
        signals = result.get("signals", [])
        if signals:
            st.markdown("**🔎 Forensic signals detected:**")
            for sig in signals[:8]:
                st.markdown(
                    f'<div class="signal-box">✗ {sig}</div>',
                    unsafe_allow_html=True
                )

        # ── Key feature values ─────────────────────────────────────
        top_feats = result.get("top_features", {})
        if top_feats:
            st.markdown("<br>**📊 Key acoustic measurements:**",
                        unsafe_allow_html=True)
            feat_cols = st.columns(len(top_feats))
            feat_labels = {
                "breathing_count":   "Breathing\nbursts",
                "snr_ratio":         "SNR\nratio",
                "f0_jitter_local":   "F0 Jitter\n(%)",
                "hf_energy_ratio":   "HF Energy\nratio",
                "shimmer":           "Shimmer",
                "pause_cv":          "Pause\nvariance",
            }
            for i, (feat, val) in enumerate(top_feats.items()):
                feat_cols[i].metric(
                    feat_labels.get(feat, feat),
                    f"{val:.3f}"
                )

        # ── Spectrogram visualization ──────────────────────────────
        st.divider()
        st.subheader("📈 Acoustic Fingerprint")

        audio, sr = load_audio(audio_path)
        # Use first 15 seconds max for display
        audio_disp = audio[:sr * 15]

        spec_buf = make_spectrogram_plot(
            audio_disp, sr,
            title = f"{'⚠️ FAKE AUDIO' if label == 'fake' else '✅ REAL AUDIO'} — Acoustic Analysis",
            color = "hot" if label == "fake" else "viridis"
        )
        st.image(spec_buf, use_container_width=True)

        st.caption(
            "**Top:** Mel spectrogram — bright regions = energy. "
            "Fake audio shows characteristic high-frequency aliasing artifacts. "
            "**Middle:** F0 pitch contour — real voices have natural micro-variations; "
            "synthetic voices are too smooth. "
            "**Bottom:** Energy profile — real voices show consistent room noise "
            "in silence; fake audio has unnaturally dead silence between words."
        )

        # ── Download report ────────────────────────────────────────
        st.divider()
        report = {
            "verdict":    verdict,
            "label":      label,
            "confidence": float(confidence),
            "scores":     {k: float(v) if v else None
                           for k, v in scores.items()},
            "signals":    signals,
            "top_features": {k: float(v) for k, v in top_feats.items()},
        }
        import json
        st.download_button(
            "📥 Download Analysis Report (JSON)",
            data    = json.dumps(report, indent=2),
            file_name = "voiceanatomy_report.json",
            mime    = "application/json",
        )


def _fallback_predict(audio_path: str) -> dict:
    """Basic prediction when full detector not available."""
    try:
        sys.path.insert(0, str(PROJECT_DIR))
        import deep_analysis as da
        audio = da.load_audio(audio_path)
        feats = {}
        feats.update(da.extract_spectral(audio))
        feats.update(da.extract_noise(audio))
        feats.update(da.extract_prosody(audio))

        # Simple heuristic score
        score = 0.5
        signals = []
        if feats.get("breathing_count", 1) == 0:
            score += 0.15
            signals.append("No breathing sounds detected")
        if feats.get("snr_ratio", 1) > 3.0:
            score += 0.15
            signals.append("Dead silence in pauses detected")
        if feats.get("f0_jitter_local", 1) < 0.01:
            score += 0.10
            signals.append("F0 pitch too smooth")
        if feats.get("hf_energy_ratio", 0) > 0.22:
            score += 0.10
            signals.append("High-frequency aliasing detected")

        label = "fake" if score > 0.5 else "real"
        return {
            "label":      label,
            "confidence": min(score, 0.99),
            "verdict":    f"{'FAKE' if label == 'fake' else 'REAL'} — {score:.0%} confidence",
            "signals":    signals,
            "scores": {
                "hardcoded": min(score, 0.99),
                "xgboost":   0.5,
                "aasist":    0.5,
                "watermark": 0.0,
            },
            "top_features": {
                k: feats.get(k, 0.0)
                for k in ["breathing_count", "snr_ratio", "f0_jitter_local",
                          "hf_energy_ratio", "shimmer", "pause_cv"]
            },
        }
    except Exception as e:
        return {
            "label": "unknown", "confidence": 0.5,
            "verdict": f"Analysis error: {e}",
            "signals": [], "scores": {}, "top_features": {},
        }


if __name__ == "__main__":
    main()