"""
ui.py — Hackathon Task 0 — Voice Clone UI
──────────────────────────────────────────────────────────────────────────────
After running generate_clones.py and listening to outputs, paste your chosen
best hyperparams into BEST_PARAMS below. This UI then uses those fixed params
for clean single-output generation.

Usage:
    python ui.py
    Open http://localhost:7860
──────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
import gradio as gr
from pathlib import Path
from datetime import datetime

# ─── PASTE YOUR BEST PARAMS HERE after running generate_clones.py ─────────────
# The filename from outputs/ tells you the config. Example:
#   sent1_temp0.65_spd1.0_topk50.wav  →  temperature=0.65, speed=1.0, top_k=50

BEST_PARAMS = {
    "xtts": {
        "temperature":        0.65,
        "speed":              1.0,
        "top_k":              50,
        "top_p":              0.85,
        "length_penalty":     1.0,
        "repetition_penalty": 10.0,
    },
    "yourtts": {
        "noise_scale":   0.667,
        "noise_scale_w": 0.8,
        "length_scale":  1.0,
    },
    "chatterbox": {
        "exaggeration": 0.5,
        "cfg_weight":   0.5,
        "temperature":  0.8,
    },
}

# ─── Output dir ───────────────────────────────────────────────────────────────

SAVE_DIR = Path("ui_outputs")
SAVE_DIR.mkdir(exist_ok=True)
for sub in ["xtts", "yourtts", "chatterbox"]:
    (SAVE_DIR / sub).mkdir(exist_ok=True)

# ─── Model cache ──────────────────────────────────────────────────────────────

_models = {}

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_xtts():
    if "xtts" not in _models:
        from TTS.api import TTS
        _models["xtts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(get_device())
    return _models["xtts"]

def load_yourtts():
    if "yourtts" not in _models:
        from TTS.api import TTS
        _models["yourtts"] = TTS("tts_models/multilingual/multi-dataset/your_tts").to(get_device())
    return _models["yourtts"]

def load_chatterbox():
    if "chatterbox" not in _models:
        from chatterbox.tts import ChatterboxTTS
        _models["chatterbox"] = ChatterboxTTS.from_pretrained(device=get_device())
    return _models["chatterbox"]

# ─── Audio utilities ──────────────────────────────────────────────────────────

def save_ref_from_gradio(audio_input) -> str:
    """
    Gradio Audio returns (sample_rate, numpy_array).
    Save to disk and return path.
    """
    sr, data = audio_input
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    data = data.astype(np.float32)
    if data.max() > 1.0:
        data = data / 32768.0     # int16 → float
    ref_path = str(SAVE_DIR / "reference_input.wav")
    sf.write(ref_path, data, sr)
    return ref_path

def preprocess(path: str, target_sr: int) -> str:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    out = f"/tmp/ref_{target_sr}.wav"
    torchaudio.save(out, waveform, target_sr)
    return out

def save_output(audio, sr: int, system: str, label: str) -> str:
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().numpy()
    audio = np.array(audio, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    ts    = datetime.now().strftime("%H%M%S")
    path  = SAVE_DIR / system / f"{label}_{ts}.wav"
    sf.write(str(path), audio, sr)
    return str(path)

# ─── Generation ───────────────────────────────────────────────────────────────

def generate_all(audio_input, text: str, progress=gr.Progress()):
    """
    Main function called by Gradio.
    Returns: (xtts_path, yourtts_path, chatterbox_path, log_text)
    """
    if audio_input is None:
        return None, None, None, "❌ Please provide reference audio first."

    if not text.strip():
        return None, None, None, "❌ Please enter text to synthesise."

    logs   = []
    ref_path = save_ref_from_gradio(audio_input)
    logs.append(f"Reference saved: {ref_path}")
    label = text[:30].replace(" ", "_").replace(".", "")

    results = {}

    # XTTS v2
    progress(0.1, desc="Loading XTTS v2...")
    try:
        t0    = time.time()
        model = load_xtts()
        ref   = preprocess(ref_path, 22050)
        cfg   = BEST_PARAMS["xtts"]
        progress(0.2, desc="XTTS v2 generating...")
        audio = model.tts(
            text               = text,
            speaker_wav        = ref,
            language           = "en",
            temperature        = cfg["temperature"],
            speed              = cfg["speed"],
            top_k              = cfg["top_k"],
            top_p              = cfg["top_p"],
            length_penalty     = cfg["length_penalty"],
            repetition_penalty = cfg["repetition_penalty"],
        )
        path = save_output(np.array(audio), 24000, "xtts", label)
        results["xtts"] = path
        logs.append(f"✅ XTTS v2:       {time.time()-t0:.1f}s → {Path(path).name}")
    except Exception as e:
        results["xtts"] = None
        logs.append(f"❌ XTTS v2 failed: {e}")

    # YourTTS
    progress(0.4, desc="Loading YourTTS...")
    try:
        t0    = time.time()
        model = load_yourtts()
        ref   = preprocess(ref_path, 16000)
        cfg   = BEST_PARAMS["yourtts"]
        progress(0.55, desc="YourTTS generating...")
        audio = model.tts(
            text          = text,
            speaker_wav   = ref,
            language      = "en-us",
            noise_scale   = cfg["noise_scale"],
            noise_scale_w = cfg["noise_scale_w"],
            length_scale  = cfg["length_scale"],
        )
        path = save_output(np.array(audio), 16000, "yourtts", label)
        results["yourtts"] = path
        logs.append(f"✅ YourTTS:        {time.time()-t0:.1f}s → {Path(path).name}")
    except Exception as e:
        results["yourtts"] = None
        logs.append(f"❌ YourTTS failed: {e}")

    # Chatterbox
    progress(0.7, desc="Loading Chatterbox...")
    try:
        t0    = time.time()
        model = load_chatterbox()
        ref   = preprocess(ref_path, 16000)
        cfg   = BEST_PARAMS["chatterbox"]
        progress(0.85, desc="Chatterbox generating...")
        wav_tensor = model.generate(
            text              = text,
            audio_prompt_path = ref,
            exaggeration      = cfg["exaggeration"],
            cfg_weight        = cfg["cfg_weight"],
            temperature       = cfg["temperature"],
        )
        path = save_output(wav_tensor, 22050, "chatterbox", label)
        results["chatterbox"] = path
        logs.append(f"✅ Chatterbox:     {time.time()-t0:.1f}s → {Path(path).name}")
    except Exception as e:
        results["chatterbox"] = None
        logs.append(f"❌ Chatterbox failed: {e}")

    progress(1.0, desc="Done.")
    log_text = "\n".join(logs)
    print("\n" + log_text)  # also print to terminal

    return results["xtts"], results["yourtts"], results["chatterbox"], log_text


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Voice Clone — Task 0", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Voice Clone Pipeline — Task 0")
        gr.Markdown(
            "Provide your reference voice (upload or record), enter text, "
            "and generate clones using XTTS v2, YourTTS, and Chatterbox. "
            "All outputs auto-saved to `ui_outputs/`."
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 1. Reference voice")
                audio_input = gr.Audio(
                    sources = ["upload", "microphone"],
                    type    = "numpy",
                    label   = "Upload WAV or record live (10–30s, quiet room)",
                )

                gr.Markdown("### 2. Text to synthesise")
                text_input = gr.Textbox(
                    value       = "The quick brown fox jumps over the lazy dog.",
                    label       = "Enter any sentence",
                    lines       = 3,
                    placeholder = "Type what you want your clone to say...",
                )

                gr.Markdown("### 3. Generate")
                generate_btn = gr.Button("Generate all 3 clones", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("**Current best params** (edit `BEST_PARAMS` in `ui.py` after sweep):")
                params_display = gr.JSON(value=BEST_PARAMS, label="Active hyperparams")

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Outputs")

                gr.Markdown("**XTTS v2** — high quality, diffusion-based")
                xtts_out = gr.Audio(label="XTTS v2 clone", type="filepath")

                gr.Markdown("**YourTTS** — medium quality, flow-matching")
                yourtts_out = gr.Audio(label="YourTTS clone", type="filepath")

                gr.Markdown("**Chatterbox** — expressive, flow-matching")
                chatterbox_out = gr.Audio(label="Chatterbox clone", type="filepath")

                log_out = gr.Textbox(label="Generation log", lines=6, interactive=False)

        generate_btn.click(
            fn      = generate_all,
            inputs  = [audio_input, text_input],
            outputs = [xtts_out, yourtts_out, chatterbox_out, log_out],
        )

    return demo


if __name__ == "__main__":
    print("="*55)
    print("  Voice Clone UI — Task 0")
    print(f"  Device: {get_device().upper()}")
    print(f"  Outputs → {SAVE_DIR.resolve()}")
    print("  Models load on first generation click.")
    print("="*55)
    demo = build_ui()
    demo.launch(server_port=7860, share=False)
