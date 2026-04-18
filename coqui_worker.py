"""
workers/coqui_worker.py
Called by generate_clones.py via subprocess using task0_coqui venv.
Usage: python workers/coqui_worker.py --model xtts --text "..." --ref ref.wav --out out.wav --cfg '{...}'
"""

import argparse
import json
import sys
import warnings
import numpy as np
import torch
import torchaudio
import soundfile as sf

warnings.filterwarnings("ignore")


def preprocess(path: str, target_sr: int) -> str:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    out = f"/tmp/ref_coqui_{target_sr}.wav"
    torchaudio.save(out, waveform, target_sr)
    return out


def save(audio, sr: int, path: str):
    audio = np.array(audio, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    sf.write(path, audio, sr)


def run_xtts(text: str, ref_path: str, cfg: dict, out_path: str):
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    ref    = preprocess(ref_path, 22050)
    audio  = model.tts(
        text               = text,
        speaker_wav        = ref,
        language           = "en",
        temperature        = cfg.get("temperature", 0.65),
        speed              = cfg.get("speed", 1.0),
        top_k              = cfg.get("top_k", 50),
        top_p              = cfg.get("top_p", 0.85),
        length_penalty     = cfg.get("length_penalty", 1.0),
        repetition_penalty = cfg.get("repetition_penalty", 10.0),
    )
    save(audio, 24000, out_path)
    print(f"SAVED:{out_path}", flush=True)


def run_yourtts(text: str, ref_path: str, cfg: dict, out_path: str):
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
    ref    = preprocess(ref_path, 16000)
    audio  = model.tts(
        text          = text,
        speaker_wav   = ref,
        language      = "en-us",
        noise_scale   = cfg.get("noise_scale", 0.667),
        noise_scale_w = cfg.get("noise_scale_w", 0.8),
        length_scale  = cfg.get("length_scale", 1.0),
    )
    save(audio, 16000, out_path)
    print(f"SAVED:{out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=["xtts", "yourtts"])
    parser.add_argument("--text",   required=True)
    parser.add_argument("--ref",    required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--cfg",    default="{}")
    args = parser.parse_args()

    cfg = json.loads(args.cfg)

    if args.model == "xtts":
        run_xtts(args.text, args.ref, cfg, args.out)
    elif args.model == "yourtts":
        run_yourtts(args.text, args.ref, cfg, args.out)
