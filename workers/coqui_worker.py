import argparse, json, warnings
import numpy as np
import torch, torchaudio, soundfile as sf
warnings.filterwarnings("ignore")

def preprocess(path, target_sr):
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    out = f"/tmp/ref_coqui_{target_sr}.wav"
    torchaudio.save(out, waveform, target_sr)
    return out

def save(audio, sr, path):
    audio = np.array(audio, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    sf.write(path, audio, sr)

def run_xtts(text, ref, cfg, out):
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    audio  = model.tts(
        text=text, speaker_wav=preprocess(ref, 22050), language="en",
        temperature=cfg.get("temperature",0.65), speed=cfg.get("speed",1.0),
        top_k=cfg.get("top_k",50), top_p=cfg.get("top_p",0.85),
        length_penalty=cfg.get("length_penalty",1.0),
        repetition_penalty=cfg.get("repetition_penalty",10.0),
    )
    save(audio, 24000, out)
    print(f"SAVED:{out}", flush=True)

def run_yourtts(text, ref, cfg, out):
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
    audio  = model.tts(
        text=text, speaker_wav=preprocess(ref, 16000), language="en",
        noise_scale=cfg.get("noise_scale",0.667),
        noise_scale_w=cfg.get("noise_scale_w",0.8),
        length_scale=cfg.get("length_scale",1.0),
    )
    save(audio, 16000, out)
    print(f"SAVED:{out}", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["xtts","yourtts"])
    p.add_argument("--text",  required=True)
    p.add_argument("--ref",   required=True)
    p.add_argument("--out",   required=True)
    p.add_argument("--cfg",   default="{}")
    a = p.parse_args()
    cfg = json.loads(a.cfg)
    if a.model == "xtts":     run_xtts(a.text, a.ref, cfg, a.out)
    elif a.model == "yourtts": run_yourtts(a.text, a.ref, cfg, a.out)
