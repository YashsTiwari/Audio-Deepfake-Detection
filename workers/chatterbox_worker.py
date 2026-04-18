import argparse, json, warnings
import numpy as np
import torch, soundfile as sf
warnings.filterwarnings("ignore")

# Patch perth before chatterbox imports it
# PerthImplicitWatermarker requires a native lib that may not be available
# DummyWatermarker is a no-op replacement — safe for hackathon use
import perth
if perth.PerthImplicitWatermarker is None:
    import unittest.mock as mock
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

def save(audio, sr, path):
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().numpy()
    audio = np.array(audio, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    sf.write(path, audio, sr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--ref",  required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--cfg",  default="{}")
    a = p.parse_args()
    cfg = json.loads(a.cfg)

    from chatterbox.tts import ChatterboxTTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ChatterboxTTS.from_pretrained(device=device)
    wav    = model.generate(
        text              = a.text,
        audio_prompt_path = a.ref,
        exaggeration      = cfg.get("exaggeration", 0.5),
        cfg_weight        = cfg.get("cfg_weight", 0.5),
        temperature       = cfg.get("temperature", 0.8),
    )
    save(wav, 22050, a.out)
    print(f"SAVED:{a.out}", flush=True)
