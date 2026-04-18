import os, sys, json, time, argparse, subprocess
from pathlib import Path
from datetime import datetime
from tabulate import tabulate

COQUI_PYTHON      = "/home/turing/venvs/task0_coqui/bin/python"
CHATTERBOX_PYTHON = "/home/turing/venvs/task0_chatterbox/bin/python"
WORKERS_DIR       = Path(__file__).parent / "workers"
INPUT_DIR         = Path("input_voices")
OUTPUT_DIR        = Path("output_voices")

SENTENCES = {
    "sent1": "The quick brown fox jumps over the lazy dog.",
    "sent2": "She sells seashells by the seashore on sunny afternoons.",
    "sent3": "Yesterday I walked through the empty streets and thought about everything that had changed.",
    "sent4": "Three large packages arrived at the front door just after noon.",
    "sent5": "I cannot believe how fast the technology has changed in just the last few years.",
}

BEST_PARAMS = {
    "xtts":       {"temperature": 0.65, "speed": 1.0, "top_k": 50,
                   "top_p": 0.85, "length_penalty": 1.0, "repetition_penalty": 10.0},
    "yourtts":    {"noise_scale": 0.667, "noise_scale_w": 0.8, "length_scale": 1.0},
    "chatterbox": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
}

def convert_to_wav(p):
    out = INPUT_DIR / f"{p.stem}_converted.wav"
    if out.exists(): return out
    print(f"  Converting {p.name} ...")
    r = subprocess.run(["ffmpeg","-y","-i",str(p),"-ar","16000","-ac","1",
                        "-acodec","pcm_s16le",str(out)],
                       capture_output=True, text=True)
    return out if r.returncode == 0 else p

def get_name(p):
    n = p.stem.lower()
    for s in ["_voice","_test","_audio","_recording","_sample","_converted","_vlsi"]:
        n = n.replace(s,"")
    return n.strip("_- ")

def find_voices(d):
    exts = {".wav",".mp3",".m4a",".ogg",".flac",".aac"}
    return [(get_name(f), f) for f in sorted(d.iterdir())
            if f.suffix.lower() in exts
            and not f.name.startswith(".")
            and "_converted" not in f.stem]

def call_worker(python, worker, extra, timeout=300):
    cmd = [python, str(WORKERS_DIR / worker)] + extra
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0: return True, r.stdout.strip()
        return False, r.stderr.strip()[-300:]
    except subprocess.TimeoutExpired: return False, "TIMEOUT"
    except Exception as e: return False, str(e)

def generate(system, text, ref, out, cfg):
    t0 = time.time()
    if system in ("xtts","yourtts"):
        ok, msg = call_worker(COQUI_PYTHON, "coqui_worker.py",
            ["--model",system,"--text",text,"--ref",str(ref),
             "--out",str(out),"--cfg",json.dumps(cfg)])
    else:
        ok, msg = call_worker(CHATTERBOX_PYTHON, "chatterbox_worker.py",
            ["--text",text,"--ref",str(ref),
             "--out",str(out),"--cfg",json.dumps(cfg)])
    return ok and out.exists(), time.time()-t0, msg

def run(systems, sentences, quick):
    voices = find_voices(INPUT_DIR)
    if not voices:
        print(f"ERROR: No audio files in {INPUT_DIR.resolve()}")
        sys.exit(1)

    print(f"\n  Found {len(voices)} voice(s):")
    for name, path in voices:
        print(f"    {name:20s} <- {path.name}")

    converted = []
    for name, path in voices:
        wav = convert_to_wav(path) if path.suffix.lower() != ".wav" else path
        converted.append((name, wav))

    active = {"sent1": sentences["sent1"]} if quick else sentences
    results = []

    for system in systems:
        cfg = BEST_PARAMS[system]
        print(f"\n{'─'*60}")
        print(f"  {system.upper()}  --  {len(converted)} voice(s) x {len(active)} sentence(s)")
        print(f"{'─'*60}")
        for person, ref in converted:
            person_dir = OUTPUT_DIR / system / person
            person_dir.mkdir(parents=True, exist_ok=True)
            for sent_key, text in active.items():
                out = person_dir / f"{sent_key}.wav"
                if out.exists():
                    print(f"  [skip] {system}/{person}/{sent_key}.wav")
                    results.append({"system":system,"person":person,
                                    "sentence":sent_key,"time_s":"-","status":"skipped"})
                    continue
                print(f"  {person:15s} | {sent_key} ... ", end="", flush=True)
                ok, elapsed, msg = generate(system, text, ref, out, cfg)
                if ok:
                    print(f"OK ({elapsed:.1f}s)")
                    status = "OK"
                else:
                    print(f"FAILED\n    {msg[:180]}")
                    status = f"ERR:{msg[:50]}"
                results.append({"system":system,"person":person,
                                "sentence":sent_key,"time_s":f"{elapsed:.1f}",
                                "status":status})
    return results

def print_summary(results):
    ok   = [r for r in results if r["status"] in ("OK","skipped")]
    fail = [r for r in results if r["status"] not in ("OK","skipped")]
    table = tabulate(
        [[r["system"],r["person"],r["sentence"],r["time_s"],r["status"]]
         for r in results],
        headers=["System","Person","Sentence","Time(s)","Status"],
        tablefmt="rounded_outline",
    )
    lines = ["="*65,
             f"  Clone Summary  --  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "="*65,"",table,"",
             f"  OK: {len(ok)}    Failed: {len(fail)}",
             f"  Output: {OUTPUT_DIR.resolve()}",
             "="*65]
    summary = "\n".join(lines)
    print("\n"+summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR/"summary.txt").write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--voices",  default="input_voices")
    p.add_argument("--out",     default="output_voices")
    p.add_argument("--systems", nargs="+",
                   default=["xtts","yourtts","chatterbox"],
                   choices=["xtts","yourtts","chatterbox"])
    p.add_argument("--quick",   action="store_true",
                   help="Only sent1 per voice")
    args = p.parse_args()
    INPUT_DIR  = Path(args.voices)
    OUTPUT_DIR = Path(args.out)

    if not INPUT_DIR.exists():
        INPUT_DIR.mkdir(parents=True)
        print(f"Created {INPUT_DIR.resolve()} -- add voice files and re-run")
        sys.exit(0)

    for py,label in [(COQUI_PYTHON,"coqui"),(CHATTERBOX_PYTHON,"chatterbox")]:
        if not Path(py).exists():
            print(f"ERROR: {label} venv not found at {py}"); sys.exit(1)

    ff = subprocess.run(["ffmpeg","-version"], capture_output=True)
    if ff.returncode != 0:
        print("WARNING: ffmpeg not found. Run: sudo apt install ffmpeg -y")

    print(f"\n  Voices : {INPUT_DIR.resolve()}")
    print(f"  Output : {OUTPUT_DIR.resolve()}")
    print(f"  Systems: {args.systems}")
    print(f"  Mode   : {'quick (sent1 only)' if args.quick else 'all 5 sentences'}")

    results = run(args.systems, SENTENCES, args.quick)
    print_summary(results)
