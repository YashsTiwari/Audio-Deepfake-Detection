# Task 0 — Voice Cloning Pipeline

## Setup (run once)
```bash
bash setup.sh
source hackathon_task_0/bin/activate
```

## Step 1 — Hyperparameter sweep (find best configs)
```bash
# Quick test first (1 config each, fast)
python generate_clones.py --ref your_voice.wav --quick

# Full sweep (all configs, takes 20-40min depending on GPU)
python generate_clones.py --ref your_voice.wav

# Run only specific systems
python generate_clones.py --ref your_voice.wav --systems xtts yourtts
```

**Output structure:**
```
outputs/
    xtts/
        sent1_temp0.65_spd1.0_topk50.wav   ← filename = hyperparams
        sent1_temp0.75_spd1.0_topk50.wav
        sent2_temp0.65_spd1.0_topk50.wav
        ...
    yourtts/
        sent1_ns0.667_nsw0.8_ls1.0.wav
        ...
    chatterbox/
        sent1_exag0.5_cfg0.5_temp0.8.wav
        ...
    summary.txt   ← copy best config from here
```

## Step 2 — Listen and pick best
Listen to the outputs folder.
Find the filename that sounds best per system.
The filename tells you the config:
`sent1_temp0.65_spd1.0_topk50.wav` → temperature=0.65, speed=1.0, top_k=50

## Step 3 — Update ui.py
Open `ui.py` and paste your chosen params into `BEST_PARAMS` at the top.

## Step 4 — Run the UI
```bash
python ui.py
# Open http://localhost:7860
```
Upload or record your voice → type any text → click Generate.
All 3 clones appear side by side and are auto-saved to `ui_outputs/`.

## Recording tips for best clone quality
- Quiet room, no echo
- 15–25 seconds of natural speech
- Vary your tone — include a question, a statement, something with a pause
- 16kHz or 44kHz WAV both fine (script resamples automatically)
- Don't hum, cough, or have music in background

## VRAM usage (all 3 models loaded)
| System     | VRAM    |
|------------|---------|
| XTTS v2    | ~4 GB   |
| YourTTS    | ~2 GB   |
| Chatterbox | ~3 GB   |
| **Total**  | **~9 GB** |
Your 3060 12GB handles this comfortably.
