#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Hackathon Task 0 — Environment Setup
# Run this once from the directory where you want the project to live.
# Usage: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

echo "========================================"
echo " Hackathon Task 0 — Environment Setup"
echo "========================================"

# Step 1 — create venv
echo ""
echo "[1/5] Creating venv: hackathon_task_0"
python3 -m venv hackathon_task_0
source hackathon_task_0/bin/activate
echo "      Python: $(python --version)"
echo "      Pip:    $(pip --version)"

# Step 2 — base tools
echo ""
echo "[2/5] Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel --quiet

# Step 3 — torch (CUDA 12.1 — matches 3060 / A6000 / H200)
# If you're on CUDA 11.8 replace cu121 with cu118 below
echo ""
echo "[3/5] Installing PyTorch (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "      Torch installed: $(python -c 'import torch; print(torch.__version__)')"
echo "      CUDA available:  $(python -c 'import torch; print(torch.cuda.is_available())')"

# Step 4 — TTS library (Coqui — contains both XTTS v2 and YourTTS)
echo ""
echo "[4/5] Installing Coqui TTS, Chatterbox, and UI/audio deps"

pip install --quiet \
    TTS \
    chatterbox-tts \
    gradio \
    soundfile \
    librosa \
    numpy \
    scipy \
    tqdm \
    tabulate

# Montreal Forced Aligner is separate — optional for Task 0 but needed later
# Install via conda:  conda install -c conda-forge montreal-forced-aligner
# or:  pip install aligner  (limited version)

# Step 5 — verify key imports
echo ""
echo "[5/5] Verifying imports"
python - <<'EOF'
imports = ["torch", "TTS", "chatterbox", "gradio", "librosa", "soundfile", "scipy"]
ok, fail = [], []
for m in imports:
    try:
        __import__(m)
        ok.append(m)
    except ImportError as e:
        fail.append(f"{m} ({e})")

print(f"  OK:     {', '.join(ok)}")
if fail:
    print(f"  FAILED: {', '.join(fail)}")
else:
    print("  All imports successful.")
EOF

echo ""
echo "========================================"
echo " Setup complete."
echo " Activate with:  source hackathon_task_0/bin/activate"
echo " Run pipeline:   python generate_clones.py"
echo " Run UI:         python ui.py"
echo "========================================"
