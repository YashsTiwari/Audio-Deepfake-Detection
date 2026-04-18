"""
deep_analysis.py — Comprehensive Audio Forensics Analysis
══════════════════════════════════════════════════════════
Extracts every possible discriminative signal from real vs fake audio.
Covers: spectral, prosodic, noise, phase, cepstral, temporal, cross-domain.

Usage:
    source /scratch/s25089/venvs/analysis/bin/activate
    cd /scratch/s25089/voice_analysis
    python deep_analysis.py

Output:
    analysis/
        features.csv          ← all features for all files
        discrimination.txt    ← which features separate real from fake
        plots/                ← spectrogram comparisons, pitch contours
        hardcode_thresholds.json ← values to bake into detector
"""

import warnings, os, json, time
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
import parselmouth
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from parselmouth.praat import call

# ── Paths ─────────────────────────────────────────────────────────
BASE      = Path("/home/turing/projects/hackathon_task_0")
REAL_DIR  = BASE / "input_voices"
CLONE_DIR = BASE / "output_voices"
OUT_DIR   = BASE / "analysis"
PLOT_DIR  = OUT_DIR / "plots"

for d in [OUT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SR = 16000  # standard sample rate for all analysis

# ── Systems to analyze ────────────────────────────────────────────
SYSTEMS = ["xtts", "yourtts", "chatterbox", "xtts_finetune"]
SENTENCES = ["sent1", "sent2", "sent3", "sent4", "sent5"]


# ══════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ══════════════════════════════════════════════════════════════════

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    # Normalise
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


# ══════════════════════════════════════════════════════════════════
# BRANCH 1: SPECTRAL FEATURES (strongest signal)
# ══════════════════════════════════════════════════════════════════

def extract_spectral(audio: np.ndarray, sr: int = SR) -> dict:
    """
    Full spectral analysis.
    Vocoder fingerprints live in frequency space — this is where we dig hardest.
    """
    feats = {}

    # ── Mel spectrogram statistics ────────────────────────────────
    mel = librosa.feature.melspectrogram(y=audio, sr=sr,
                                          n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    feats["mel_mean"]       = float(np.mean(mel_db))
    feats["mel_std"]        = float(np.std(mel_db))
    feats["mel_skew"]       = float(skew(mel_db.flatten()))
    feats["mel_kurtosis"]   = float(kurtosis(mel_db.flatten()))

    # ── Sub-band energy ratios ─────────────────────────────────────
    # THE strongest vocoder signal — neural vocoders leave aliasing above 4kHz
    stft  = np.abs(librosa.stft(audio, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_energy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.mean(stft[mask, :] ** 2)) if mask.any() else 0.0

    e_total = band_energy(0, sr//2) + 1e-10
    feats["energy_0_1k"]    = band_energy(0,    1000)  / e_total
    feats["energy_1_4k"]    = band_energy(1000, 4000)  / e_total
    feats["energy_4_8k"]    = band_energy(4000, 8000)  / e_total
    feats["energy_8k_plus"] = band_energy(8000, sr//2) / e_total
    feats["hf_energy_ratio"]= feats["energy_4_8k"] + feats["energy_8k_plus"]

    # ── Spectral entropy per band ──────────────────────────────────
    def spectral_entropy(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        if not mask.any():
            return 0.0
        mag = stft[mask, :].mean(axis=1)
        mag = mag / (mag.sum() + 1e-10)
        return float(-np.sum(mag * np.log2(mag + 1e-10)))

    feats["entropy_0_1k"]   = spectral_entropy(0,    1000)
    feats["entropy_1_4k"]   = spectral_entropy(1000, 4000)
    feats["entropy_4_8k"]   = spectral_entropy(4000, 8000)
    feats["entropy_total"]  = spectral_entropy(0,    sr//2)

    # ── Spectral flatness ──────────────────────────────────────────
    # Real speech is spiky (resonant), vocoded is smoother/flatter
    flatness = librosa.feature.spectral_flatness(y=audio)
    feats["spectral_flatness_mean"] = float(np.mean(flatness))
    feats["spectral_flatness_std"]  = float(np.std(flatness))
    feats["spectral_flatness_max"]  = float(np.max(flatness))

    # ── Spectral centroid & rolloff ────────────────────────────────
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)
    feats["centroid_mean"]  = float(np.mean(centroid))
    feats["centroid_std"]   = float(np.std(centroid))
    feats["rolloff_mean"]   = float(np.mean(rolloff))
    feats["rolloff_std"]    = float(np.std(rolloff))

    # ── Spectral flux ──────────────────────────────────────────────
    # Real speech has higher flux at phoneme boundaries
    # TTS over-smooths transitions
    flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    feats["spectral_flux_mean"] = float(np.mean(flux))
    feats["spectral_flux_std"]  = float(np.std(flux))
    feats["spectral_flux_max"]  = float(np.max(flux))

    # ── MFCC statistics ────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    for i in range(13):  # first 13 MFCCs
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    # ── MFCC delta (rate of change) ───────────────────────────────
    # Real speech has higher MFCC delta (more dynamic)
    mfcc_delta = librosa.feature.delta(mfcc)
    feats["mfcc_delta_mean"] = float(np.mean(np.abs(mfcc_delta)))
    feats["mfcc_delta_std"]  = float(np.std(mfcc_delta))

    # ── LFCC (Linear Frequency Cepstral Coefficients) ─────────────
    # Used in ASVspoof — captures different artifacts than MFCC
    # LFCC — use linear filterbank via scipy directly
    f, t, Sxx = scipy_signal.spectrogram(audio, fs=sr, nperseg=2048)
    log_spec = np.log(Sxx + 1e-10)
    lfcc = np.mean(log_spec, axis=1)
    lfcc_feat = lfcc[:20] if len(lfcc) >= 20 else np.pad(lfcc, (0, 20-len(lfcc)))
    feats["lfcc_mean"] = float(np.mean(lfcc_feat))
    feats["lfcc_std"]  = float(np.std(lfcc_feat))

    # ── Chroma features ────────────────────────────────────────────
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    feats["chroma_mean"] = float(np.mean(chroma))
    feats["chroma_std"]  = float(np.std(chroma))

    # ── Zero crossing rate ─────────────────────────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # ── DCT periodic peaks (vocoder aliasing fingerprint) ─────────
    # Neural vocoders upsample at fixed rates leaving periodic peaks
    # in DCT domain — classic detection signal from image deepfake literature
    dct_audio = np.abs(np.fft.rfft(audio))
    # Look for periodic peaks in 100-500 sample range
    peak_spacing = _find_periodic_peaks(dct_audio[:1000])
    feats["dct_periodic_peak_score"] = float(peak_spacing)

    # ── RMS energy statistics ──────────────────────────────────────
    rms = librosa.feature.rms(y=audio)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))
    feats["rms_min"]  = float(np.min(rms))

    return feats


def _find_periodic_peaks(spectrum: np.ndarray) -> float:
    """
    Detect periodic peaks in spectrum — vocoder aliasing fingerprint.
    Returns score: higher = more periodic (more synthetic).
    """
    if len(spectrum) < 10:
        return 0.0
    # Autocorrelation of the spectrum
    spec_norm = spectrum / (np.max(spectrum) + 1e-10)
    autocorr  = np.correlate(spec_norm, spec_norm, mode='full')
    autocorr  = autocorr[len(autocorr)//2:]
    # Find peaks in autocorrelation
    peaks, _  = scipy_signal.find_peaks(autocorr[1:100], height=0.1)
    return float(len(peaks))


# ══════════════════════════════════════════════════════════════════
# BRANCH 2: PHASE ANALYSIS
# ══════════════════════════════════════════════════════════════════

def extract_phase(audio: np.ndarray, sr: int = SR) -> dict:
    """
    Phase consistency analysis.
    Neural vocoders optimize magnitude well but phase poorly.
    Group delay reveals this clearly.
    """
    feats = {}

    stft_complex = librosa.stft(audio, n_fft=2048)
    phase        = np.angle(stft_complex)
    magnitude    = np.abs(stft_complex)

    # ── Phase variance ────────────────────────────────────────────
    feats["phase_variance_mean"] = float(np.mean(np.var(phase, axis=1)))
    feats["phase_variance_std"]  = float(np.std(np.var(phase, axis=1)))

    # ── Group delay ───────────────────────────────────────────────
    # Group delay = -d(phase)/d(omega)
    # Real speech: smooth group delay
    # Vocoded speech: jagged group delay especially at consonant boundaries
    phase_diff     = np.diff(np.unwrap(phase, axis=0), axis=0)
    group_delay    = -phase_diff
    feats["group_delay_mean"]    = float(np.mean(np.abs(group_delay)))
    feats["group_delay_std"]     = float(np.std(group_delay))
    feats["group_delay_jag"]     = float(np.mean(np.abs(np.diff(group_delay, axis=0))))

    # ── Phase-magnitude consistency ───────────────────────────────
    # In real speech, phase and magnitude are correlated in specific ways
    # In vocoded speech this correlation breaks
    mag_flat   = magnitude.flatten()
    phase_flat = np.abs(phase.flatten())
    if len(mag_flat) > 100:
        corr = np.corrcoef(mag_flat[:1000], phase_flat[:1000])[0, 1]
        feats["phase_mag_correlation"] = float(corr) if not np.isnan(corr) else 0.0
    else:
        feats["phase_mag_correlation"] = 0.0

    # ── Instantaneous frequency ───────────────────────────────────
    inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
    feats["inst_freq_mean"] = float(np.mean(np.abs(inst_freq)))
    feats["inst_freq_std"]  = float(np.std(inst_freq))

    return feats


# ══════════════════════════════════════════════════════════════════
# BRANCH 3: PROSODIC FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_prosody(audio: np.ndarray, sr: int = SR) -> dict:
    """
    F0, jitter, shimmer, HNR via Praat (parselmouth).
    THE most explainable signal — real voices have natural micro-variations
    that neural TTS smooths out.
    """
    feats = {}

    try:
        snd   = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array["frequency"]
        pitch_values = pitch_values[pitch_values > 0]  # voiced frames only

        if len(pitch_values) > 10:
            feats["f0_mean"]   = float(np.mean(pitch_values))
            feats["f0_std"]    = float(np.std(pitch_values))
            feats["f0_min"]    = float(np.min(pitch_values))
            feats["f0_max"]    = float(np.max(pitch_values))
            feats["f0_range"]  = float(np.max(pitch_values) - np.min(pitch_values))

            # Jitter — cycle-to-cycle frequency variation
            # Real speech: 1-3%, synthetic: 0.1-0.5%
            f0_diff = np.abs(np.diff(pitch_values))
            feats["f0_jitter_local"]    = float(np.mean(f0_diff) / (np.mean(pitch_values) + 1e-10))
            feats["f0_jitter_abs"]      = float(np.mean(f0_diff))
            feats["f0_jitter_std"]      = float(np.std(f0_diff))

            # Smoothness — TTS is too smooth
            feats["f0_smoothness"]      = float(1.0 / (np.std(f0_diff) + 1e-10))

            # F0 contour complexity
            feats["f0_complexity"]      = float(np.sum(np.abs(np.diff(pitch_values, 2))))
        else:
            for k in ["f0_mean","f0_std","f0_min","f0_max","f0_range",
                      "f0_jitter_local","f0_jitter_abs","f0_jitter_std",
                      "f0_smoothness","f0_complexity"]:
                feats[k] = 0.0

        # ── Point process for shimmer & HNR ──────────────────────
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

            # Shimmer — amplitude variation
            # Real: 3-8%, synthetic: 0.5-2%
            shimmer = call([snd, point_process],
                           "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            feats["shimmer"] = float(shimmer) if shimmer and not np.isnan(shimmer) else 0.0

            shimmer_db = call([snd, point_process],
                              "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            feats["shimmer_db"] = float(shimmer_db) if shimmer_db and not np.isnan(shimmer_db) else 0.0

            # HNR — harmonics to noise ratio
            # Real: varies naturally, synthetic: often too high or too low
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr          = call(harmonicity, "Get mean", 0, 0)
            feats["hnr"] = float(hnr) if hnr and not np.isnan(hnr) else 0.0

            # Jitter from point process (more accurate)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            feats["jitter_praat"] = float(jitter_local) if jitter_local and not np.isnan(jitter_local) else 0.0

            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            feats["jitter_rap"] = float(jitter_rap) if jitter_rap and not np.isnan(jitter_rap) else 0.0

        except Exception:
            feats["shimmer"]      = 0.0
            feats["shimmer_db"]   = 0.0
            feats["hnr"]          = 0.0
            feats["jitter_praat"] = 0.0
            feats["jitter_rap"]   = 0.0

        # ── Formant analysis ──────────────────────────────────────
        # F1, F2, F3 — vocal tract resonances
        # TTS formant transitions are less smooth than real speech
        try:
            formants = snd.to_formant_burg()
            f1_vals, f2_vals, f3_vals = [], [], []
            times = np.linspace(0, snd.duration, 50)
            for t in times:
                f1 = formants.get_value_at_time(1, t)
                f2 = formants.get_value_at_time(2, t)
                f3 = formants.get_value_at_time(3, t)
                if f1 and not np.isnan(f1): f1_vals.append(f1)
                if f2 and not np.isnan(f2): f2_vals.append(f2)
                if f3 and not np.isnan(f3): f3_vals.append(f3)

            for name, vals in [("f1",f1_vals),("f2",f2_vals),("f3",f3_vals)]:
                if vals:
                    feats[f"{name}_mean"] = float(np.mean(vals))
                    feats[f"{name}_std"]  = float(np.std(vals))
                    feats[f"{name}_transition_smoothness"] = float(
                        1.0 / (np.std(np.diff(vals)) + 1e-10))
                else:
                    feats[f"{name}_mean"] = 0.0
                    feats[f"{name}_std"]  = 0.0
                    feats[f"{name}_transition_smoothness"] = 0.0
        except Exception:
            for name in ["f1","f2","f3"]:
                feats[f"{name}_mean"] = 0.0
                feats[f"{name}_std"]  = 0.0
                feats[f"{name}_transition_smoothness"] = 0.0

    except Exception as e:
        print(f"    Prosody extraction failed: {e}")
        for k in ["f0_mean","f0_std","f0_min","f0_max","f0_range",
                  "f0_jitter_local","f0_jitter_abs","f0_jitter_std",
                  "f0_smoothness","f0_complexity","shimmer","shimmer_db",
                  "hnr","jitter_praat","jitter_rap",
                  "f1_mean","f1_std","f1_transition_smoothness",
                  "f2_mean","f2_std","f2_transition_smoothness",
                  "f3_mean","f3_std","f3_transition_smoothness"]:
            feats[k] = 0.0

    return feats


# ══════════════════════════════════════════════════════════════════
# BRANCH 4: NOISE & ENVIRONMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════

def extract_noise(audio: np.ndarray, sr: int = SR) -> dict:
    """
    The most original signal in this analysis.
    Real recordings have consistent room noise.
    Synthetic speech has dead silence between phonemes.
    Breathing, lip smacks, room tone — all missing in clones.
    """
    feats = {}

    # ── Voice activity detection ──────────────────────────────────
    frame_len  = int(0.025 * sr)  # 25ms frames
    hop_len    = int(0.010 * sr)  # 10ms hop
    rms_frames = librosa.feature.rms(y=audio,
                                      frame_length=frame_len,
                                      hop_length=hop_len)[0]
    threshold  = np.mean(rms_frames) * 0.3

    voiced_mask  = rms_frames > threshold
    silence_mask = ~voiced_mask

    feats["voiced_ratio"]  = float(voiced_mask.mean())
    feats["silence_ratio"] = float(silence_mask.mean())

    # ── SNR voiced vs silence ──────────────────────────────────────
    # KEY SIGNAL: In real speech SNR_speech / SNR_silence ≈ 1
    # (same room noise everywhere)
    # In synthetic: silence is DEAD QUIET → ratio is very high
    if voiced_mask.any() and silence_mask.any():
        snr_voiced  = float(np.mean(rms_frames[voiced_mask]))
        snr_silence = float(np.mean(rms_frames[silence_mask]))
        feats["snr_voiced"]       = snr_voiced
        feats["snr_silence"]      = snr_silence
        feats["snr_ratio"]        = snr_voiced / (snr_silence + 1e-10)
        feats["snr_consistency"]  = 1.0 / (feats["snr_ratio"] + 1e-10)
    else:
        feats["snr_voiced"]      = 0.0
        feats["snr_silence"]     = 0.0
        feats["snr_ratio"]       = 0.0
        feats["snr_consistency"] = 0.0

    # ── Noise floor analysis ──────────────────────────────────────
    # Extract silence segments and analyze their spectral profile
    silence_rms_vals = rms_frames[silence_mask]
    if len(silence_rms_vals) > 5:
        feats["noise_floor_mean"]     = float(np.mean(silence_rms_vals))
        feats["noise_floor_std"]      = float(np.std(silence_rms_vals))
        feats["noise_floor_cv"]       = float(np.std(silence_rms_vals) /
                                              (np.mean(silence_rms_vals) + 1e-10))
        # Low CV = consistent noise (real), high CV = irregular (synthetic)
    else:
        feats["noise_floor_mean"] = 0.0
        feats["noise_floor_std"]  = 0.0
        feats["noise_floor_cv"]   = 0.0

    # ── Breathing detection ────────────────────────────────────────
    # Breathing appears as low-frequency bursts (50-300Hz) before utterances
    # COMPLETELY ABSENT in synthetic speech
    lo_audio = librosa.effects.preemphasis(audio, coef=-0.97)  # keep lows
    breathing_frames = _detect_breathing(audio, sr)
    feats["breathing_count"]   = float(breathing_frames)
    feats["has_breathing"]     = float(breathing_frames > 0)

    # ── Pause statistics ──────────────────────────────────────────
    # Real speech: variable pauses, micro-hesitations
    # TTS: uniform pauses, metronomic
    pause_durations = _get_pause_durations(rms_frames, threshold,
                                            hop_len, sr)
    if pause_durations:
        feats["pause_count"]        = float(len(pause_durations))
        feats["pause_mean_dur"]     = float(np.mean(pause_durations))
        feats["pause_std_dur"]      = float(np.std(pause_durations))
        feats["pause_cv"]           = float(np.std(pause_durations) /
                                            (np.mean(pause_durations) + 1e-10))
        feats["pause_max"]          = float(np.max(pause_durations))
        feats["pause_min"]          = float(np.min(pause_durations))
    else:
        feats["pause_count"]    = 0.0
        feats["pause_mean_dur"] = 0.0
        feats["pause_std_dur"]  = 0.0
        feats["pause_cv"]       = 0.0
        feats["pause_max"]      = 0.0
        feats["pause_min"]      = 0.0

    # ── Background noise spectral profile ─────────────────────────
    # Extract noise-only segments and get their spectral centroid
    silence_audio = _extract_silence_segments(audio, rms_frames,
                                               silence_mask, hop_len)
    if len(silence_audio) > sr * 0.1:
        stft_noise   = np.abs(librosa.stft(silence_audio))
        noise_cent   = librosa.feature.spectral_centroid(
            S=stft_noise, sr=sr)
        feats["noise_spectral_centroid"]  = float(np.mean(noise_cent))
        feats["noise_spectral_centroid_std"] = float(np.std(noise_cent))
        feats["noise_spectral_drift"]     = float(np.std(noise_cent) /
                                                  (np.mean(noise_cent) + 1e-10))
    else:
        feats["noise_spectral_centroid"]      = 0.0
        feats["noise_spectral_centroid_std"]  = 0.0
        feats["noise_spectral_drift"]         = 0.0

    # ── Amplitude envelope smoothness ─────────────────────────────
    envelope = np.abs(librosa.effects.preemphasis(audio))
    env_smooth = librosa.feature.rms(y=envelope,
                                      frame_length=frame_len,
                                      hop_length=hop_len)[0]
    feats["envelope_smoothness"] = float(1.0 / (np.std(np.diff(env_smooth)) + 1e-10))
    feats["envelope_cv"]         = float(np.std(env_smooth) /
                                         (np.mean(env_smooth) + 1e-10))

    return feats


def _detect_breathing(audio: np.ndarray, sr: int) -> int:
    """Count breathing events: low-freq bursts 50-300Hz before voiced segments."""
    try:
        # Bandpass filter 50-300Hz
        b, a = scipy_signal.butter(4, [50/(sr/2), 300/(sr/2)], btype='band')
        lo   = scipy_signal.filtfilt(b, a, audio)
        rms  = librosa.feature.rms(y=lo,
                                    frame_length=int(0.05*sr),
                                    hop_length=int(0.01*sr))[0]
        thresh     = np.mean(rms) * 2.0
        peaks, _   = scipy_signal.find_peaks(rms, height=thresh,
                                              distance=int(0.3*sr/int(0.01*sr)))
        return len(peaks)
    except Exception:
        return 0


def _get_pause_durations(rms_frames, threshold, hop_len, sr):
    """Extract pause durations from VAD mask."""
    in_pause = False
    start    = 0
    durations = []
    for i, rms in enumerate(rms_frames):
        if rms < threshold and not in_pause:
            in_pause = True
            start    = i
        elif rms >= threshold and in_pause:
            dur = (i - start) * hop_len / sr
            if dur > 0.05:  # min 50ms pause
                durations.append(dur)
            in_pause = False
    return durations


def _extract_silence_segments(audio, rms_frames, silence_mask, hop_len):
    """Concatenate all silence segments into one array."""
    segments = []
    for i, is_silence in enumerate(silence_mask):
        if is_silence:
            start = i * hop_len
            end   = min(start + hop_len, len(audio))
            segments.append(audio[start:end])
    return np.concatenate(segments) if segments else np.array([])


# ══════════════════════════════════════════════════════════════════
# BRANCH 5: TEMPORAL / RHYTHM FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_temporal(audio: np.ndarray, sr: int = SR) -> dict:
    """
    Speaking rate, rhythm, temporal dynamics.
    TTS is metronomic; real speech has natural acceleration/deceleration.
    """
    feats = {}

    # ── Onset detection (syllable timing) ─────────────────────────
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr,
                                               units='frames')
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr)

    if len(onset_times) > 2:
        ioi = np.diff(onset_times)  # inter-onset intervals
        feats["onset_count"]    = float(len(onset_times))
        feats["ioi_mean"]       = float(np.mean(ioi))
        feats["ioi_std"]        = float(np.std(ioi))
        feats["ioi_cv"]         = float(np.std(ioi) / (np.mean(ioi) + 1e-10))
        feats["speaking_rate"]  = float(len(onset_times) / (len(audio)/sr))
        # Rhythm regularity — TTS is too regular (low CV)
        feats["rhythm_regularity"] = float(1.0 / (np.std(ioi) + 1e-10))
    else:
        for k in ["onset_count","ioi_mean","ioi_std","ioi_cv",
                  "speaking_rate","rhythm_regularity"]:
            feats[k] = 0.0

    # ── Tempo & beat strength ──────────────────────────────────────
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        feats["tempo"]        = float(tempo)
        feats["beat_count"]   = float(len(beats))
    except Exception:
        feats["tempo"]      = 0.0
        feats["beat_count"] = 0.0

    # ── Energy dynamics ────────────────────────────────────────────
    hop = int(0.01 * sr)
    rms = librosa.feature.rms(y=audio,
                               frame_length=int(0.025*sr),
                               hop_length=hop)[0]
    feats["energy_range"]   = float(np.max(rms) - np.min(rms))
    feats["energy_cv"]      = float(np.std(rms) / (np.mean(rms) + 1e-10))
    feats["energy_peaks"]   = float(len(scipy_signal.find_peaks(rms,
                                    height=np.mean(rms))[0]))

    return feats


# ══════════════════════════════════════════════════════════════════
# BRANCH 6: CROSS-DOMAIN FEATURES
# ══════════════════════════════════════════════════════════════════

def extract_crossdomain(audio: np.ndarray, sr: int = SR) -> dict:
    """
    Techniques borrowed from other domains:
    - DCT periodic peaks (image deepfake detection)
    - Multiscale entropy (medical signal processing)
    - Amplitude modulation (radar/sonar DEMON analysis)
    - Glottal pulse irregularity (speech science)
    """
    feats = {}

    # ── Multiscale entropy ─────────────────────────────────────────
    # Real speech has high complexity at fine scales
    # TTS smooths out micro-variations → lower entropy at fine scales
    for scale in [1, 2, 4, 8, 16]:
        if scale == 1:
            scaled = audio
        else:
            # Coarse-grain the signal
            n = len(audio) // scale * scale
            scaled = audio[:n].reshape(-1, scale).mean(axis=1)
        feats[f"entropy_scale{scale}"] = float(_sample_entropy(scaled[:2000]))

    # ── Amplitude modulation (respiratory rhythm) ─────────────────
    # Real speech has 0.2-0.5Hz respiratory modulation
    # TTS does not model breathing → no AM at respiratory frequency
    envelope  = np.abs(scipy_signal.hilbert(audio))
    # Look for modulation in 0.1-2Hz range
    if len(envelope) > sr * 2:
        am_freqs = np.fft.rfftfreq(len(envelope), 1/sr)
        am_spec  = np.abs(np.fft.rfft(envelope))
        resp_mask = (am_freqs >= 0.1) & (am_freqs <= 2.0)
        if resp_mask.any():
            feats["respiratory_am_energy"] = float(np.mean(am_spec[resp_mask]))
            feats["respiratory_am_peak"]   = float(np.max(am_spec[resp_mask]))
        else:
            feats["respiratory_am_energy"] = 0.0
            feats["respiratory_am_peak"]   = 0.0
    else:
        feats["respiratory_am_energy"] = 0.0
        feats["respiratory_am_peak"]   = 0.0

    # ── Cepstral peak prominence (CPP) ────────────────────────────
    # Strong predictor of voice quality — used in clinical voice assessment
    # TTS voices have abnormally smooth CPP
    feats["cpp"] = float(_compute_cpp(audio, sr))

    # ── Spectral irregularity ──────────────────────────────────────
    # Jensen irregularity — adjacent harmonic amplitude variation
    # Real: irregular, TTS: too smooth
    stft = np.abs(librosa.stft(audio, n_fft=2048))
    irregularity = np.mean(np.sum((stft[1:,:] - stft[:-1,:]) ** 2, axis=0) /
                           (np.sum(stft ** 2, axis=0) + 1e-10))
    feats["spectral_irregularity"] = float(irregularity)

    # ── Kurtosis of waveform ──────────────────────────────────────
    # Real speech waveform has high kurtosis (impulsive)
    # TTS is more Gaussian
    feats["waveform_kurtosis"] = float(kurtosis(audio))
    feats["waveform_skew"]     = float(skew(audio))

    # ── Crest factor ──────────────────────────────────────────────
    # Peak to RMS ratio — real speech has higher crest factor
    rms_val = np.sqrt(np.mean(audio ** 2))
    feats["crest_factor"] = float(np.max(np.abs(audio)) / (rms_val + 1e-10))

    return feats


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy — measure of signal complexity/unpredictability."""
    try:
        n   = len(x)
        r  *= np.std(x)
        if r == 0 or n < 10:
            return 0.0
        def count_matches(template_len):
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    if np.max(np.abs(x[i:i+template_len] - x[j:j+template_len])) < r:
                        count += 1
            return count
        A = count_matches(m + 1)
        B = count_matches(m)
        if B == 0:
            return 0.0
        return float(-np.log(A / B)) if A > 0 else 2.0
    except Exception:
        return 0.0


def _compute_cpp(audio: np.ndarray, sr: int) -> float:
    """Cepstral Peak Prominence — voice quality measure."""
    try:
        frame_len = int(0.025 * sr)
        hop_len   = int(0.010 * sr)
        cpp_vals  = []
        for i in range(0, len(audio) - frame_len, hop_len):
            frame = audio[i:i+frame_len] * np.hanning(frame_len)
            spec  = np.abs(np.fft.rfft(frame)) + 1e-10
            ceps  = np.real(np.fft.irfft(np.log(spec)))
            # Find peak in quefrency range corresponding to F0 (50-500Hz)
            q_min = int(sr / 500)
            q_max = int(sr / 50)
            if q_max < len(ceps):
                peak  = np.max(ceps[q_min:q_max])
                # Linear regression baseline
                baseline = np.polyval(np.polyfit(
                    np.arange(q_min, q_max),
                    ceps[q_min:q_max], 1),
                    np.arange(q_min, q_max))
                cpp_vals.append(peak - np.mean(baseline))
        return float(np.mean(cpp_vals)) if cpp_vals else 0.0
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════

def plot_comparison(real_audio, fake_audio, person, system, sent,
                    sr=SR):
    """Side-by-side spectrogram + pitch comparison."""
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    audios = [("Real", real_audio), (f"Fake ({system})", fake_audio)]

    for col, (label, audio) in enumerate(audios):
        # Mel spectrogram
        ax1 = fig.add_subplot(gs[0, col])
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                  x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_title(f"{label} — Mel Spectrogram", fontsize=10)
        plt.colorbar(ax1.images[0], ax=ax1, format='%+2.0f dB')

        # Pitch contour
        ax2 = fig.add_subplot(gs[1, col])
        pitch = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
        times = librosa.times_like(pitch, sr=sr)
        pitch_voiced = np.where(pitch > 0, pitch, np.nan)
        ax2.plot(times, pitch_voiced, color='steelblue' if col==0 else 'coral',
                 linewidth=0.8, alpha=0.8)
        ax2.set_ylabel("F0 (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title(f"{label} — Pitch Contour", fontsize=10)
        ax2.set_ylim(0, 500)

        # RMS energy
        ax3 = fig.add_subplot(gs[2, col])
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr,
                                        hop_length=256)
        ax3.plot(t_rms, rms, color='green' if col==0 else 'orange',
                 linewidth=0.8)
        ax3.set_ylabel("RMS Energy")
        ax3.set_xlabel("Time (s)")
        ax3.set_title(f"{label} — Energy Profile", fontsize=10)

    plt.suptitle(f"{person} | {system} | {sent}", fontsize=12, fontweight='bold')
    out = PLOT_DIR / f"{person}_{system}_{sent}.png"
    plt.savefig(str(out), dpi=100, bbox_inches='tight')
    plt.close()
    return str(out)


# ══════════════════════════════════════════════════════════════════
# DISCRIMINATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_discrimination(df: pd.DataFrame) -> dict:
    """
    For each feature, compute:
    - Mean for real vs each fake system
    - Effect size (Cohen's d)
    - Consistency across speakers
    Returns ranked feature list with hardcode thresholds.
    """
    feature_cols = [c for c in df.columns
                    if c not in ["person","system","sentence","label","filepath"]]

    results = {}
    real_df = df[df["label"] == "real"]
    fake_df = df[df["label"] == "fake"]

    for feat in feature_cols:
        real_vals = real_df[feat].dropna().values
        fake_vals = fake_df[feat].dropna().values

        if len(real_vals) < 3 or len(fake_vals) < 3:
            continue

        real_mean = np.mean(real_vals)
        fake_mean = np.mean(fake_vals)
        real_std  = np.std(real_vals)
        fake_std  = np.std(fake_vals)

        # Cohen's d — effect size
        pooled_std = np.sqrt((real_std**2 + fake_std**2) / 2) + 1e-10
        cohens_d   = abs(real_mean - fake_mean) / pooled_std

        # Consistency: does this feature work across all fake systems?
        per_system = {}
        for sys in df[df["label"]=="fake"]["system"].unique():
            sys_vals = df[(df["label"]=="fake") & (df["system"]==sys)][feat].dropna().values
            if len(sys_vals) > 0:
                per_system[sys] = float(np.mean(sys_vals))

        system_consistency = 1.0 - np.std(list(per_system.values())) / (
            abs(np.mean(list(per_system.values()))) + 1e-10
        ) if per_system else 0.0

        # Suggest threshold
        # Use midpoint between real and fake means as threshold
        threshold = (real_mean + fake_mean) / 2
        direction = ">" if fake_mean > real_mean else "<"

        results[feat] = {
            "real_mean":    float(real_mean),
            "real_std":     float(real_std),
            "fake_mean":    float(fake_mean),
            "fake_std":     float(fake_std),
            "cohens_d":     float(cohens_d),
            "consistency":  float(system_consistency),
            "threshold":    float(threshold),
            "direction":    direction,
            "score":        float(cohens_d * max(0, system_consistency)),
        }

    return dict(sorted(results.items(),
                       key=lambda x: x[1]["score"], reverse=True))


def print_discrimination_report(disc: dict, top_n: int = 30) -> str:
    lines = [
        "=" * 80,
        "  FEATURE DISCRIMINATION REPORT",
        "  Real vs Fake — Cohen's d effect size + consistency across systems",
        "=" * 80,
        "",
        f"{'Feature':<40} {'Real':>8} {'Fake':>8} {'Cohen_d':>8} {'Consist':>8} {'Threshold':>12}",
        "─" * 80,
    ]

    hardcode = {}
    for feat, vals in list(disc.items())[:top_n]:
        hardcode_flag = "★ HARDCODE" if vals["cohens_d"] > 1.0 and vals["consistency"] > 0.5 else ""
        lines.append(
            f"{feat:<40} "
            f"{vals['real_mean']:>8.3f} "
            f"{vals['fake_mean']:>8.3f} "
            f"{vals['cohens_d']:>8.3f} "
            f"{vals['consistency']:>8.3f} "
            f"{vals['direction']}{vals['threshold']:>11.3f}  {hardcode_flag}"
        )
        if vals["cohens_d"] > 1.0 and vals["consistency"] > 0.5:
            hardcode[feat] = {
                "threshold": vals["threshold"],
                "direction": vals["direction"],
                "real_mean": vals["real_mean"],
                "fake_mean": vals["fake_mean"],
                "cohens_d":  vals["cohens_d"],
            }

    lines += [
        "",
        f"  Features suitable for hardcoding (Cohen's d > 1.0, consistency > 0.5):",
        f"  {len(hardcode)} features identified",
        "",
    ]
    for feat, vals in hardcode.items():
        lines.append(f"  {feat}: {vals['direction']}{vals['threshold']:.4f}  "
                    f"(real={vals['real_mean']:.3f}, fake={vals['fake_mean']:.3f}, "
                    f"d={vals['cohens_d']:.2f})")

    lines.append("=" * 80)
    return "\n".join(lines), hardcode


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def extract_all_features_from_audio(audio: np.ndarray) -> dict:
    """Extract all features from pre-loaded audio array."""
    feats = {}
    feats.update(extract_spectral(audio))
    feats.update(extract_phase(audio))
    feats.update(extract_prosody(audio))
    feats.update(extract_noise(audio))
    feats.update(extract_temporal(audio))
    # feats.update(extract_crossdomain(audio))  # skipped
    return feats


def extract_all_features(audio_path: str) -> dict:
    """Extract all features from one audio file."""
    audio = load_audio(audio_path)
    feats = {}
    feats.update(extract_spectral(audio))
    feats.update(extract_phase(audio))
    feats.update(extract_prosody(audio))
    feats.update(extract_noise(audio))
    feats.update(extract_temporal(audio))
    # feats.update(extract_crossdomain(audio))  # skipped
    return feats



def find_all_files() -> list[dict]:
    """Find all real and fake audio files."""
    entries = []

    # Build real voice lookup: person_name → filepath
    real_lookup = {}
    for wav in REAL_DIR.glob("*_converted.wav"):
        name = wav.stem.lower()
        for suffix in ["_converted","_vlsi","_voice","_test","_audio"]:
            name = name.replace(suffix, "")
        name = name.strip("_- ")
        real_lookup[name] = str(wav)

    print(f"  Real voices found: {list(real_lookup.keys())}")

    # Fake clones
    for system in SYSTEMS:
        sys_dir = CLONE_DIR / system
        if not sys_dir.exists():
            print(f"  [skip] {system} — folder not found")
            continue
        for person_dir in sorted(sys_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            person = person_dir.name
            wavs = list(person_dir.glob("sent*.wav"))
            if not wavs:
                continue
            for wav in sorted(wavs):
                sent = wav.stem
                entries.append({
                    "person":   person,
                    "system":   system,
                    "sentence": sent,
                    "label":    "fake",
                    "filepath": str(wav),
                })
            # Add matching real voice once per person
            if person in real_lookup:
                real_already = any(
                    e["person"] == person and e["system"] == "real"
                    for e in entries
                )
                if not real_already:
                    entries.append({
                        "person":   person,
                        "system":   "real",
                        "sentence": "full",
                        "label":    "real",
                        "filepath": real_lookup[person],
                    })
            else:
                print(f"  [warn] No real voice found for: {person}")

    print(f"  Total entries: {len(entries)}")
    real_count = sum(1 for e in entries if e["label"] == "real")
    fake_count = sum(1 for e in entries if e["label"] == "fake")
    print(f"  Real: {real_count}  Fake: {fake_count}")
    return entries



def main():
    print("\n" + "="*70)
    print("  DEEP AUDIO FORENSICS ANALYSIS")
    print(f"  Output: {OUT_DIR}")
    print("="*70)

    files = find_all_files()
    print(f"\n  Found {len(files)} files to analyze")

    systems = {}
    for e in files:
        systems[e["system"]] = systems.get(e["system"], 0) + 1
    for sys, count in sorted(systems.items()):
        print(f"    {sys:20s}: {count} files")

    # Check for existing results
    csv_path = OUT_DIR / "features.csv"
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        done_paths = set(existing["filepath"].tolist())
        files = [f for f in files if f["filepath"] not in done_paths]
        print(f"\n  Resuming — {len(files)} remaining")
        all_rows = existing.to_dict("records")
    else:
        all_rows = []

    # Extract features — parallel
    t_start = time.time()
    n_jobs  = 16

    def process_one(entry):
        try:
            if entry["label"] == "real":
                audio_full = load_audio(entry["filepath"])
                chunk = int(8 * SR)
                mid   = len(audio_full) // 2
                clips = [audio_full[SR*10:SR*10+chunk],
                         audio_full[mid:mid+chunk],
                         audio_full[-chunk-SR*5:-SR*5]]
                clips = [c for c in clips if len(c) == chunk]
                if clips:
                    all_feats = [extract_all_features_from_audio(c)
                                 for c in clips]
                    feats = {k: float(np.mean([f[k] for f in all_feats]))
                             for k in all_feats[0]}
                else:
                    feats = extract_all_features(entry["filepath"])
            else:
                feats = extract_all_features(entry["filepath"])
            print(f"  OK: {entry['person']:12s} | {entry['system']:15s} | {entry['sentence']}")
            return {**entry, **feats}
        except Exception as e:
            print(f"  FAIL: {entry['person']} {entry['system']}: {e}")
            return None

    print(f"  Running {n_jobs} parallel workers on {len(files)} files...")
    results = Parallel(n_jobs=n_jobs, verbose=0, prefer="threads")(
        delayed(process_one)(entry) for entry in files
    )
    all_rows.extend([r for r in results if r is not None])
    elapsed = (time.time() - t_start) / 60
    print(f"  Done: {len(all_rows)} files in {elapsed:.1f} min")

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Features saved: {csv_path}")
    print(f"  Total rows: {len(df)}")

    # Generate comparison plots (first person only to save time)
    print("\n  Generating comparison plots...")
    xtts_dir = CLONE_DIR / "xtts"
    if xtts_dir.exists():
        first_person = sorted(xtts_dir.iterdir())[0].name
        real_entry   = next((r for r in all_rows
                             if r["person"] == first_person
                             and r["system"] == "real"), None)
        if real_entry:
            real_audio = load_audio(real_entry["filepath"])
            # Trim to 10 seconds for comparison
            real_audio = real_audio[:SR*10]
            for system in ["xtts", "yourtts", "chatterbox"]:
                fake_entry = next((r for r in all_rows
                                   if r["person"] == first_person
                                   and r["system"] == system
                                   and r["sentence"] == "sent1"), None)
                if fake_entry:
                    fake_audio = load_audio(fake_entry["filepath"])
                    plot_comparison(real_audio, fake_audio,
                                    first_person, system, "sent1")
                    print(f"  Plot: {first_person} vs {system}")

    # Discrimination analysis
    print("\n  Running discrimination analysis...")
    disc      = analyze_discrimination(df)
    report, hardcode = print_discrimination_report(disc)
    print("\n" + report)

    # Save report
    report_path = OUT_DIR / "discrimination.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save hardcode thresholds
    thresh_path = OUT_DIR / "hardcode_thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(hardcode, f, indent=2)
    print(f"\n  Discrimination report: {report_path}")
    print(f"  Hardcode thresholds:   {thresh_path}")
    print(f"\n  DONE. Top discriminating features saved.")


if __name__ == "__main__":
    main()
