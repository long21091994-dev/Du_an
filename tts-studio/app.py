import os
import re
import gc
import time
import inspect
import urllib.request
from pathlib import Path
from datetime import datetime

# Store all HuggingFace caches on D: inside this project instead of C:
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
KOKORO_DIR = MODELS_DIR / "kokoro"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"
VOICES_DIR = BASE_DIR / "voices"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (MODELS_DIR, KOKORO_DIR, HF_CACHE_DIR, VOICES_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR / "hub"))

import numpy as np
import soundfile as sf
import gradio as gr

KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
KOKORO_MODEL_PATH = KOKORO_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = KOKORO_DIR / "voices-v1.0.bin"

KOKORO_LANGS = {
    "Tây Ban Nha (es)": "es",
    "Tiếng Anh - Mỹ (en-us)": "en-us",
    "Tiếng Anh - Anh (en-gb)": "en-gb",
    "Tiếng Pháp (fr-fr)": "fr-fr",
    "Tiếng Ý (it)": "it",
    "Bồ Đào Nha - Brazil (pt-br)": "pt-br",
    "Tiếng Nhật (ja)": "ja",
    "Tiếng Trung (zh)": "zh",
    "Tiếng Hindi (hi)": "hi",
}

# Expressive 12-15s reference scripts (question + pause + punchy payoff) so Chatterbox learns natural pitch range
SAMPLE_TEXTS = {
    "es": (
        "¿Alguna vez te has preguntado por qué algunas historias se quedan contigo para siempre? "
        "No es solo por lo que pasó... sino por la forma en que se cuenta. "
        "Hoy vamos a descubrir un detalle fascinante que casi nadie nota, ¡y te aseguro que te va a sorprender!"
    ),
    "en-us": (
        "Have you ever wondered why some stories stay with you forever? "
        "It is not just about what happened... it is about how the truth unfolds. "
        "Today, we are going to uncover a fascinating hidden detail that almost everyone overlooks!"
    ),
    "en-gb": (
        "Have you ever stopped to think about how a single quiet decision can change history? "
        "Behind every famous turning point... there is a story nobody ever told you. "
        "Let us take a closer look at what actually happened."
    ),
    "fr-fr": (
        "Vous êtes-vous déjà demandé pourquoi certaines histoires restent gravées dans notre mémoire ? "
        "Ce n'est pas seulement ce qui s'est passé... mais la manière dont on le découvre. "
        "Aujourd'hui, nous allons explorer un secret fascinant que presque personne ne connaît !"
    ),
    "it": (
        "Ti sei mai chiesto perché alcune storie restano impresse per sempre? "
        "Non è solo per ciò che è successo... ma per il modo in cui viene raccontato. "
        "Oggi scopriremo insieme un dettaglio sorprendente che quasi nessuno conosce!"
    ),
    "pt-br": (
        "Você já parou para pensar por que algumas histórias ficam na nossa mente para sempre? "
        "Não é apenas o que aconteceu... mas a forma como a verdade aparece. "
        "Hoje vamos descobrir um detalhe fascinante que quase ninguém percebe!"
    ),
}

CHATTERBOX_LANGS = {
    "Tây Ban Nha (es)": "es",
    "Tiếng Anh (en)": "en",
    "Tiếng Đức (de)": "de",
    "Tiếng Pháp (fr)": "fr",
    "Tiếng Ý (it)": "it",
    "Bồ Đào Nha (pt)": "pt",
    "Hà Lan (nl)": "nl",
    "Ba Lan (pl)": "pl",
    "Thụy Điển (sv)": "sv",
    "Na Uy (no)": "no",
    "Đan Mạch (da)": "da",
    "Phần Lan (fi)": "fi",
    "Hy Lạp (el)": "el",
    "Thổ Nhĩ Kỳ (tr)": "tr",
    "Tiếng Nga (ru)": "ru",
    "Ả Rập (ar)": "ar",
    "Tiếng Nhật (ja)": "ja",
    "Tiếng Hàn (ko)": "ko",
    "Tiếng Trung (zh)": "zh",
    "Tiếng Hindi (hi)": "hi",
    "Do Thái (he)": "he",
    "Mã Lai (ms)": "ms",
    "Swahili (sw)": "sw",
}

# One-click natural voice presets for Chatterbox Multilingual
CHATTERBOX_PRESETS = {
    "Kể chuyện YouTube tự nhiên (Khuyên dùng)": (0.55, 0.45, 0.75, 1.80, 0.28),
    "Tài liệu / Phân tích trầm ấm": (0.40, 0.50, 0.65, 1.80, 0.35),
    "Kịch tính / Review / Top List năng lượng": (0.75, 0.40, 0.85, 1.90, 0.22),
    "Clone xuyên ngôn ngữ (không dính giọng gốc)": (0.50, 0.15, 0.75, 2.00, 0.28),
}

_kokoro_instance = None
_chatterbox_instance = None


# ==================== TEXT CLEANING & CHUNKING ====================

def clean_script_text(text: str) -> str:
    """Clean pasted AI/Markdown scripts so TTS doesn't stumble or sound robotic."""
    if not text:
        return ""
    s = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove URLs
    s = re.sub(r"https?://\S+", "", s)
    # Remove markdown headings, bold/italic, inline code, blockquotes, bullets
    s = re.sub(r"(?m)^\s*#{1,6}\s+", "", s)
    s = re.sub(r"(?m)^\s*[>\-\*\•\●\▪]\s+", "", s)
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)
    s = re.sub(r"__(.*?)__", r"\1", s)
    s = re.sub(r"(?<!\w)\*(?!\s)(.*?)(?<!\s)\*(?!\w)", r"\1", s)
    s = s.replace("`", "")
    # Normalize typography for natural pauses
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s = s.replace("—", ", ").replace("–", ", ")
    s = re.sub(r"\.{4,}", "...", s)
    # Remove emojis / dingbats while preserving all world scripts & punctuation
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)
    # Collapse extra spaces per line, preserve paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def split_script_into_chunks(text: str, max_chars: int = 240):
    """Split long YouTube script into natural sentence-aligned chunks with punctuation & paragraph metadata."""
    cleaned = clean_script_text(text)
    paragraphs = [p.strip() for p in re.split(r"\n+", cleaned) if p.strip()]
    chunks = []
    for p_idx, para in enumerate(paragraphs):
        sentences = re.split(r"(?<=[.!?…])\s+", para)
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                subparts = re.split(r"(?<=[,;:])\s+", sent)
                for sub in subparts:
                    sub = sub.strip()
                    if not sub:
                        continue
                    if current and len(current) + 1 + len(sub) > max_chars:
                        chunks.append((current, False))
                        current = sub
                    else:
                        current = f"{current} {sub}".strip() if current else sub
            else:
                if current and len(current) + 1 + len(sent) > max_chars:
                    chunks.append((current, False))
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
        if current:
            chunks.append((current, p_idx < len(paragraphs) - 1))
    return chunks


# ==================== STUDIO AUDIO POST-PROCESSING ====================

def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -42.0, pad_ms: float = 35.0) -> np.ndarray:
    """Trim excessive leading/trailing silence while keeping a natural breath margin."""
    if audio.size == 0:
        return audio
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = audio - float(np.mean(audio))  # Remove DC offset
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    thresh = peak * (10.0 ** (threshold_db / 20.0))
    idx = np.where(np.abs(audio) > thresh)[0]
    if idx.size == 0:
        return audio
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, int(idx[0]) - pad)
    end = min(audio.size, int(idx[-1]) + pad)
    return audio[start:end]


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float = 15.0) -> np.ndarray:
    """Apply short cosine fade-in/out to prevent clicks between stitched chunks."""
    n = audio.size
    fade_len = min(int(sr * fade_ms / 1000.0), n // 4)
    if fade_len <= 1:
        return audio
    out = audio.copy()
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade_len, dtype=np.float32)))
    out[:fade_len] *= ramp
    out[-fade_len:] *= ramp[::-1]
    return out


def normalize_chunk_rms(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Level each sentence chunk so volume never jumps awkwardly between sentences."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    target_rms = 10.0 ** (target_db / 20.0)
    gain = min(target_rms / rms, 4.0)  # Cap max boost to avoid raising noise floor
    out = audio * gain
    peak = float(np.max(np.abs(out)))
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out.astype(np.float32)


def make_natural_pause(
    base_sec: float,
    sr: int,
    prev_text: str,
    is_para_end: bool,
    rng: np.random.Generator,
    use_room_tone: bool = True,
) -> tuple[np.ndarray, float]:
    """Generate human-like variable pause (punctuation-aware + subtle jitter + ultra-soft booth floor)."""
    t = (prev_text or "").rstrip()
    mult = 1.0
    if is_para_end:
        mult = 1.75
    elif t.endswith("...") or t.endswith("…"):
        mult = 1.35
    elif t.endswith("?") or t.endswith("!"):
        mult = 1.15
    elif t.endswith(",") or t.endswith(";") or t.endswith(":"):
        mult = 0.60

    jitter = float(rng.uniform(0.88, 1.14))
    gap = max(0.08, float(base_sec) * mult * jitter)
    n_samples = int(round(gap * sr))
    if not use_room_tone or n_samples == 0:
        return np.zeros(n_samples, dtype=np.float32), n_samples / sr

    # Ultra-subtle -70 dBFS room tone avoids artificial "dead digital vacuum" between sentences
    room = rng.normal(0.0, 10.0 ** (-70.0 / 20.0), size=n_samples).astype(np.float32)
    return room, n_samples / sr


def highpass_70hz(audio: np.ndarray, sr: int) -> np.ndarray:
    """Lightweight 1st-order high-pass filter (~70 Hz) in pure NumPy to clean muddy sub-rumble."""
    if audio.size < 4:
        return audio
    rc = 1.0 / (2.0 * np.pi * 70.0)
    dt = 1.0 / float(sr)
    alpha = rc / (rc + dt)
    out = np.empty_like(audio, dtype=np.float32)
    out[0] = audio[0]
    # Vectorized difference + simple loop only if short; for long tracks use fast chunked IIR via scipy if available
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(2, 70.0, btype="highpass", fs=sr, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        # Fast fallback: DC + gentle sub-bass subtraction via moving average (~14ms window)
        win = max(3, int(sr / 70.0))
        kernel = np.ones(win, dtype=np.float32) / win
        low = np.convolve(audio, kernel, mode="same")
        return (audio - low * 0.35).astype(np.float32)


def master_youtube_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Gentle YouTube dialogue mastering: sub-rumble cleanup, warm loudness leveling, soft limiter (-1 dBFS)."""
    if audio.size == 0:
        return audio
    x = highpass_70hz(audio.astype(np.float32), sr)
    # Measure active speech RMS (ignore silent gaps)
    active = x[np.abs(x) > 0.005]
    rms = float(np.sqrt(np.mean(active ** 2))) if active.size > 100 else float(np.sqrt(np.mean(x ** 2)))
    if rms > 1e-6:
        # Target ~ -16 dBFS active RMS (sits right around YouTube -14 LUFS loudness)
        target_rms = 10.0 ** (-16.0 / 20.0)
        x = x * min(target_rms / rms, 3.5)
    # Soft-knee limiter at -1 dBFS (0.891) so peaks never harshly clip
    ceiling = 0.891
    x = ceiling * np.tanh(x / ceiling)
    return x.astype(np.float32)


def process_chunk_audio(audio: np.ndarray, sr: int, enable_mastering: bool) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if not enable_mastering:
        return audio
    audio = trim_silence(audio, sr, threshold_db=-42.0, pad_ms=35.0)
    audio = normalize_chunk_rms(audio, target_db=-20.0)
    audio = apply_fade(audio, sr, fade_ms=15.0)
    return audio


# ==================== DOWNLOAD & MODEL LOADERS ====================

def download_file(url: str, dest: Path, label: str = ""):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[Download] {label or dest.name} <- {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"[Download] Xong {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def ensure_kokoro_models():
    download_file(KOKORO_MODEL_URL, KOKORO_MODEL_PATH, "kokoro-v1.0.onnx (~310 MB)")
    download_file(KOKORO_VOICES_URL, KOKORO_VOICES_PATH, "voices-v1.0.bin (~27 MB)")


def get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is None:
        ensure_kokoro_models()
        from kokoro_onnx import Kokoro
        _kokoro_instance = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
    return _kokoro_instance


def list_kokoro_voices():
    try:
        k = get_kokoro()
        return sorted(k.get_voices())
    except Exception:
        return [
            "em_alex", "em_santa", "ef_dora",
            "am_adam", "am_michael", "am_onyx", "am_fenrir", "af_heart", "af_bella", "af_sarah",
            "bm_george", "bm_lewis", "bm_daniel", "bf_emma", "bf_isabella",
            "ff_siwis", "im_nicola", "if_sara", "pm_alex", "pm_santa", "pf_dora",
        ]


def list_saved_voices():
    return sorted(
        [p.name for p in VOICES_DIR.glob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
    )


def format_srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    hrs, ms = divmod(ms, 3_600_000)
    mins, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def build_srt(entries) -> str:
    lines = []
    for idx, (start, end, txt) in enumerate(entries, start=1):
        lines.append(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{txt}\n")
    return "\n".join(lines)


def sanitize_filename(name: str, default: str = "custom_voice") -> str:
    clean = re.sub(r"[^a-zA-Z0-9_\-]+", "_", (name or "").strip()).strip("_")
    return clean or default


def get_device_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            return f"GPU sẵn sàng: {gpu_name} ({vram_gb:.1f} GB VRAM | Đang dùng: {alloc_gb:.2f} GB)"
        return "Đang chạy CPU (Chưa phát hiện CUDA GPU)"
    except Exception:
        return "Tab 1 (Kokoro-82M CPU) sẵn sàng | Chưa cài PyTorch CUDA cho Tab 2 trên máy này"


def free_vram():
    global _chatterbox_instance
    _chatterbox_instance = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return f"Đã giải phóng VRAM. {get_device_info()}"


def get_chatterbox():
    global _chatterbox_instance
    if _chatterbox_instance is None:
        try:
            import torch
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        except ImportError as exc:
            raise gr.Error(
                "Chưa cài gói Chatterbox Multilingual (CUDA). "
                "Hãy chạy file install_chatterbox_gpu.bat trên máy có RTX 2060 để cài thêm Tab 2!"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Chatterbox] Đang tải ChatterboxMultilingualTTS trên {device}...")
        _chatterbox_instance = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _chatterbox_instance


# ==================== TAB 1: KOKORO VOICE BLENDER ====================

def load_sample_for_lang(lang_label: str):
    code = KOKORO_LANGS.get(lang_label, "es")
    return SAMPLE_TEXTS.get(code, SAMPLE_TEXTS["es"])


def blend_kokoro_style(kokoro, voice1: str, voice2: str, ratio_v1: float):
    s1 = kokoro.get_voice_style(voice1)
    if not voice2 or voice2 == "(Không trộn)" or ratio_v1 >= 100:
        return s1
    if ratio_v1 <= 0:
        return kokoro.get_voice_style(voice2)
    s2 = kokoro.get_voice_style(voice2)
    w1 = ratio_v1 / 100.0
    w2 = 1.0 - w1
    return (s1 * w1) + (s2 * w2)


def generate_kokoro_sample(
    voice1: str,
    voice2: str,
    ratio_v1: float,
    lang_label: str,
    speed: float,
    sample_text: str,
    save_name: str,
    enable_mastering: bool = True,
):
    cleaned = clean_script_text(sample_text)
    if not cleaned:
        raise gr.Error("Vui lòng nhập đoạn văn bản mẫu (khoảng 2–3 câu, 10–15 giây).")

    kokoro = get_kokoro()
    lang_code = KOKORO_LANGS.get(lang_label, "es")
    style = blend_kokoro_style(kokoro, voice1, voice2, ratio_v1)

    t0 = time.time()
    audio, sr = kokoro.create(cleaned, voice=style, speed=float(speed), lang=lang_code)
    audio = process_chunk_audio(audio, sr, enable_mastering)
    if enable_mastering:
        audio = master_youtube_audio(audio, sr)
    dur = len(audio) / sr
    elapsed = time.time() - t0

    fname = sanitize_filename(save_name, f"{voice1}_{voice2}_{int(ratio_v1)}") + ".wav"
    out_path = VOICES_DIR / fname
    sf.write(str(out_path), audio, sr)

    saved_list = list_saved_voices()
    status = (
        f"Đã tạo và lưu giọng mẫu (đã mastering sạch): voices/{fname}\n"
        f"Độ dài audio: {dur:.1f}s (tối ưu cho clone: 10–15s) | Thời gian render: {elapsed:.2f}s\n"
        f"Giờ bạn có thể chuyển sang Tab 2 để nhân bản giọng này sang 23 ngôn ngữ!"
    )
    return str(out_path), status, gr.Dropdown(choices=saved_list, value=fname)


def generate_kokoro_long(
    voice1: str,
    voice2: str,
    ratio_v1: float,
    lang_label: str,
    speed: float,
    pause_sec: float,
    enable_mastering: bool,
    script_text: str,
    progress=gr.Progress(),
):
    chunks = split_script_into_chunks(script_text, max_chars=240)
    if not chunks:
        raise gr.Error("Vui lòng nhập kịch bản cần đọc.")

    kokoro = get_kokoro()
    lang_code = KOKORO_LANGS.get(lang_label, "es")
    style = blend_kokoro_style(kokoro, voice1, voice2, ratio_v1)
    rng = np.random.default_rng(42)

    segments = []
    srt_entries = []
    cursor = 0.0
    sr = 24000
    t0 = time.time()

    for idx, (chunk_text, is_para_end) in enumerate(chunks):
        progress((idx + 1) / len(chunks), desc=f"Kokoro đang đọc đoạn {idx + 1}/{len(chunks)}...")
        audio, sr = kokoro.create(chunk_text, voice=style, speed=float(speed), lang=lang_code)
        audio = process_chunk_audio(audio, sr, enable_mastering)
        dur = len(audio) / sr
        srt_entries.append((cursor, cursor + dur, chunk_text))
        segments.append(audio)
        cursor += dur

        if idx < len(chunks) - 1:
            pause_arr, actual_gap = make_natural_pause(
                pause_sec, sr, chunk_text, is_para_end, rng, use_room_tone=enable_mastering
            )
            segments.append(pause_arr)
            cursor += actual_gap

    full_audio = np.concatenate(segments)
    if enable_mastering:
        full_audio = master_youtube_audio(full_audio, sr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUTS_DIR / f"kokoro_{lang_code}_{stamp}.wav"
    srt_path = OUTPUTS_DIR / f"kokoro_{lang_code}_{stamp}.srt"

    sf.write(str(wav_path), full_audio, sr)
    srt_path.write_text(build_srt(srt_entries), encoding="utf-8")

    elapsed = time.time() - t0
    status = (
        f"Xong {len(chunks)} đoạn | Tổng độ dài: {cursor:.1f}s | Render mất: {elapsed:.2f}s"
        f" | Mastering: {'Bật' if enable_mastering else 'Tắt'}\n"
        f"Audio: {wav_path}\nPhụ đề: {srt_path}"
    )
    return str(wav_path), str(srt_path), status


# ==================== TAB 2: CHATTERBOX MULTILINGUAL CLONER ====================

def apply_chatterbox_preset(preset_name: str):
    vals = CHATTERBOX_PRESETS.get(preset_name)
    if not vals:
        return 0.55, 0.45, 0.75, 1.80, 0.28
    return vals


def is_chunk_duration_sane(chunk_text: str, dur_sec: float) -> bool:
    """Detect Chatterbox autoregressive glitches (swallowed audio or runaway looping tail)."""
    n_chars = max(1, len(chunk_text.strip()))
    min_expected = max(0.35, n_chars / 32.0)   # >32 chars/sec is impossibly fast (cut off)
    max_expected = max(6.0, n_chars / 4.0)     # <4 chars/sec is likely hallucination/looping
    return min_expected <= dur_sec <= max_expected


def generate_chatterbox_long(
    saved_voice_name: str,
    uploaded_voice_path: str,
    lang_label: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    pause_sec: float,
    max_chunk_chars: int,
    enable_mastering: bool,
    auto_retry_glitch: bool,
    script_text: str,
    progress=gr.Progress(),
):
    chunks = split_script_into_chunks(script_text, max_chars=int(max_chunk_chars))
    if not chunks:
        raise gr.Error("Vui lòng nhập kịch bản cần đọc.")

    ref_path = None
    if uploaded_voice_path:
        ref_path = Path(uploaded_voice_path)
    elif saved_voice_name:
        ref_path = VOICES_DIR / saved_voice_name

    if not ref_path or not ref_path.exists():
        raise gr.Error("Chưa chọn giọng mẫu! Hãy tạo 1 giọng mẫu ở Tab 1 hoặc upload file .wav 10–15s.")

    import torch

    progress(0.05, desc="Đang tải model Chatterbox Multilingual (lần đầu tải ~2GB)...")
    model = get_chatterbox()
    lang_code = CHATTERBOX_LANGS.get(lang_label, "es")
    rng = np.random.default_rng(42)

    # Check supported kwargs dynamically for compatibility across chatterbox-tts versions
    gen_params = set(inspect.signature(model.generate).parameters.keys())

    # Pre-compute speaker conditionals once if supported (faster + consistent tone across long scripts)
    prepped = False
    if hasattr(model, "prepare_conditionals"):
        try:
            model.prepare_conditionals(str(ref_path), exaggeration=float(exaggeration))
            prepped = True
        except Exception:
            prepped = False

    segments = []
    srt_entries = []
    cursor = 0.0
    sr = model.sr
    retried_count = 0
    t0 = time.time()

    for idx, (chunk_text, is_para_end) in enumerate(chunks):
        progress(
            0.1 + 0.88 * ((idx + 1) / len(chunks)),
            desc=f"Chatterbox đang clone đoạn {idx + 1}/{len(chunks)} ({lang_code})...",
        )
        max_attempts = 3 if auto_retry_glitch else 1
        best_audio = None

        for attempt in range(max_attempts):
            kwargs = {
                "text": chunk_text,
                "language_id": lang_code,
            }
            if not prepped:
                kwargs["audio_prompt_path"] = str(ref_path)
            if "exaggeration" in gen_params:
                kwargs["exaggeration"] = float(exaggeration)
            if "cfg_weight" in gen_params:
                kwargs["cfg_weight"] = float(cfg_weight)
            if "temperature" in gen_params:
                # Slightly lower temperature on retry for cleaner stability
                kwargs["temperature"] = max(0.35, float(temperature) - 0.1 * attempt)
            if "repetition_penalty" in gen_params:
                kwargs["repetition_penalty"] = float(repetition_penalty)

            with torch.inference_mode():
                wav_tensor = model.generate(**kwargs)
            cand = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            cand = process_chunk_audio(cand, sr, enable_mastering)
            best_audio = cand
            if is_chunk_duration_sane(chunk_text, len(cand) / sr):
                break
            if attempt < max_attempts - 1:
                retried_count += 1

        dur = len(best_audio) / sr
        srt_entries.append((cursor, cursor + dur, chunk_text))
        segments.append(best_audio)
        cursor += dur

        if idx < len(chunks) - 1:
            pause_arr, actual_gap = make_natural_pause(
                pause_sec, sr, chunk_text, is_para_end, rng, use_room_tone=enable_mastering
            )
            segments.append(pause_arr)
            cursor += actual_gap

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_audio = np.concatenate(segments)
    if enable_mastering:
        full_audio = master_youtube_audio(full_audio, sr)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUTS_DIR / f"chatterbox_{lang_code}_{stamp}.wav"
    srt_path = OUTPUTS_DIR / f"chatterbox_{lang_code}_{stamp}.srt"

    sf.write(str(wav_path), full_audio, sr)
    srt_path.write_text(build_srt(srt_entries), encoding="utf-8")

    elapsed = time.time() - t0
    status = (
        f"Hoàn tất {len(chunks)} đoạn ({lang_code}) | Tổng độ dài: {cursor:.1f}s | Render mất: {elapsed:.1f}s\n"
        f"Tự động đọc lại đoạn lỗi: {retried_count} lần | Mastering: {'Bật' if enable_mastering else 'Tắt'}\n"
        f"Giọng mẫu dùng: {ref_path.name}\n"
        f"Audio: {wav_path}\nPhụ đề: {srt_path}\n{get_device_info()}"
    )
    return str(wav_path), str(srt_path), status


# ==================== UI BUILDER ====================

def build_ui():
    kokoro_voices = list_kokoro_voices()
    default_v1 = "em_alex" if "em_alex" in kokoro_voices else kokoro_voices[0]
    default_v2 = "am_adam" if "am_adam" in kokoro_voices else (kokoro_voices[1] if len(kokoro_voices) > 1 else "(Không trộn)")
    voice2_choices = ["(Không trộn)"] + kokoro_voices
    saved_voices = list_saved_voices()

    with gr.Blocks(title="YouTube Local TTS Studio (Cách 2)", analytics_enabled=False) as demo:
        gr.Markdown(
            "# YouTube Local TTS Studio — Cách 2 (Trộn Giọng Bản Xứ + Nhân Bản 23 Ngôn Ngữ)\n"
            "Chạy **100% offline trên máy** (tối ưu cho **RTX 2060 6GB**), tích hợp **YouTube Mastering** "
            "(cắt khoảng lặng dư, cân bằng âm lượng, nhịp nghỉ tự nhiên, tự lọc markdown & chống cà lăm)."
        )

        with gr.Row():
            gpu_status = gr.Textbox(value=get_device_info(), label="Trạng thái phần cứng", interactive=False, scale=4)
            btn_free_vram = gr.Button("Giải phóng VRAM", scale=1)
        btn_free_vram.click(fn=free_vram, outputs=[gpu_status])

        with gr.Tabs():
            # -------------------- TAB 1 --------------------
            with gr.Tab("Bước 1: Tạo Giọng Độc Quyền (Kokoro-82M)"):
                gr.Markdown(
                    "**Mẹo chọn giọng trộn nghe tự nhiên nhất cho YouTube:**\n"
                    "- Nam Tây Ban Nha trầm ấm: `em_alex` (60%) + `am_adam` hoặc `bm_george` (40%)\n"
                    "- Nam Tây Ban Nha năng lượng: `em_alex` (65%) + `em_santa` (35%)\n"
                    "- Nữ Tây Ban Nha / Anh mượt: `ef_dora` (60%) + `af_heart` hoặc `bf_emma` (40%)"
                )
                with gr.Row():
                    k_v1 = gr.Dropdown(choices=kokoro_voices, value=default_v1, label="Giọng chính (Giọng 1)")
                    k_v2 = gr.Dropdown(choices=voice2_choices, value=default_v2, label="Giọng pha trộn (Giọng 2)")
                    k_ratio = gr.Slider(0, 100, value=60, step=5, label="Tỷ lệ Giọng 1 (%)")
                with gr.Row():
                    k_lang = gr.Dropdown(
                        choices=list(KOKORO_LANGS.keys()),
                        value="Tây Ban Nha (es)",
                        label="Ngôn ngữ đọc",
                    )
                    k_speed = gr.Slider(0.7, 1.3, value=0.96, step=0.02, label="Tốc độ đọc (0.94–0.98 nghe chậm rãi, thật hơn)")
                    k_save_name = gr.Textbox(value="giong_tay_ban_nha_01", label="Tên lưu file giọng mẫu")

                k_sample_text = gr.Textbox(
                    value=SAMPLE_TEXTS["es"],
                    lines=5,
                    label="Đoạn văn bản mẫu (có câu hỏi + điểm nhấn giúp Tab 2 học ngữ điệu sống động hơn, hoặc dán cả kịch bản dài vào đây)",
                )
                k_lang.change(fn=load_sample_for_lang, inputs=[k_lang], outputs=[k_sample_text])

                with gr.Row():
                    k_master = gr.Checkbox(value=True, label="Bật Mastering phòng thu YouTube (cắt lặng, cân bằng âm lượng, nghỉ tự nhiên)")
                    k_pause = gr.Slider(0.1, 1.0, value=0.26, step=0.02, label="Nhịp nghỉ cơ sở giữa câu (giây)")

                with gr.Row():
                    btn_make_sample = gr.Button("Tạo & Lưu làm Giọng Mẫu (dùng cho Tab 2)", variant="primary")
                    btn_kokoro_long = gr.Button("Xuất nhanh kịch bản dài + SRT bằng Kokoro")

                with gr.Row():
                    k_audio_out = gr.Audio(label="Nghe thử Audio", type="filepath")
                    k_srt_out = gr.File(label="Tải phụ đề (.srt)")
                k_status = gr.Textbox(label="Kết quả", lines=3, interactive=False)

            # -------------------- TAB 2 --------------------
            with gr.Tab("Bước 2: Clone Giọng Tự Nhiên 23 Ngôn Ngữ (Chatterbox RTX 2060)"):
                gr.Markdown(
                    "Lấy file giọng mẫu độc quyền ở **Bước 1** để đọc kịch bản dài với **hơi thở & cảm xúc giống người thật** cho **23 ngôn ngữ**.\n"
                    "*(Mẹo: Khi dùng giọng Tây Ban Nha để đọc sang tiếng Đức / Ba Lan / Hà Lan…, hãy chọn preset **Clone xuyên ngôn ngữ** để không bị dính giọng gốc.)*"
                )
                with gr.Row():
                    cb_saved_voice = gr.Dropdown(
                        choices=saved_voices,
                        value=saved_voices[0] if saved_voices else None,
                        label="Chọn giọng mẫu đã lưu (từ Tab 1)",
                        scale=3,
                    )
                    btn_refresh_voices = gr.Button("Tải lại danh sách", scale=1)
                    cb_upload_voice = gr.Audio(
                        label="Hoặc upload file giọng mẫu khác (10–15s .wav)",
                        type="filepath",
                        scale=3,
                    )

                with gr.Row():
                    cb_lang = gr.Dropdown(
                        choices=list(CHATTERBOX_LANGS.keys()),
                        value="Tây Ban Nha (es)",
                        label="Ngôn ngữ xuất ra (23 nước)",
                    )
                    cb_preset = gr.Dropdown(
                        choices=list(CHATTERBOX_PRESETS.keys()),
                        value="Kể chuyện YouTube tự nhiên (Khuyên dùng)",
                        label="Preset phong cách đọc (bấm chọn nhanh)",
                    )

                with gr.Row():
                    cb_exag = gr.Slider(
                        0.2, 1.2, value=0.55, step=0.05,
                        label="Cảm xúc / Nhấn nhá (Exaggeration)",
                    )
                    cb_cfg = gr.Slider(
                        0.0, 1.0, value=0.45, step=0.05,
                        label="Độ bám giọng & nhịp (CFG — giảm 0.0–0.2 nếu xuyên ngôn ngữ)",
                    )
                    cb_temp = gr.Slider(
                        0.3, 1.2, value=0.75, step=0.05,
                        label="Độ biến hóa ngữ điệu (Temperature)",
                    )
                    cb_rep = gr.Slider(
                        1.0, 2.5, value=1.80, step=0.1,
                        label="Chống cà lăm (Repetition Penalty)",
                    )

                with gr.Row():
                    cb_pause = gr.Slider(0.1, 1.0, value=0.28, step=0.02, label="Nhịp nghỉ cơ sở giữa câu (giây)")
                    cb_chunk = gr.Slider(
                        150, 360, value=240, step=10,
                        label="Độ dài cắt đoạn (ký tự — giữ ~240 cho RTX 2060 6GB)",
                    )
                    cb_master = gr.Checkbox(value=True, label="Mastering phòng thu YouTube")
                    cb_retry = gr.Checkbox(value=True, label="Tự đọc lại nếu phát hiện lỗi/cà lăm")

                cb_preset.change(
                    fn=apply_chatterbox_preset,
                    inputs=[cb_preset],
                    outputs=[cb_exag, cb_cfg, cb_temp, cb_rep, cb_pause],
                )

                cb_script = gr.Textbox(
                    value=SAMPLE_TEXTS["es"],
                    lines=8,
                    label="Kịch bản YouTube (dán thẳng từ ChatGPT/Gemini — app tự lọc markdown, cắt câu mượt & xuất kèm .srt)",
                )

                btn_cb_run = gr.Button("Bắt đầu Clone & Xuất Audio + SRT (GPU)", variant="primary")

                with gr.Row():
                    cb_audio_out = gr.Audio(label="Audio hoàn chỉnh", type="filepath")
                    cb_srt_out = gr.File(label="File phụ đề (.srt)")
                cb_status = gr.Textbox(label="Trạng thái & Đường dẫn file", lines=5, interactive=False)

        # Wire events
        btn_make_sample.click(
            fn=generate_kokoro_sample,
            inputs=[k_v1, k_v2, k_ratio, k_lang, k_speed, k_sample_text, k_save_name, k_master],
            outputs=[k_audio_out, k_status, cb_saved_voice],
        )
        btn_kokoro_long.click(
            fn=generate_kokoro_long,
            inputs=[k_v1, k_v2, k_ratio, k_lang, k_speed, k_pause, k_master, k_sample_text],
            outputs=[k_audio_out, k_srt_out, k_status],
        )
        btn_refresh_voices.click(
            fn=lambda: gr.Dropdown(choices=list_saved_voices()),
            outputs=[cb_saved_voice],
        )
        btn_cb_run.click(
            fn=generate_chatterbox_long,
            inputs=[
                cb_saved_voice, cb_upload_voice, cb_lang,
                cb_exag, cb_cfg, cb_temp, cb_rep,
                cb_pause, cb_chunk, cb_master, cb_retry, cb_script,
            ],
            outputs=[cb_audio_out, cb_srt_out, cb_status],
        )

    return demo


if __name__ == "__main__":
    print("=== YouTube Local TTS Studio ===")
    print(get_device_info())
    app = build_ui()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        theme=gr.themes.Soft(),
    )
