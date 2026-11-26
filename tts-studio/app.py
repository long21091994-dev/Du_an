import os
import re
import gc
import time
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

SAMPLE_TEXTS = {
    "es": (
        "En un mundo que cambia cada segundo, las historias mejor contadas son las que logran "
        "detener el tiempo. Hoy vamos a descubrir un secreto fascinante que muy pocas personas "
        "conocen, explicado paso a paso de una manera clara, directa y fácil de recordar."
    ),
    "en-us": (
        "In a world that moves faster every single day, the best stories are the ones that make "
        "you pause and think. Today, we are going to uncover a fascinating hidden detail that "
        "most people overlook, broken down clearly step by step."
    ),
    "en-gb": (
        "History is rarely as simple as the textbooks make it sound. Behind every famous turning "
        "point lies a quiet decision that changed everything. Let us take a closer look at how it "
        "actually unfolded."
    ),
    "fr-fr": (
        "Dans un monde en constante évolution, les meilleures histoires sont celles qui captent "
        "notre attention dès la première seconde. Aujourd'hui, nous allons explorer un sujet "
        "fascinant expliqué de manière claire et captivante."
    ),
    "it": (
        "In un mondo che cambia continuamente, le storie migliori sono quelle che riescono a "
        "fermare il tempo. Oggi scopriremo insieme un dettaglio affascinante che quasi nessuno "
        "conosce davvero."
    ),
    "pt-br": (
        "Em um mundo que muda a cada segundo, as melhores histórias são aquelas que fazem você "
        "parar para pensar. Hoje vamos descobrir um detalhe fascinante explicado de forma clara "
        "e direta."
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

_kokoro_instance = None
_chatterbox_instance = None


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
        voices = sorted(k.get_voices())
        return voices
    except Exception:
        return [
            "em_alex", "em_santa", "ef_dora",
            "am_adam", "am_michael", "am_onyx", "am_fenrir", "af_heart", "af_bella", "af_sarah",
            "bm_george", "bm_lewis", "bm_daniel", "bf_emma", "bf_isabella",
            "ff_siwis", "im_nicola", "if_sara", "pm_alex", "pm_santa", "pf_dora",
        ]


def list_saved_voices():
    files = sorted(
        [p.name for p in VOICES_DIR.glob("*") if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")]
    )
    return files


def split_script_into_chunks(text: str, max_chars: int = 260):
    """Split long YouTube script into natural sentence-aligned chunks with paragraph pause markers."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text.strip()) if p.strip()]
    chunks = []
    for p_idx, para in enumerate(paragraphs):
        # Split on sentence end punctuation while keeping Spanish ¿ ¡ intact
        sentences = re.split(r"(?<=[.!?…])\s+", para)
        current = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                # Fallback split by comma/semicolon if a single sentence is huge
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
    except Exception as e:
        return f"Không kiểm tra được PyTorch/GPU: {e}"


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
):
    if not sample_text or not sample_text.strip():
        raise gr.Error("Vui lòng nhập đoạn văn bản mẫu (khoảng 2–3 câu, 10–15 giây).")

    kokoro = get_kokoro()
    lang_code = KOKORO_LANGS.get(lang_label, "es")
    style = blend_kokoro_style(kokoro, voice1, voice2, ratio_v1)

    t0 = time.time()
    audio, sr = kokoro.create(sample_text.strip(), voice=style, speed=float(speed), lang=lang_code)
    dur = len(audio) / sr
    elapsed = time.time() - t0

    fname = sanitize_filename(save_name, f"{voice1}_{voice2}_{int(ratio_v1)}") + ".wav"
    out_path = VOICES_DIR / fname
    sf.write(str(out_path), audio, sr)

    saved_list = list_saved_voices()
    status = (
        f"Đã tạo và lưu giọng mẫu: voices/{fname}\n"
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
    script_text: str,
    progress=gr.Progress(),
):
    if not script_text or not script_text.strip():
        raise gr.Error("Vui lòng nhập kịch bản cần đọc.")

    kokoro = get_kokoro()
    lang_code = KOKORO_LANGS.get(lang_label, "es")
    style = blend_kokoro_style(kokoro, voice1, voice2, ratio_v1)
    chunks = split_script_into_chunks(script_text, max_chars=260)

    segments = []
    srt_entries = []
    cursor = 0.0
    sr = 24000
    t0 = time.time()

    for idx, (chunk_text, is_para_end) in enumerate(chunks):
        progress((idx + 1) / len(chunks), desc=f"Kokoro đang đọc đoạn {idx + 1}/{len(chunks)}...")
        audio, sr = kokoro.create(chunk_text, voice=style, speed=float(speed), lang=lang_code)
        dur = len(audio) / sr
        srt_entries.append((cursor, cursor + dur, chunk_text))
        segments.append(audio)
        cursor += dur

        if idx < len(chunks) - 1:
            gap = float(pause_sec) * (1.8 if is_para_end else 1.0)
            segments.append(np.zeros(int(gap * sr), dtype=np.float32))
            cursor += gap

    full_audio = np.concatenate(segments)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUTS_DIR / f"kokoro_{lang_code}_{stamp}.wav"
    srt_path = OUTPUTS_DIR / f"kokoro_{lang_code}_{stamp}.srt"

    sf.write(str(wav_path), full_audio, sr)
    srt_path.write_text(build_srt(srt_entries), encoding="utf-8")

    elapsed = time.time() - t0
    status = (
        f"Xong {len(chunks)} đoạn | Tổng độ dài: {cursor:.1f}s | Render mất: {elapsed:.2f}s\n"
        f"Audio: {wav_path}\nPhụ đề: {srt_path}"
    )
    return str(wav_path), str(srt_path), status


# ==================== TAB 2: CHATTERBOX MULTILINGUAL CLONER ====================

def generate_chatterbox_long(
    saved_voice_name: str,
    uploaded_voice_path: str,
    lang_label: str,
    exaggeration: float,
    cfg_weight: float,
    pause_sec: float,
    max_chunk_chars: int,
    script_text: str,
    progress=gr.Progress(),
):
    if not script_text or not script_text.strip():
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
    chunks = split_script_into_chunks(script_text, max_chars=int(max_chunk_chars))

    segments = []
    srt_entries = []
    cursor = 0.0
    sr = model.sr
    t0 = time.time()

    for idx, (chunk_text, is_para_end) in enumerate(chunks):
        progress(
            0.1 + 0.88 * ((idx + 1) / len(chunks)),
            desc=f"Chatterbox đang clone đoạn {idx + 1}/{len(chunks)} ({lang_code})...",
        )
        with torch.inference_mode():
            wav_tensor = model.generate(
                text=chunk_text,
                language_id=lang_code,
                audio_prompt_path=str(ref_path),
                exaggeration=float(exaggeration),
                cfg_weight=float(cfg_weight),
            )
        audio = wav_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
        dur = len(audio) / sr
        srt_entries.append((cursor, cursor + dur, chunk_text))
        segments.append(audio)
        cursor += dur

        if idx < len(chunks) - 1:
            gap = float(pause_sec) * (1.8 if is_para_end else 1.0)
            segments.append(np.zeros(int(gap * sr), dtype=np.float32))
            cursor += gap

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_audio = np.concatenate(segments)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = OUTPUTS_DIR / f"chatterbox_{lang_code}_{stamp}.wav"
    srt_path = OUTPUTS_DIR / f"chatterbox_{lang_code}_{stamp}.srt"

    sf.write(str(wav_path), full_audio, sr)
    srt_path.write_text(build_srt(srt_entries), encoding="utf-8")

    elapsed = time.time() - t0
    status = (
        f"Hoàn tất {len(chunks)} đoạn ({lang_code}) | Tổng độ dài: {cursor:.1f}s | Render mất: {elapsed:.1f}s\n"
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

    with gr.Blocks(title="YouTube Local TTS Studio (Cách 2)") as demo:
        gr.Markdown(
            "# YouTube Local TTS Studio — Cách 2 (Trộn Giọng Bản Xứ + Nhân Bản 23 Ngôn Ngữ)\n"
            "Chạy **100% offline trên máy** (tối ưu cho **RTX 2060 6GB**). "
            "Bước 1: Trộn 2 giọng bản xứ thành giọng độc quyền ở **Tab 1** → Bước 2: Đọc kịch bản giàu cảm xúc cho **23 quốc gia** ở **Tab 2**."
        )

        with gr.Row():
            gpu_status = gr.Textbox(value=get_device_info(), label="Trạng thái phần cứng", interactive=False, scale=4)
            btn_free_vram = gr.Button("Giải phóng VRAM", scale=1)
        btn_free_vram.click(fn=free_vram, outputs=[gpu_status])

        with gr.Tabs():
            # -------------------- TAB 1 --------------------
            with gr.Tab("Bước 1: Tạo Giọng Độc Quyền (Kokoro-82M)"):
                gr.Markdown(
                    "**Mẹo chọn giọng trộn cực chất cho YouTube:**\n"
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
                    k_speed = gr.Slider(0.7, 1.3, value=0.98, step=0.02, label="Tốc độ đọc")
                    k_save_name = gr.Textbox(value="giong_tay_ban_nha_01", label="Tên lưu file giọng mẫu")

                k_sample_text = gr.Textbox(
                    value=SAMPLE_TEXTS["es"],
                    lines=4,
                    label="Đoạn văn bản mẫu (10–15 giây để làm giọng gốc, hoặc dán cả kịch bản dài vào đây)",
                )
                k_lang.change(fn=load_sample_for_lang, inputs=[k_lang], outputs=[k_sample_text])

                with gr.Row():
                    btn_make_sample = gr.Button("Tạo & Lưu làm Giọng Mẫu (dùng cho Tab 2)", variant="primary")
                    btn_kokoro_long = gr.Button("Xuất nhanh kịch bản dài + SRT bằng Kokoro")
                    k_pause = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Nghỉ giữa câu (giây)")

                with gr.Row():
                    k_audio_out = gr.Audio(label="Nghe thử Audio", type="filepath")
                    k_srt_out = gr.File(label="Tải phụ đề (.srt)")
                k_status = gr.Textbox(label="Kết quả", lines=3, interactive=False)

            # -------------------- TAB 2 --------------------
            with gr.Tab("Bước 2: Clone Giọng & Đọc 23 Ngôn Ngữ (Chatterbox RTX 2060)"):
                gr.Markdown(
                    "Lấy file giọng mẫu độc quyền vừa tạo ở **Bước 1** để đọc kịch bản dài với **cảm xúc tự nhiên** cho "
                    "**23 ngôn ngữ** (Tây Ban Nha, Anh, Đức, Pháp, Ý, Bồ Đào Nha, Hà Lan, Ba Lan, Thụy Điển, Na Uy…).\n"
                    "*(Lần bấm đầu tiên sẽ tự tải model Chatterbox ~2GB vào thư mục `d:\\Du_an\\tts-studio\\models` — từ lần sau chạy hoàn toàn offline.)*"
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
                    cb_exag = gr.Slider(
                        0.2, 1.2, value=0.55, step=0.05,
                        label="Độ cảm xúc / Nhấn nhá (0.5=tự nhiên, 0.7=kịch tính)",
                    )
                    cb_cfg = gr.Slider(
                        0.1, 1.0, value=0.45, step=0.05,
                        label="Độ bám giọng & nhịp (giảm xuống 0.3–0.4 nếu đọc quá nhanh)",
                    )

                with gr.Row():
                    cb_pause = gr.Slider(0.1, 1.0, value=0.28, step=0.05, label="Nghỉ giữa các câu (giây)")
                    cb_chunk = gr.Slider(
                        150, 380, value=250, step=10,
                        label="Độ dài cắt đoạn (ký tự — giữ ~250 cho RTX 2060 6GB)",
                    )

                cb_script = gr.Textbox(
                    value=SAMPLE_TEXTS["es"],
                    lines=8,
                    label="Kịch bản YouTube (dán kịch bản dài thoải mái — app tự cắt câu & xuất kèm file .srt)",
                )

                btn_cb_run = gr.Button("Bắt đầu Clone & Xuất Audio + SRT (GPU)", variant="primary")

                with gr.Row():
                    cb_audio_out = gr.Audio(label="Audio hoàn chỉnh", type="filepath")
                    cb_srt_out = gr.File(label="File phụ đề (.srt)")
                cb_status = gr.Textbox(label="Trạng thái & Đường dẫn file", lines=4, interactive=False)

        # Wire Tab 1 -> Tab 2 voice dropdown auto-refresh
        btn_make_sample.click(
            fn=generate_kokoro_sample,
            inputs=[k_v1, k_v2, k_ratio, k_lang, k_speed, k_sample_text, k_save_name],
            outputs=[k_audio_out, k_status, cb_saved_voice],
        )
        btn_kokoro_long.click(
            fn=generate_kokoro_long,
            inputs=[k_v1, k_v2, k_ratio, k_lang, k_speed, k_pause, k_sample_text],
            outputs=[k_audio_out, k_srt_out, k_status],
        )
        btn_refresh_voices.click(
            fn=lambda: gr.Dropdown(choices=list_saved_voices()),
            outputs=[cb_saved_voice],
        )
        btn_cb_run.click(
            fn=generate_chatterbox_long,
            inputs=[cb_saved_voice, cb_upload_voice, cb_lang, cb_exag, cb_cfg, cb_pause, cb_chunk, cb_script],
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
