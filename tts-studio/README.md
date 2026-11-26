# YouTube Local TTS Studio (Cách 2: Trộn giọng bản xứ + Nhân bản 23 ngôn ngữ)

Công cụ chạy **100% offline** trên máy Windows (tối ưu cho **RTX 2060 6GB**), tích hợp **YouTube Audio Mastering** giúp giọng đọc tự nhiên, đều âm lượng và không có cảm giác ghép máy:

1. **Tab 1 (Kokoro-82M ONNX):** Trộn 2 giọng bản xứ bất kỳ (vd: `em_alex` + `am_adam`) theo tỷ lệ tùy chỉnh để tạo ra **giọng mẫu bản xứ độc quyền (10–15s)** có đầy đủ ngữ điệu câu hỏi & nhấn nhá, lưu vào `voices/`. Hỗ trợ xuất nhanh kịch bản dài + phụ đề `.srt`.
2. **Tab 2 (Chatterbox Multilingual):** Dùng chính giọng mẫu ở Tab 1 để đọc kịch bản dài với cảm xúc giống người thật cho **23 ngôn ngữ** (Tây Ban Nha, Anh, Đức, Pháp, Ý, Bồ Đào Nha, Hà Lan, Ba Lan, Thụy Điển, Na Uy, Đan Mạch, Phần Lan, Hy Lạp, Thổ Nhĩ Kỳ…), gồm:
   - **4 Preset chọn nhanh:** *Kể chuyện YouTube tự nhiên*, *Tài liệu trầm ấm*, *Kịch tính / Review*, *Clone xuyên ngôn ngữ (không dính giọng gốc)*.
   - **Tự động dọn kịch bản:** Bỏ markdown (`**`, `#`, `-`, emoji, URL) khi copy từ ChatGPT/Gemini.
   - **Mastering phòng thu YouTube:** Tự cắt khoảng lặng dư, cân bằng âm lượng từng câu, fade chống tiếng "tách", nhịp nghỉ biến thiên theo dấu câu + room-tone siêu nhẹ, chuẩn hóa âm lượng `-16 dBFS` + limiter chống rè.
   - **Tự phát hiện đoạn cà lăm/lỗi để đọc lại** + xuất kèm phụ đề `.srt` khớp thời gian.

## Cách cài đặt trên PC có RTX 2060 (sau khi `git pull`)
1. Mở thư mục `d:\Du_an\tts-studio` trên PC.
2. Nhấn đúp **[install_chatterbox_gpu.bat](install_chatterbox_gpu.bat)** để tự động cài Python 3.11 (`.venv`), Kokoro-82M, PyTorch CUDA 12.4 và Chatterbox Multilingual.
3. Sau khi cài xong, nhấn đúp **[start.bat](start.bat)** → mở trình duyệt tại `http://127.0.0.1:7860`.

## Thư mục lưu trữ (tự động ignore khỏi Git)
- `voices/` — Các file giọng mẫu độc quyền đã tạo.
- `outputs/` — File audio `.wav` và phụ đề `.srt` xuất ra cho video YouTube.
- `models/` — Lưu model Kokoro (~337 MB) và cache HuggingFace Chatterbox (~2 GB) đặt gọn trên ổ `D:`.
