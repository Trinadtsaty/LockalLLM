import os

from app.config import settings



# Важно: задаём HF_HOME ДО импорта faster_whisper
if settings.hf_home:
    os.environ["HF_HOME"] = settings.hf_home


from faster_whisper import WhisperModel


class STTService:

    def __init__(self):
        print("Загрузка Whisper...")

        self.model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            cpu_threads=settings.stt_cpu_threads,
        )

        print("Whisper загружен.")

    def transcribe(self, audio_path: str):

        segments, info = self.model.transcribe(
            audio_path,
            language=settings.stt_language,
            beam_size=5,
            vad_filter=True,
        )

        segments = list(segments)

        text_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                text_parts.append(text)

        text = " ".join(text_parts)

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
                for segment in segments
            ],
        }
