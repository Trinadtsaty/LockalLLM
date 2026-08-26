import os

os.environ["HF_HOME"] = r"F:\OllamaModels\HuggingFace"

import time
from pathlib import Path

from faster_whisper import WhisperModel


# =========================
# Настройки
# =========================

MODEL_SIZE = "small"
AUDIO_FILE = Path("audio/test.m4a")

CPU_THREADS = 4


# =========================
# Проверка файла
# =========================

if not AUDIO_FILE.exists():
    print(f"Ошибка: файл не найден: {AUDIO_FILE}")
    print("Положи аудиофайл в папку audio/ и назови его test.m4a")
    exit(1)


# =========================
# Загрузка модели
# =========================

print("=" * 50)
print("Локальное распознавание речи")
print("=" * 50)
print()

print(f"Модель: Whisper {MODEL_SIZE}")
print("Устройство: CPU")
print("Тип вычислений: INT8")
print(f"CPU threads: {CPU_THREADS}")
print()

print("Загрузка модели...")

start_time = time.time()

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
    cpu_threads=CPU_THREADS,
)

model_load_time = time.time() - start_time

print(f"Модель загружена за {model_load_time:.2f} сек.")
print()


# =========================
# Распознавание
# =========================

print(f"Файл: {AUDIO_FILE}")
print("Начинаю распознавание...")
print()

start_time = time.time()

segments, info = model.transcribe(
    str(AUDIO_FILE),
    language="ru",
    beam_size=5,
    vad_filter=True,
)

# faster-whisper выполняет распознавание лениво,
# поэтому превращаем генератор в список именно здесь.
segments = list(segments)

transcription_time = time.time() - start_time


# =========================
# Вывод результата
# =========================

print("=" * 50)
print("РЕЗУЛЬТАТ")
print("=" * 50)
print()

full_text = []

for segment in segments:
    text = segment.text.strip()

    if text:
        full_text.append(text)

        print(
            f"[{segment.start:6.2f} → {segment.end:6.2f}] "
            f"{text}"
        )

print()

print("=" * 50)
print("ПОЛНЫЙ ТЕКСТ")
print("=" * 50)
print()

print(" ".join(full_text))

print()

# =========================
# Информация
# =========================

print("=" * 50)
print("СТАТИСТИКА")
print("=" * 50)

print(f"Определённый язык: {info.language}")
print(f"Вероятность языка: {info.language_probability:.2f}")
print(f"Время загрузки модели: {model_load_time:.2f} сек.")
print(f"Время распознавания: {transcription_time:.2f} сек.")
print(f"Количество сегментов: {len(segments)}")
