from __future__ import annotations

import hashlib
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError


# ============================================================
# COMMIT 001
# feat: standalone image analysis application
#
# Программа запускается одним файлом main.py.
#
# Никаких аргументов командной строки.
# Никаких ручных подтверждений.
# Никакого обязательного просмотра JSON.
#
# BASE_PATH берётся из .env, расположенного на уровень выше.
# ============================================================


# ============================================================
# CONSTANTS
# ============================================================

ANALYSIS_DIR = "_IMG_ANALYSIS"

ORIGINAL = "original"
DUPLICATE = "duplicate"
LOW_QUALITY = "low_quality"
ERROR = "error"


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
    ".avif",
}


# ============================================================
# CONFIG
# ============================================================

class Config:
    """
    Конфигурация приложения.

    Ожидаемая структура:

        .env
        IMG_analysis/
            main.py

    В .env:

        BASE_PATH=F:\\Images

    Атрибуты:
        base_path:
            Корневая директория с изображениями.

        report_path:
            Путь к итоговому report.txt.

        phash_threshold:
            Допустимое расстояние между pHash.

        dhash_threshold:
            Допустимое расстояние между dHash.
    """

    def __init__(
        self,
        base_path: Path,
    ):
        self.base_path = base_path

        self.project_dir = (
            Path(__file__).resolve().parent
        )

        self.report_path = (
            self.project_dir / "report.txt"
        )

        # Чем меньше значение,
        # тем строже поиск похожих изображений.
        self.phash_threshold = 8
        self.dhash_threshold = 8

    @classmethod
    def load(cls) -> "Config":
        """
        Загружает BASE_PATH из .env.

        Returns:
            Config: объект конфигурации.

        Raises:
            FileNotFoundError:
                Если .env не найден или BASE_PATH
                не существует.

            ValueError:
                Если BASE_PATH отсутствует в .env.

            NotADirectoryError:
                Если BASE_PATH не является директорией.
        """

        main_file = Path(
            __file__
        ).resolve()

        # .env находится на уровень выше main.py
        env_file = (
            main_file.parent.parent / ".env"
        )

        if not env_file.exists():
            raise FileNotFoundError(
                f".env не найден:\n{env_file}"
            )

        load_dotenv(
            dotenv_path=env_file
        )

        base_path_raw = os.getenv(
            "BASE_PATH"
        )

        if not base_path_raw:
            raise ValueError(
                "В .env отсутствует BASE_PATH"
            )

        base_path = (
            Path(base_path_raw)
            .expanduser()
            .resolve()
        )

        if not base_path.exists():
            raise FileNotFoundError(
                f"BASE_PATH не существует:\n"
                f"{base_path}"
            )

        if not base_path.is_dir():
            raise NotADirectoryError(
                f"BASE_PATH не является директорией:\n"
                f"{base_path}"
            )

        return cls(
            base_path=base_path
        )


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class ImageInfo:
    """
    Информация об одном изображении.

    Здесь хранятся все характеристики, необходимые
    для дальнейшего анализа.

    Атрибуты:

        path:
            Путь к исходному файлу.

        width / height:
            Разрешение изображения.

        file_size:
            Размер файла в байтах.

        sha256:
            Хэш содержимого файла.

        phash:
            Перцептивный хэш изображения.

        dhash:
            Разностный хэш изображения.

        format:
            Формат изображения.

        quality_score:
            Приблизительная оценка качества.

        category:
            Итоговая категория.

        original_name:
            Имя изображения, которое алгоритм считает
            оригиналом для текущей копии.
    """

    path: Path

    width: int = 0
    height: int = 0

    file_size: int = 0

    sha256: str = ""

    phash: str = ""
    dhash: str = ""

    format: str = ""

    quality_score: float = 0.0

    category: str = ""

    original_name: str = ""


# ============================================================
# IMAGE SCANNER
# ============================================================

class ImageScanner:
    """
    Рекурсивно ищет изображения в BASE_PATH.

    Директории _IMG_ANALYSIS игнорируются, чтобы при
    повторном запуске программа не анализировала уже
    обработанные изображения.
    """

    def __init__(
        self,
        base_path: Path,
    ):
        self.base_path = base_path

    def scan(self) -> list[Path]:
        """
        Рекурсивно ищет все поддерживаемые изображения.

        Returns:
            list[Path]:
                Список путей найденных изображений.
        """

        result = []

        for path in self.base_path.rglob("*"):

            if not path.is_file():
                continue

            # Не анализируем уже созданные каталоги
            # с результатами.
            if ANALYSIS_DIR in path.parts:
                continue

            if (
                path.suffix.lower()
                not in SUPPORTED_EXTENSIONS
            ):
                continue

            result.append(path)

        return result


# ============================================================
# IMAGE ANALYZER
# ============================================================

class ImageAnalyzer:
    """
    Анализирует отдельные изображения.

    Для каждого файла вычисляются:

        - разрешение;
        - размер;
        - SHA-256;
        - pHash;
        - dHash;
        - формат;
        - quality score.

    SHA-256 используется для поиска абсолютно
    одинаковых файлов.

    pHash и dHash используются для поиска визуально
    похожих изображений.
    """

    def analyze(
        self,
        path: Path,
    ) -> ImageInfo:
        """
        Анализирует один файл изображения.

        Args:
            path:
                Путь к изображению.

        Returns:
            ImageInfo:
                Результат анализа.
        """

        info = ImageInfo(
            path=path,
            file_size=path.stat().st_size,
        )

        try:

            with Image.open(path) as image:

                info.width = image.width
                info.height = image.height

                info.format = (
                    image.format or ""
                )

                info.phash = str(
                    imagehash.phash(image)
                )

                info.dhash = str(
                    imagehash.dhash(image)
                )

                info.quality_score = (
                    self._quality_score(info)
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ):

            info.category = ERROR

            return info

        info.sha256 = (
            self._calculate_sha256(path)
        )

        return info

    @staticmethod
    def _calculate_sha256(
        path: Path,
    ) -> str:
        """
        Вычисляет SHA-256 файла.

        Файл читается блоками по 1 МБ, поэтому
        большие изображения не загружаются целиком
        в оперативную память.

        Args:
            path:
                Путь к файлу.

        Returns:
            str:
                SHA-256 в hexadecimal формате.
        """

        sha = hashlib.sha256()

        with path.open("rb") as file:

            while True:

                chunk = file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()

    @staticmethod
    def _quality_score(
        info: ImageInfo,
    ) -> float:
        """
        Вычисляет приблизительный показатель качества.

        Сейчас используются:

            - количество пикселей;
            - размер файла.

        Это НЕ полноценная оценка визуального качества.

        Она используется только для выбора наиболее
        качественной версии среди визуально похожих файлов.

        Returns:
            float:
                Чем больше значение, тем выше
                предполагаемое качество.
        """

        pixels = (
            info.width * info.height
        )

        megapixels = (
            pixels / 1_000_000
        )

        megabytes = (
            info.file_size / 1_000_000
        )

        return (
            megapixels * 10
            + megabytes
        )


# ============================================================
# EXACT DUPLICATE DETECTOR
# ============================================================

class ExactDuplicateDetector:
    """
    Находит абсолютно одинаковые файлы.

    Если SHA-256 двух файлов совпадает, содержимое
    файлов идентично.

    Результат представляет собой список групп:

        [
            [image1, image2],
            [image3, image4, image5],
        ]
    """

    def find(
        self,
        images: list[ImageInfo],
    ) -> list[list[ImageInfo]]:
        """
        Группирует изображения по SHA-256.

        Returns:
            list[list[ImageInfo]]:
                Группы идентичных файлов.
        """

        groups = defaultdict(list)

        for image in images:

            if not image.sha256:
                continue

            groups[
                image.sha256
            ].append(image)

        return [
            group
            for group in groups.values()
            if len(group) > 1
        ]


# ============================================================
# VISUAL DUPLICATE DETECTOR
# ============================================================

class VisualDuplicateDetector:
    """
    Ищет визуально похожие изображения.

    Используются:

        pHash
        dHash

    Для ускорения используется bucketization,
    поэтому программа не сравнивает абсолютно
    каждое изображение с каждым.

    Это особенно важно при нескольких тысячах файлов.
    """

    def __init__(
        self,
        phash_threshold: int,
        dhash_threshold: int,
    ):
        self.phash_threshold = (
            phash_threshold
        )

        self.dhash_threshold = (
            dhash_threshold
        )

    def find(
        self,
        images: list[ImageInfo],
    ) -> list[set[Path]]:
        """
        Находит группы визуально похожих изображений.

        Returns:
            list[set[Path]]:
                Набор групп путей.
        """

        valid_images = [
            image
            for image in images
            if (
                image.phash
                and image.dhash
                and image.category != ERROR
            )
        ]

        if len(valid_images) < 2:
            return []

        # ----------------------------------------------------
        # BUCKETIZATION
        # ----------------------------------------------------
        #
        # Изображения предварительно распределяются
        # по части pHash.
        #
        # Это значительно уменьшает количество сравнений.
        #

        buckets = defaultdict(list)

        for image in valid_images:

            bucket = image.phash[:4]

            buckets[
                bucket
            ].append(image)

        pairs = []

        for bucket_images in buckets.values():

            if len(bucket_images) < 2:
                continue

            for index, first in enumerate(
                bucket_images
            ):

                first_phash = (
                    imagehash.hex_to_hash(
                        first.phash
                    )
                )

                first_dhash = (
                    imagehash.hex_to_hash(
                        first.dhash
                    )
                )

                for second in bucket_images[
                    index + 1:
                ]:

                    second_phash = (
                        imagehash.hex_to_hash(
                            second.phash
                        )
                    )

                    second_dhash = (
                        imagehash.hex_to_hash(
                            second.dhash
                        )
                    )

                    phash_distance = (
                        first_phash
                        - second_phash
                    )

                    dhash_distance = (
                        first_dhash
                        - second_dhash
                    )

                    if (
                        phash_distance
                        <= self.phash_threshold
                        and
                        dhash_distance
                        <= self.dhash_threshold
                    ):

                        pairs.append(
                            (
                                first.path,
                                second.path,
                            )
                        )

        return self._build_groups(
            pairs
        )

    @staticmethod
    def _build_groups(
        pairs: list[tuple[Path, Path]],
    ) -> list[set[Path]]:
        """
        Объединяет пары похожих изображений в группы.

        Например:

            A ~ B
            B ~ C

        преобразуется в:

            A, B, C
        """

        groups = []

        for first, second in pairs:

            current = {
                first,
                second,
            }

            new_groups = []

            for group in groups:

                if current & group:

                    current.update(
                        group
                    )

                else:

                    new_groups.append(
                        group
                    )

            new_groups.append(
                current
            )

            groups = new_groups

        return groups


# ============================================================
# IMAGE CLASSIFIER
# ============================================================

class ImageClassifier:
    """
    Определяет итоговую категорию изображения.

    Логика:

        1. Полностью одинаковые файлы:
           один становится original,
           остальные duplicate.

        2. Визуально похожие файлы:
           лучший по quality_score становится original,
           остальные low_quality.

        3. Изображения без совпадений:
           original.

    Для duplicate и low_quality дополнительно
    записывается имя оригинального изображения.
    """

    def classify(
        self,
        images: list[ImageInfo],
        exact_groups: list[list[ImageInfo]],
        visual_groups: list[set[Path]],
    ) -> None:

        exact_paths = set()

        # ====================================================
        # EXACT DUPLICATES
        # ====================================================

        for group in exact_groups:

            best = max(
                group,
                key=lambda x: x.quality_score,
            )

            for image in group:

                exact_paths.add(
                    image.path
                )

                if image.path == best.path:

                    image.category = (
                        ORIGINAL
                    )

                else:

                    image.category = (
                        DUPLICATE
                    )

                    image.original_name = (
                        best.path.stem
                    )

        # ====================================================
        # VISUAL DUPLICATES
        # ====================================================

        for group in visual_groups:

            candidates = [
                image
                for image in images
                if image.path in group
            ]

            if len(candidates) < 2:
                continue

            best = max(
                candidates,
                key=lambda x: x.quality_score,
            )

            for image in candidates:

                # Если файл уже определён как
                # абсолютно идентичная копия,
                # повторно классифицировать его нельзя.
                if image.path in exact_paths:
                    continue

                if image.path == best.path:

                    image.category = (
                        ORIGINAL
                    )

                else:

                    image.category = (
                        LOW_QUALITY
                    )

                    image.original_name = (
                        best.path.stem
                    )

        # ====================================================
        # UNIQUE IMAGES
        # ====================================================

        for image in images:

            if image.category:
                continue

            image.category = (
                ORIGINAL
            )


# ============================================================
# FILE ORGANIZER
# ============================================================

class FileOrganizer:
    """
    Перемещает и переименовывает изображения.

    Структура:

        _IMG_ANALYSIS/
            original/
            duplicate/
            low_quality/
            error/

    Для duplicate:

        original.jpg
        original_d.jpg

    Для low_quality:

        original.jpg
        original_lq.jpg

    Если имя уже существует:

        original_d.jpg
        original_d_001.jpg
        original_d_002.jpg
    """

    def organize(
        self,
        images: list[ImageInfo],
    ) -> int:
        """
        Перемещает все обработанные изображения
        в соответствующие каталоги.

        Returns:
            int:
                Количество успешно перемещённых файлов.
        """

        moved = 0

        for image in images:

            if not image.category:
                continue

            source = image.path

            if not source.exists():
                continue

            # ------------------------------------------------
            # Создание директории результата
            # ------------------------------------------------

            destination_dir = (
                source.parent
                / ANALYSIS_DIR
                / image.category
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Новое имя
            # ------------------------------------------------

            destination_name = (
                self._build_filename(
                    image
                )
            )

            destination = (
                destination_dir
                / destination_name
            )

            # ------------------------------------------------
            # Защита от одинаковых имён
            # ------------------------------------------------

            destination = (
                self._unique_path(
                    destination
                )
            )

            try:

                shutil.move(
                    str(source),
                    str(destination),
                )

                moved += 1

            except OSError as exc:

                print(
                    "\nОшибка перемещения:"
                )

                print(
                    f"Файл: {source}"
                )

                print(
                    f"Ошибка: {exc}"
                )

        return moved

    @staticmethod
    def _build_filename(
        image: ImageInfo,
    ) -> str:
        """
        Формирует новое имя файла.

        original:

            сохраняется исходное имя.

        duplicate:

            original_d.ext

        low_quality:

            original_lq.ext

        error:

            сохраняется исходное имя.
        """

        if image.category == ORIGINAL:

            return image.path.name

        if image.category == DUPLICATE:

            base_name = (
                image.original_name
                or image.path.stem
            )

            return (
                f"{base_name}_d"
                f"{image.path.suffix}"
            )

        if image.category == LOW_QUALITY:

            base_name = (
                image.original_name
                or image.path.stem
            )

            return (
                f"{base_name}_lq"
                f"{image.path.suffix}"
            )

        return image.path.name

    @staticmethod
    def _unique_path(
        path: Path,
    ) -> Path:
        """
        Создаёт уникальный путь, если файл с таким
        именем уже существует.

        Например:

            image_d.jpg
            image_d_001.jpg
            image_d_002.jpg
        """

        if not path.exists():
            return path

        counter = 1

        while True:

            new_path = (
                path.parent
                / (
                    f"{path.stem}_"
                    f"{counter:03d}"
                    f"{path.suffix}"
                )
            )

            if not new_path.exists():
                return new_path

            counter += 1


# ============================================================
# REPORT WRITER
# ============================================================

class ReportWriter:
    """
    Создаёт небольшой итоговый текстовый отчёт.

    В отчёте нет списка всех изображений.
    Только общая статистика и количество ошибок.

    Файл создаётся рядом с main.py.
    """

    def __init__(
        self,
        path: Path,
    ):
        self.path = path

    def write(
        self,
        images: list[ImageInfo],
        moved: int,
    ) -> None:
        """
        Записывает итоговую статистику.
        """

        counts = defaultdict(int)

        for image in images:

            counts[
                image.category or "unknown"
            ] += 1

        lines = [
            "IMAGE ANALYSIS RESULT",
            "=" * 60,
            "",
            f"Всего изображений: {len(images)}",
            f"Перемещено: {moved}",
            "",
            "КАТЕГОРИИ:",
            f"Original: {counts[ORIGINAL]}",
            f"Duplicate: {counts[DUPLICATE]}",
            f"Low quality: {counts[LOW_QUALITY]}",
            f"Error: {counts[ERROR]}",
            "",
            "Анализ завершён.",
        ]

        self.path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )


# ============================================================
# APPLICATION
# ============================================================

class ImageAnalysisApplication:
    """
    Главный класс приложения.

    Полный pipeline:

        1. Загрузка конфигурации
        2. Поиск изображений
        3. Анализ изображений
        4. Поиск точных дублей
        5. Поиск визуальных дублей
        6. Определение оригиналов
        7. Переименование
        8. Перемещение
        9. Создание отчёта
    """

    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        self.scanner = (
            ImageScanner(
                config.base_path
            )
        )

        self.analyzer = (
            ImageAnalyzer()
        )

        self.exact_detector = (
            ExactDuplicateDetector()
        )

        self.visual_detector = (
            VisualDuplicateDetector(
                config.phash_threshold,
                config.dhash_threshold,
            )
        )

        self.classifier = (
            ImageClassifier()
        )

        self.organizer = (
            FileOrganizer()
        )

        self.report_writer = (
            ReportWriter(
                config.report_path
            )
        )

    def run(self) -> None:
        """
        Запускает полный процесс анализа.
        """

        print("=" * 60)
        print("IMAGE ANALYSIS")
        print("=" * 60)

        print(
            "\nДиректория:"
        )

        print(
            self.config.base_path
        )

        # ====================================================
        # COMMIT 002
        # feat: recursive image discovery
        # ====================================================

        print(
            "\n[1/5] Поиск изображений..."
        )

        paths = (
            self.scanner.scan()
        )

        print(
            f"Найдено изображений: "
            f"{len(paths)}"
        )

        if not paths:

            print(
                "\nИзображения не найдены."
            )

            return

        # ====================================================
        # COMMIT 003
        # feat: image fingerprint analysis
        # ====================================================

        print(
            "\n[2/5] Анализ изображений..."
        )

        images = []

        total = len(paths)

        for index, path in enumerate(
            paths,
            start=1,
        ):

            images.append(
                self.analyzer.analyze(
                    path
                )
            )

            if (
                index == 1
                or index % 50 == 0
                or index == total
            ):

                print(
                    f"\rОбработано: "
                    f"{index}/{total}",
                    end="",
                    flush=True,
                )

        print()

        # ====================================================
        # COMMIT 004
        # feat: exact and visual duplicate detection
        # ====================================================

        print(
            "\n[3/5] Поиск дублей..."
        )

        exact_groups = (
            self.exact_detector.find(
                images
            )
        )

        print(
            f"Групп точных дублей: "
            f"{len(exact_groups)}"
        )

        visual_groups = (
            self.visual_detector.find(
                images
            )
        )

        print(
            f"Групп визуальных дублей: "
            f"{len(visual_groups)}"
        )

        # ====================================================
        # COMMIT 005
        # feat: classify originals and copies
        # ====================================================

        print(
            "\n[4/5] Классификация..."
        )

        self.classifier.classify(
            images,
            exact_groups,
            visual_groups,
        )

        # ====================================================
        # COMMIT 006
        # feat: rename and move analyzed images
        # ====================================================

        print(
            "\n[5/5] Перемещение и "
            "переименование..."
        )

        moved = (
            self.organizer.organize(
                images
            )
        )

        print(
            f"Перемещено файлов: {moved}"
        )

        # ====================================================
        # REPORT
        # ====================================================

        self.report_writer.write(
            images,
            moved,
        )

        self._print_result(
            images,
            moved,
        )

    @staticmethod
    def _print_result(
        images: list[ImageInfo],
        moved: int,
    ) -> None:
        """
        Выводит итоговую статистику.
        """

        counts = defaultdict(int)

        for image in images:

            counts[
                image.category
            ] += 1

        print(
            "\n" + "=" * 60
        )

        print(
            "ГОТОВО"
        )

        print(
            "=" * 60
        )

        print(
            f"Всего:       {len(images)}"
        )

        print(
            f"Original:    {counts[ORIGINAL]}"
        )

        print(
            f"Duplicate:   {counts[DUPLICATE]}"
        )

        print(
            f"Low quality: {counts[LOW_QUALITY]}"
        )

        print(
            f"Errors:      {counts[ERROR]}"
        )

        print(
            f"\nПеремещено: {moved}"
        )

        print(
            "\nРезультаты находятся в "
            "_IMG_ANALYSIS внутри исходных папок."
        )

        print(
            "\nИтоговый отчёт:"
        )

        print(
            "report.txt"
        )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """
    Единственная точка входа приложения.

    Достаточно запустить main.py.
    """

    try:

        config = Config.load()

        application = (
            ImageAnalysisApplication(
                config
            )
        )

        application.run()

    except KeyboardInterrupt:

        print(
            "\n\nПрограмма остановлена."
        )

    except Exception as exc:

        print(
            "\nКРИТИЧЕСКАЯ ОШИБКА:"
        )

        print(
            exc
        )


if __name__ == "__main__":
    main()
