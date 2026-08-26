from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    stt_model: str = "small"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_cpu_threads: int = 4
    stt_language: str = "ru"

    stt_host: str = "127.0.0.1"
    stt_port: int = 8001

    hf_home: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
