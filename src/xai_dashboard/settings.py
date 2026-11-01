"""Application settings loaded from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Multi-Disease XAI Research Dashboard"
    environment: str = "development"
    secret_key: SecretStr = Field(default=SecretStr("development-only-change-before-deploy"))
    database_url: str = "sqlite:///./xai_dashboard.db"
    access_token_expire_minutes: int = 60
    login_max_attempts: int = 5
    login_window_seconds: int = 300
    artifacts_dir: Path = Path("artifacts")
    configs_dir: Path = Path("configs")
    api_base_url: str = "http://127.0.0.1:8000/api/v1"

    def validate_deployment_secret(self) -> None:
        if (
            self.environment.strip().lower() in {"prod", "production"}
            and (
                self.secret_key.get_secret_value() == "development-only-change-before-deploy"
                or len(self.secret_key.get_secret_value()) < 32
            )
        ):
            raise ValueError("Set a unique SECRET_KEY of at least 32 characters in production")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_deployment_secret()
    return settings
