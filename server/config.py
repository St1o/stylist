"""Папка данных и токен доступа. Токен нужен, потому что на сервер идут снимки лица."""
from __future__ import annotations

import os
import secrets
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("STYLIST_DATA", "data")).resolve()


def resolve_token(data: Path) -> str:
    """STYLIST_TOKEN из окружения, иначе токен из data/token.txt (создаётся при первом запуске)."""
    from_env = os.environ.get("STYLIST_TOKEN", "").strip()
    if from_env:
        return from_env
    path = data / "token.txt"
    if path.exists():
        saved = path.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    data.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    path.write_text(token + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass  # в Windows права файла не те же, это не помеха
    return token
