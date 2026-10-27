"""Запуск: python -m server [--host 127.0.0.1] [--port 8000] [--data папка]"""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import data_dir, resolve_token


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m server", description="Сервер Стилиста")
    parser.add_argument("--host", default="127.0.0.1", help="адрес; 0.0.0.0 открывает сервер для домашней сети")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", type=Path, default=None, help="папка данных (иначе STYLIST_DATA или ./data)")
    args = parser.parse_args(argv)

    data = args.data.resolve() if args.data else data_dir()
    token = resolve_token(data)
    print(f"Папка данных: {data}")
    print(f"Открыть на компьютере: http://127.0.0.1:{args.port}/?token={token}")
    print("Камера в Safari работает только по HTTPS: для телефона нужен туннель (см. README.md, раздел «Телефон»).")
    uvicorn.run(create_app(token, data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
