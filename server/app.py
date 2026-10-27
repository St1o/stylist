"""Сайт и API: создание сеанса, поток кадров по WebSocket, загрузка полных кадров."""
from __future__ import annotations

import hmac
import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import quality
from .protocol import BROW_SCAN, ScanSession

MAX_UPLOAD = 25 * 1024 * 1024  # байт на один полный кадр
MAX_SESSIONS = 20  # сеансов в памяти; старые вытесняются (файлы остаются на диске)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _same(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def _write_atomic(path: Path, payload: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(payload)
    os.replace(part, path)


def create_app(
    token: str,
    data: Path,
    web_dir: Path | None = WEB_DIR,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    app = FastAPI(title="Стилист", docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, ScanSession] = {}
    metas: dict[str, dict] = {}

    def require_token(request: Request) -> None:
        header = request.headers.get("authorization", "")
        given = header[7:].strip() if header.lower().startswith("bearer ") else request.query_params.get("token", "")
        if not _same(given, token):
            raise HTTPException(status_code=401, detail="нужен токен")

    def save_meta(scan_id: str) -> None:
        meta = metas[scan_id]
        meta["skipped"] = sessions[scan_id].skipped
        folder = data / "scans" / scan_id
        _write_atomic(folder / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    @app.post("/api/scan", dependencies=[Depends(require_token)])
    async def create_scan() -> dict:
        scan_id = secrets.token_hex(8)
        while len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            del sessions[oldest]
            del metas[oldest]
        sessions[scan_id] = ScanSession(scan_id, clock=clock)
        metas[scan_id] = {
            "scan_id": scan_id,
            "created": _now(),
            "protocol": [step.key for step in BROW_SCAN],
            "steps": {},
            "skipped": [],
        }
        (data / "scans" / scan_id).mkdir(parents=True, exist_ok=True)
        save_meta(scan_id)
        return {
            "scan_id": scan_id,
            "steps": [{"key": s.key, "title": s.title, "instruction": s.instruction} for s in BROW_SCAN],
        }

    @app.put("/api/scan/{scan_id}/{step_key}", dependencies=[Depends(require_token)])
    async def upload_frame(scan_id: str, step_key: str, request: Request, frame: int = 0) -> dict:
        session = sessions.get(scan_id)
        if session is None:
            raise HTTPException(status_code=404, detail="сеанс не найден")
        if step_key not in {step.key for step in session.steps}:
            raise HTTPException(status_code=404, detail="такого шага нет")
        step = session.current
        if step is None or step.key != step_key:
            raise HTTPException(status_code=409, detail="сейчас идёт другой шаг")
        if not 0 <= frame < step.frames:
            raise HTTPException(status_code=400, detail="номер кадра вне серии")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="кадр слишком большой")
        body = await request.body()
        if len(body) > MAX_UPLOAD:
            raise HTTPException(status_code=413, detail="кадр слишком большой")
        decoded = await run_in_threadpool(quality.decode_full, body)
        if decoded is None:
            raise HTTPException(status_code=400, detail="файл не похож на JPEG")
        width, height, small = decoded

        name = f"{step_key}.jpg" if step.frames == 1 else f"{step_key}_{frame:02d}.jpg"
        _write_atomic(data / "scans" / scan_id / name, body)
        entry = metas[scan_id]["steps"].setdefault(step_key, {"frames": step.frames, "files": {}})
        entry["files"][name] = {
            "width": width,
            "height": height,
            "bytes": len(body),
            "quality": quality.measure(small).as_dict(),
            "saved": _now(),
            "pose_checked": False,  # поза головы пока не проверяется
        }
        session.accept_capture(step_key, frame)
        save_meta(scan_id)
        following = session.current
        return {"ok": True, "next": None if following is None else following.key, "done": session.done}

    @app.websocket("/ws/scan/{scan_id}")
    async def scan_socket(websocket: WebSocket, scan_id: str) -> None:
        session = sessions.get(scan_id)
        if session is None or not _same(websocket.query_params.get("token", ""), token):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_json({**session.snapshot(), "type": "hello"})
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    small = await run_in_threadpool(quality.prepare, message["bytes"])
                    if small is None:
                        await websocket.send_json({"type": "error", "message": "кадр не читается"})
                    else:
                        await websocket.send_json(session.feed(small))
                elif message.get("text"):
                    command = _command(message["text"])
                    if command == "skip":
                        session.skip()
                        save_meta(scan_id)
                    elif command == "force":
                        session.force_capture()
                    await websocket.send_json(session.snapshot())
        except WebSocketDisconnect:
            pass

    if web_dir is not None and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app


def _command(text: str) -> str:
    try:
        payload = json.loads(text)
    except ValueError:
        return ""
    command = payload.get("cmd") if isinstance(payload, dict) else None
    return command if command in ("skip", "force") else ""
