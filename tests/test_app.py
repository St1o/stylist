import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from server import app as app_module
from server.config import resolve_token
from server.protocol import BROW_SCAN, STEADY_FRAMES
from tests.helpers import jpeg, texture

TOKEN = "secret-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class AppTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.data = Path(self._tmp.name)
        self.client = TestClient(app_module.create_app(TOKEN, self.data, web_dir=None))
        self.addCleanup(self.client.close)
        self.good = jpeg(texture())

    def new_scan(self) -> str:
        return self.client.post("/api/scan", headers=AUTH).json()["scan_id"]

    def test_health_is_open_but_scan_needs_token(self):
        self.assertEqual(self.client.get("/api/health").json(), {"ok": True})
        self.assertEqual(self.client.post("/api/scan").status_code, 401)
        self.assertEqual(self.client.post("/api/scan", headers={"Authorization": "Bearer wrong"}).status_code, 401)

    def test_create_scan_lists_steps_and_writes_meta(self):
        created = self.client.post("/api/scan", headers=AUTH).json()
        self.assertEqual([s["key"] for s in created["steps"]], [s.key for s in BROW_SCAN])
        meta = json.loads((self.data / "scans" / created["scan_id"] / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["steps"], {})

    def test_socket_rejects_wrong_token_and_unknown_scan(self):
        scan_id = self.new_scan()
        for path in (f"/ws/scan/{scan_id}?token=wrong", f"/ws/scan/nope?token={TOKEN}"):
            with self.assertRaises(WebSocketDisconnect):
                with self.client.websocket_connect(path):
                    pass

    def test_stream_then_upload_advances_the_step(self):
        scan_id = self.new_scan()
        with self.client.websocket_connect(f"/ws/scan/{scan_id}?token={TOKEN}") as socket:
            self.assertEqual(socket.receive_json()["step"]["key"], "front")
            replies = []
            for _ in range(STEADY_FRAMES + 1):
                socket.send_bytes(self.good)
                replies.append(socket.receive_json())
            self.assertTrue(replies[-1]["capture"])

            done = self.client.put(f"/api/scan/{scan_id}/front", content=self.good, headers=AUTH)
            self.assertEqual(done.json(), {"ok": True, "next": "yaw_left", "done": False})

            socket.send_bytes(self.good)
            self.assertEqual(socket.receive_json()["step"]["key"], "yaw_left")

        folder = self.data / "scans" / scan_id
        self.assertEqual((folder / "front.jpg").read_bytes(), self.good)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        saved = meta["steps"]["front"]["files"]["front.jpg"]
        self.assertEqual((saved["width"], saved["height"]), (640, 480))
        self.assertFalse(saved["pose_checked"])

    def test_commands_skip_and_force(self):
        scan_id = self.new_scan()
        with self.client.websocket_connect(f"/ws/scan/{scan_id}?token={TOKEN}") as socket:
            socket.receive_json()
            socket.send_text(json.dumps({"cmd": "skip"}))
            self.assertEqual(socket.receive_json()["step"]["key"], "yaw_left")
            socket.send_text("не json")
            self.assertEqual(socket.receive_json()["step"]["key"], "yaw_left")
            socket.send_text(json.dumps({"cmd": "force"}))
            socket.receive_json()
            socket.send_bytes(self.good)
            self.assertTrue(socket.receive_json()["capture"])
        meta = json.loads((self.data / "scans" / scan_id / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["skipped"], ["front"])

    def test_upload_checks(self):
        scan_id = self.new_scan()
        url = f"/api/scan/{scan_id}"
        self.assertEqual(self.client.put("/api/scan/nope/front", content=self.good, headers=AUTH).status_code, 404)
        self.assertEqual(self.client.put(f"{url}/unknown", content=self.good, headers=AUTH).status_code, 404)
        self.assertEqual(self.client.put(f"{url}/yaw_left", content=self.good, headers=AUTH).status_code, 409)
        self.assertEqual(self.client.put(f"{url}/front", content=b"garbage", headers=AUTH).status_code, 400)
        self.assertEqual(self.client.put(f"{url}/front?frame=5", content=self.good, headers=AUTH).status_code, 400)
        self.assertEqual(self.client.put(f"{url}/front", content=self.good).status_code, 401)

    def test_upload_size_limit(self):
        scan_id = self.new_scan()
        original = app_module.MAX_UPLOAD
        app_module.MAX_UPLOAD = 100
        self.addCleanup(setattr, app_module, "MAX_UPLOAD", original)
        response = self.client.put(f"/api/scan/{scan_id}/front", content=self.good, headers=AUTH)
        self.assertEqual(response.status_code, 413)

    def test_series_frames_get_numbered_files(self):
        scan_id = self.new_scan()
        for key in ("front", "yaw_left", "yaw_right", "brow_left_close", "brow_right_close"):
            response = self.client.put(f"/api/scan/{scan_id}/{key}", content=self.good, headers=AUTH)
            self.assertEqual(response.status_code, 200, key)
        for frame in range(12):
            response = self.client.put(f"/api/scan/{scan_id}/brow_left_move?frame={frame}", content=self.good, headers=AUTH)
            self.assertEqual(response.status_code, 200)
        self.assertTrue((self.data / "scans" / scan_id / "brow_left_move_11.jpg").exists())
        self.assertEqual(response.json()["next"], "brow_right_move")


class TokenTest(unittest.TestCase):
    def test_token_is_created_once_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, clear=False):
            os.environ.pop("STYLIST_TOKEN", None)
            data = Path(tmp) / "data"
            first = resolve_token(data)
            self.assertGreater(len(first), 20)
            self.assertEqual(resolve_token(data), first)


if __name__ == "__main__":
    unittest.main()
