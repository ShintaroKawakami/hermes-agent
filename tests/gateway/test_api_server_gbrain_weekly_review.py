import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server
from gateway.platforms.api_server import APIServerAdapter


SECRET = "test-gbrain-weekly-review-secret"


def _signed_query(action: str, candidate_id: str = "gbrain-20260701-001", expires: int | None = None) -> str:
    expires_text = str(expires if expires is not None else int(time.time()) + 300)
    signing_input = "\n".join([action, candidate_id, expires_text]).encode("utf-8")
    sig = hmac.new(SECRET.encode("utf-8"), signing_input, hashlib.sha256).hexdigest()
    return f"action={action}&candidate_id={candidate_id}&expires={expires_text}&sig={sig}"


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/gbrain/weekly-review/action", adapter._handle_gbrain_weekly_review_action)
    app.router.add_post("/gbrain/weekly-review/action", adapter._handle_gbrain_weekly_review_action)
    return app


@pytest.fixture
def fake_review_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    calls = tmp_path / "calls.jsonl"
    script = tmp_path / "gbrain-weekly-review.sh"
    script.write_text(
        """#!/bin/sh
python3 - "$@" <<'PY'
import json, os, sys
calls = os.environ["GBRAIN_TEST_CALLS"]
with open(calls, "a", encoding="utf-8") as f:
    f.write(json.dumps(sys.argv[1:], ensure_ascii=False) + "\\n")
if os.environ.get("GBRAIN_TEST_SCRIPT_OK_FALSE") == "1":
    print(json.dumps({"ok": False, "text": "script blocked"}, ensure_ascii=False))
else:
    print(json.dumps({"ok": True, "text": "script ok"}, ensure_ascii=False))
PY
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("GBRAIN_TEST_CALLS", str(calls))
    monkeypatch.setenv("GBRAIN_WEEKLY_REVIEW_APPROVAL_SECRET", SECRET)
    monkeypatch.setattr(api_server, "GBRAIN_WEEKLY_REVIEW_SCRIPT", script)
    return calls


@pytest.mark.asyncio
async def test_gbrain_weekly_review_approve_runs_script(fake_review_script: Path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(f"/gbrain/weekly-review/action?{_signed_query('approve')}")
        text = await resp.text()

    assert resp.status == 200
    assert "script ok" in text
    calls = [json.loads(line) for line in fake_review_script.read_text(encoding="utf-8").splitlines()]
    assert calls == [["--action", "approve", "--candidate-id", "gbrain-20260701-001"]]


@pytest.mark.asyncio
async def test_gbrain_weekly_review_rejects_tampered_signature(fake_review_script: Path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(f"/gbrain/weekly-review/action?{_signed_query('approve')[:-1]}0")

    assert resp.status == 403
    assert not fake_review_script.exists() or fake_review_script.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_gbrain_weekly_review_rejects_expired_url(fake_review_script: Path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(f"/gbrain/weekly-review/action?{_signed_query('approve', expires=int(time.time()) - 1)}")

    assert resp.status == 403
    assert not fake_review_script.exists() or fake_review_script.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_gbrain_weekly_review_revise_form_posts_revision(fake_review_script: Path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    query = _signed_query("revise")
    async with TestClient(TestServer(_app(adapter))) as cli:
        form_resp = await cli.get(f"/gbrain/weekly-review/action?{query}")
        assert form_resp.status == 200
        assert "修正文" in await form_resp.text()

        submit_resp = await cli.post(
            "/gbrain/weekly-review/action",
            data={
                "action": "revise",
                "candidate_id": "gbrain-20260701-001",
                "expires": query.split("expires=", 1)[1].split("&", 1)[0],
                "sig": query.split("sig=", 1)[1],
                "revision": "修正文です",
            },
        )
        text = await submit_resp.text()

    assert submit_resp.status == 200
    assert "script ok" in text
    calls = [json.loads(line) for line in fake_review_script.read_text(encoding="utf-8").splitlines()]
    assert calls == [
        [
            "--action",
            "revise_submit",
            "--candidate-id",
            "gbrain-20260701-001",
            "--revision",
            "修正文です",
        ]
    ]


@pytest.mark.asyncio
async def test_gbrain_weekly_review_script_ok_false_returns_500(
    fake_review_script: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GBRAIN_TEST_SCRIPT_OK_FALSE", "1")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.get(f"/gbrain/weekly-review/action?{_signed_query('approve')}")
        text = await resp.text()

    assert resp.status == 500
    assert "script blocked" in text
