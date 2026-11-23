import json
from datetime import datetime, timedelta, timezone

import pytest

from nasdaq_signal import kakao
from nasdaq_signal.kakao import FileTokenStore, KakaoAuthError, KakaoClient


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers})
        return self.responses.pop(0)


def _store(tmp_path, **tokens):
    s = FileTokenStore(str(tmp_path / "tokens.json"))
    base = {"access_token": "old-a", "refresh_token": "old-r",
            "refresh_token_expires_at": "2026-12-31T00:00:00+00:00"}
    base.update(tokens)
    s.save(base)
    return s


def test_refresh_rotates_refresh_token_and_expiry(tmp_path, monkeypatch):
    rec = Recorder(FakeResp({"access_token": "new-a", "refresh_token": "new-r", "refresh_token_expires_in": 86400 * 60}))
    monkeypatch.setattr(kakao.requests, "post", rec)
    store = _store(tmp_path)
    KakaoClient("key", store).refresh()
    t = store.load()
    assert t["access_token"] == "new-a" and t["refresh_token"] == "new-r"
    exp = datetime.fromisoformat(t["refresh_token_expires_at"])
    assert timedelta(days=59) < exp - datetime.now(timezone.utc) <= timedelta(days=60)
    assert rec.calls[0]["data"]["grant_type"] == "refresh_token"
    assert rec.calls[0]["data"]["refresh_token"] == "old-r"


def test_refresh_keeps_old_refresh_token_when_not_returned(tmp_path, monkeypatch):
    monkeypatch.setattr(kakao.requests, "post", Recorder(FakeResp({"access_token": "new-a"})))
    store = _store(tmp_path)
    KakaoClient("key", store).refresh()
    t = store.load()
    assert t["access_token"] == "new-a" and t["refresh_token"] == "old-r"
    assert t["refresh_token_expires_at"] == "2026-12-31T00:00:00+00:00"


def test_refresh_sends_client_secret_when_given(tmp_path, monkeypatch):
    rec = Recorder(FakeResp({"access_token": "a"}))
    monkeypatch.setattr(kakao.requests, "post", rec)
    KakaoClient("key", _store(tmp_path), client_secret="sec").refresh()
    assert rec.calls[0]["data"]["client_secret"] == "sec"


def test_refresh_invalid_grant_raises_auth_error(tmp_path, monkeypatch):
    monkeypatch.setattr(kakao.requests, "post", Recorder(FakeResp({"error": "invalid_grant"}, status=400)))
    with pytest.raises(KakaoAuthError, match="kakao_auth"):
        KakaoClient("key", _store(tmp_path)).refresh()


def test_refresh_without_tokens_raises_auth_error(tmp_path):
    with pytest.raises(KakaoAuthError, match="kakao_auth"):
        KakaoClient("key", FileTokenStore(str(tmp_path / "none.json"))).refresh()


def test_send_text_rejects_over_200_chars(tmp_path):
    with pytest.raises(ValueError):
        KakaoClient("key", _store(tmp_path)).send_text("가" * 201, "https://x")


def test_send_text_posts_template_with_link(tmp_path, monkeypatch):
    rec = Recorder(FakeResp({"result_code": 0}))
    monkeypatch.setattr(kakao.requests, "post", rec)
    KakaoClient("key", _store(tmp_path)).send_text("나스닥 ▲", "https://x")
    call = rec.calls[0]
    assert call["url"].endswith("/v2/api/talk/memo/default/send")
    assert call["headers"]["Authorization"] == "Bearer old-a"
    assert "나스닥 ▲" in call["data"]["template_object"]
    obj = json.loads(call["data"]["template_object"])
    assert obj["object_type"] == "text" and obj["link"]["web_url"] == "https://x"


def test_send_text_raises_on_api_error(tmp_path, monkeypatch):
    monkeypatch.setattr(kakao.requests, "post", Recorder(FakeResp({"code": -401, "msg": "bad"}, status=401)))
    with pytest.raises(RuntimeError):
        KakaoClient("key", _store(tmp_path)).send_text("hi", "https://x")


def test_days_until_reauth(tmp_path):
    c = KakaoClient("key", _store(tmp_path))
    now = datetime(2026, 12, 21, tzinfo=timezone.utc)
    assert c.days_until_reauth(now) == 10


def test_extract_code_from_redirect_url():
    from nasdaq_signal.kakao_auth import extract_code
    assert extract_code("http://localhost:5000/callback?code=abc123 ") == "abc123"
    with pytest.raises(ValueError):
        extract_code("http://localhost:5000/callback?error=access_denied")


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX permissions only")
def test_token_file_is_owner_only(tmp_path):
    import os
    import stat
    s = FileTokenStore(str(tmp_path / "t.json"))
    s.save({"access_token": "a"})
    assert stat.S_IMODE(os.stat(tmp_path / "t.json").st_mode) == 0o600


def test_kakao_auth_without_args_prints_url_and_does_not_prompt(monkeypatch, capsys):
    import nasdaq_signal.kakao_auth as ka
    monkeypatch.setenv("KAKAO_REST_API_KEY", "abc")
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(AssertionError("prompted")))
    monkeypatch.setattr(ka, "load_dotenv", lambda *a, **k: None)
    assert ka.main([]) == 0
    out = capsys.readouterr().out
    assert "kauth.kakao.com/oauth/authorize" in out and "client_id=abc" in out


def test_kakao_auth_with_redirect_arg_exchanges_code(tmp_path, monkeypatch):
    import nasdaq_signal.kakao_auth as ka
    monkeypatch.setenv("KAKAO_REST_API_KEY", "abc")
    monkeypatch.setattr(ka, "load_dotenv", lambda *a, **k: None)
    posts = []

    def post(url, data=None, headers=None, timeout=None):
        posts.append((url, data))
        if url.endswith("/oauth/token"):
            return FakeResp({"access_token": "a", "refresh_token": "r", "refresh_token_expires_in": 100})
        return FakeResp({"result_code": 0})
    monkeypatch.setattr(ka.requests, "post", post)
    monkeypatch.setattr(kakao.requests, "post", post)
    cfg = ka.load_config(apply_tuning=False)
    cfg["paths"]["tokens"] = str(tmp_path / "t.json")
    monkeypatch.setattr(ka, "load_config", lambda *a, **k: cfg)
    assert ka.main(["http://localhost:5000/callback?code=XYZ"]) == 0
    assert posts[0][1]["code"] == "XYZ"
    assert FileTokenStore(cfg["paths"]["tokens"]).load()["refresh_token"] == "r"
