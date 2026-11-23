"""카카오톡 '나에게 보내기': 토큰 저장·갱신과 텍스트 발송."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import requests

TOKEN_URL = "https://kauth.kakao.com/oauth/token"
AUTHORIZE_URL = "https://kauth.kakao.com/oauth/authorize"
MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
MAX_TEXT = 200
REAUTH_HINT = "카카오 재인증 필요: python -m nasdaq_signal.kakao_auth"


class KakaoAuthError(RuntimeError):
    pass


class TokenStore(Protocol):
    def load(self) -> dict | None: ...
    def save(self, tokens: dict) -> None: ...


class FileTokenStore:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, tokens: dict) -> None:
        # 소유자 전용 권한(0600) — 리눅스/클라우드 실행 시 토큰 노출 방지. Windows에선 무시됨.
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(tokens, ensure_ascii=False, indent=1))
        os.chmod(self.path, 0o600)


class EncryptedFileTokenStore:
    """암호화된 토큰 파일 (저장소에 커밋해도 안전). 키는 환경변수 KAKAO_TOKEN_KEY(Fernet 키).

    리프레시 토큰·만료시각이 바뀔 때만 파일을 다시 쓴다 → 매일 바뀌는 액세스 토큰 때문에 커밋이 생기지 않음.
    액세스 토큰은 실행 중 메모리에만 둔다 (매 실행마다 리프레시로 새로 받음).
    """
    PERSIST = ("refresh_token", "refresh_token_expires_at")

    def __init__(self, path: str, key: str):
        from cryptography.fernet import Fernet
        self.path = Path(path)
        self._f = Fernet(key.encode() if isinstance(key, str) else key)
        self._mem: dict | None = None

    def load(self) -> dict | None:
        if self._mem is not None:
            return dict(self._mem)
        if not self.path.exists():
            return None
        from cryptography.fernet import InvalidToken
        try:
            data = json.loads(self._f.decrypt(self.path.read_bytes()).decode("utf-8"))
        except InvalidToken as e:
            raise KakaoAuthError("토큰 파일을 풀 수 없습니다: KAKAO_TOKEN_KEY 가 맞는지 확인하세요") from e
        self._mem = data
        return dict(data)

    def save(self, tokens: dict) -> None:
        old = self.load() or {}
        self._mem = dict(tokens)
        if all(old.get(k) == tokens.get(k) for k in self.PERSIST) and self.path.exists():
            return
        persist = {k: tokens.get(k) for k in self.PERSIST if tokens.get(k) is not None}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self._f.encrypt(json.dumps(persist).encode("utf-8")))


def merge_token_response(old: dict | None, resp: dict, now: datetime | None = None) -> dict:
    """토큰 응답을 저장 형식으로 병합. 새 리프레시 토큰이 오면 교체하고 만료시각 갱신."""
    now = now or datetime.now(timezone.utc)
    t = dict(old or {})
    t["access_token"] = resp["access_token"]
    if resp.get("refresh_token"):
        t["refresh_token"] = resp["refresh_token"]
        if resp.get("refresh_token_expires_in"):
            t["refresh_token_expires_at"] = (now + timedelta(seconds=int(resp["refresh_token_expires_in"]))).isoformat()
    return t


class KakaoClient:
    def __init__(self, rest_api_key: str, store: TokenStore, client_secret: str | None = None):
        self.key = rest_api_key
        self.store = store
        self.secret = client_secret

    def refresh(self) -> None:
        tokens = self.store.load()
        if not tokens or not tokens.get("refresh_token"):
            raise KakaoAuthError(REAUTH_HINT)
        data = {"grant_type": "refresh_token", "client_id": self.key, "refresh_token": tokens["refresh_token"]}
        if self.secret:
            data["client_secret"] = self.secret
        r = requests.post(TOKEN_URL, data=data, timeout=30)
        body = r.json()
        if r.status_code != 200 or "access_token" not in body:
            raise KakaoAuthError(f"{REAUTH_HINT} ({body.get('error', r.status_code)})")
        self.store.save(merge_token_response(tokens, body))

    def days_until_reauth(self, now: datetime | None = None) -> int | None:
        tokens = self.store.load() or {}
        exp = tokens.get("refresh_token_expires_at")
        if not exp:
            return None
        now = now or datetime.now(timezone.utc)
        return (datetime.fromisoformat(exp) - now).days

    def send_text(self, text: str, link_url: str) -> None:
        if len(text) > MAX_TEXT:
            raise ValueError(f"message is {len(text)} chars, max {MAX_TEXT}")
        tokens = self.store.load() or {}
        template = {"object_type": "text", "text": text,
                    "link": {"web_url": link_url, "mobile_web_url": link_url}}
        r = requests.post(MEMO_URL, data={"template_object": json.dumps(template, ensure_ascii=False)},
                          headers={"Authorization": f"Bearer {tokens.get('access_token', '')}"}, timeout=30)
        body = r.json()
        if r.status_code != 200 or body.get("result_code") != 0:
            raise RuntimeError(f"kakao send failed: {r.status_code} {body}")
