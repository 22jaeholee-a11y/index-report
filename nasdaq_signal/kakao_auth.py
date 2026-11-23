"""최초 1회 카카오 인증 (두 단계, 키보드 입력 없음)

1) python -m nasdaq_signal.kakao_auth                  → 로그인 주소 출력
2) python -m nasdaq_signal.kakao_auth "<이동된 주소>"     → 토큰 발급·저장, 테스트 메시지 발송
"""
from __future__ import annotations

import os
import sys
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dotenv import load_dotenv

from .config import load_config
from .kakao import AUTHORIZE_URL, TOKEN_URL, FileTokenStore, KakaoClient, merge_token_response


def extract_code(redirected: str) -> str:
    q = parse_qs(urlparse(redirected.strip()).query)
    if "code" not in q:
        raise ValueError("주소에 code 파라미터가 없습니다")
    return q["code"][0]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(".env")
    cfg = load_config()
    key = os.environ.get("KAKAO_REST_API_KEY")
    secret = os.environ.get("KAKAO_CLIENT_SECRET") or None
    if not key:
        print(".env 에 KAKAO_REST_API_KEY 를 먼저 입력하세요.")
        return 1
    redirect = cfg["kakao"]["redirect_uri"]

    if not argv:
        url = AUTHORIZE_URL + "?" + urlencode({"client_id": key, "redirect_uri": redirect,
                                               "response_type": "code", "scope": "talk_message"})
        print("1) 아래 주소를 브라우저에서 열고 카카오 로그인·동의하세요:
")
        print(url)
        print("
2) 이동된 페이지(접속 오류가 나도 괜찮습니다)의 주소창 전체를 복사해 다시 실행하세요:")
        print('   python -m nasdaq_signal.kakao_auth "http://localhost:5000/callback?code=..."')
        return 0

    code = extract_code(argv[0])
    data = {"grant_type": "authorization_code", "client_id": key, "redirect_uri": redirect, "code": code}
    if secret:
        data["client_secret"] = secret
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    body = r.json()
    if r.status_code != 200:
        print(f"토큰 발급 실패: {body}")
        return 1
    store = FileTokenStore(cfg["paths"]["tokens"])
    store.save(merge_token_response(None, body))
    KakaoClient(key, store, secret).send_text("[나스닥 방향성] 카카오톡 연결 완료", cfg["kakao"]["link_url"])
    print("연결 완료 — 카카오톡 '나와의 채팅'을 확인하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
