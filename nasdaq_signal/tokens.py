"""카카오 토큰을 클라우드(GitHub Actions)용으로 암호화해 내보내기.

1) 로컬에서 카카오 인증(python -m nasdaq_signal.kakao_auth)을 마친 뒤
2) python -m nasdaq_signal.tokens export
   → .env 에 KAKAO_TOKEN_KEY 가 없으면 새로 만들어 추가하고, model/kakao_tokens.enc 를 만든다.
3) KAKAO_TOKEN_KEY 값을 GitHub 저장소 Secrets 에 같은 이름으로 등록하고 model/kakao_tokens.enc 를 커밋.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .kakao import EncryptedFileTokenStore


def export_tokens(plain_path: str, out_path: str, key: str) -> None:
    tokens = json.loads(Path(plain_path).read_text(encoding="utf-8"))
    EncryptedFileTokenStore(out_path, key).save(tokens)


def _ensure_key(env_path: Path) -> str:
    from cryptography.fernet import Fernet
    from dotenv import dotenv_values

    key = (dotenv_values(env_path) if env_path.exists() else {}).get("KAKAO_TOKEN_KEY")
    if key:
        return key
    key = Fernet.generate_key().decode()
    with env_path.open("a", encoding="utf-8") as f:
        f.write(f"\nKAKAO_TOKEN_KEY={key}\n")
    return key


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from .config import load_config

    if sys.argv[1:] != ["export"]:
        print(__doc__)
        return 1
    cfg = load_config()
    plain = cfg["paths"]["tokens"]
    if not Path(plain).exists():
        print(f"{plain} 가 없습니다. 먼저 python -m nasdaq_signal.kakao_auth 로 인증하세요.")
        return 1
    key = _ensure_key(Path(".env"))
    export_tokens(plain, cfg["paths"]["tokens_enc"], key)
    print(f"저장: {cfg['paths']['tokens_enc']} (암호화됨)")
    print("다음: .env 의 KAKAO_TOKEN_KEY 값을 GitHub 저장소 Settings → Secrets and variables → Actions 에 같은 이름으로 등록하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
