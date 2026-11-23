import json

from cryptography.fernet import Fernet

from nasdaq_signal.kakao import EncryptedFileTokenStore
from nasdaq_signal.tokens import export_tokens


def test_export_encrypts_local_tokens(tmp_path):
    plain = tmp_path / "kakao_tokens.json"
    plain.write_text(json.dumps({"access_token": "a", "refresh_token": "r",
                                 "refresh_token_expires_at": "2027-01-01T00:00:00+00:00"}), encoding="utf-8")
    key = Fernet.generate_key().decode()
    out = tmp_path / "model" / "kakao_tokens.enc"
    export_tokens(str(plain), str(out), key)
    t = EncryptedFileTokenStore(str(out), key).load()
    assert t["refresh_token"] == "r" and t["refresh_token_expires_at"].startswith("2027")
