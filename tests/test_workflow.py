import yaml


def _wf():
    return yaml.safe_load(open(".github/workflows/daily.yml", encoding="utf-8"))


def test_schedule_is_0907_kst_and_manual_run_allowed():
    wf = _wf()
    on = wf.get("on", wf.get(True))            # PyYAML은 'on' 키를 True로 읽음
    assert on["schedule"][0]["cron"] == "7 0 * * *"   # 00:07 UTC = 09:07 KST
    assert "workflow_dispatch" in on


def test_secrets_and_commit_step():
    wf = _wf()
    assert wf["permissions"]["contents"] == "write"
    steps = wf["jobs"]["run"]["steps"]
    run_env = next(s for s in steps if s.get("name") == "Run")["env"]
    for k in ("FRED_API_KEY", "KAKAO_REST_API_KEY", "KAKAO_CLIENT_SECRET", "KAKAO_TOKEN_KEY"):
        assert run_env[k] == "${{ secrets.%s }}" % k
    commit = next(s for s in steps if s.get("name") == "Commit results")
    assert "public/index.html" in commit["run"] and "model/" in commit["run"]
    assert commit["if"] == "always()"
