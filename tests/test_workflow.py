import yaml


def _wf():
    return yaml.safe_load(open(".github/workflows/daily.yml", encoding="utf-8"))


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


def test_schedule_2155_kst_weekdays_and_manual_force():
    wf = _wf()
    on = wf.get("on", wf.get(True))
    assert on["schedule"][0]["cron"] == "55 12 * * 1-5"     # 21:55 KST 월~금 → 22시 전후 도착
    run = next(s for s in wf["jobs"]["run"]["steps"] if s.get("name") == "Run")["run"]
    assert "--force" in run and "workflow_dispatch" in run
