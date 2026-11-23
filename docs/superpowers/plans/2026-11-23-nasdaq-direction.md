# 나스닥 방향성 일일 알림 (1단계: 규칙 기반) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 매일 09:00 KST에 나스닥 단기·중기·장기 방향성 점수(-100~+100)와 확률을 카카오톡 '나에게 보내기'로 발송하는 Python 도구.

**Architecture:** 수집(`data/`) → 원지표 계산(`indicators.py`) → 기간별 신호 [-1,+1] → 카테고리·기간 가중 합산(`scoring.py`) → 메시지(`message.py`) → 카카오 발송(`kakao.py`). 점수 계산은 날짜×지표 DataFrame에 벡터 연산으로 한 번만 구현하고, 일일 실행은 마지막 행을, 백테스트(`backtest.py`)는 전체 행을 쓴다.

**Tech Stack:** Python 3.13, pandas, numpy, requests, yfinance, PyYAML, python-dotenv, xlrd(GPR .xls), pytest.

**Spec:** `docs/superpowers/specs/2026-11-23-nasdaq-direction-design.md`

## Global Constraints

- 무료 데이터 소스만 사용. FRED는 `requests`로 REST 직접 호출 (`https://api.stlouisfed.org/fred/series/observations`).
- 비밀값(`FRED_API_KEY`, `KAKAO_REST_API_KEY`, `KAKAO_CLIENT_SECRET`)은 `.env`/환경변수로만. `.env`, `state/` 는 `.gitignore`.
- 모든 경로는 프로젝트 루트 기준 상대경로 또는 config 값 — Windows/Linux(GitHub Actions) 양쪽에서 동작.
- 점수 범위 -100~+100. 화살표: ≥ +20 `▲`, ≤ -20 `▼`, 그 외 `─` (config).
- 기간·라벨: short 10거래일 ±1.5%, mid 63거래일 ±4%, long 252거래일 ±8% (config).
- `ai_adjustment` 기본 0, 상한 ±10 (2단계용 자리).
- 누락 가중치 비율 > 0.3 이면 "신뢰도 낮음".
- 확률은 보정표(검증 구간 2018-01-01~ 결과로만 생성)가 있을 때만 표시.
- Fear & Greed, 월요일 NQ 선물 등락은 **참고값** — 점수에 넣지 않음.
- 카카오 메시지는 2개, 각 200자 이하.
- 리프레시 토큰 만료 14일 전부터 재인증 경고.
- 파일 인코딩 UTF-8 (CSV는 `utf-8-sig` — 엑셀에서 한글 깨짐 방지).

## Review Focus

1. **yfinance가 일부 티커만 비거나 마지막 값이 NaN** (예: RSP 지연) → 해당 지표만 누락 처리, 나머지로 점수. 마지막 유효값이 기준일보다 5영업일 넘게 오래되면 누락으로 간주. (Task 2 테스트)
2. **FRED 키 없음/FRED 장애** → 매크로 지표 전부 누락, 실행은 계속, 메시지에 누락·신뢰도 낮음 표시. (Task 2, Task 8 테스트)
3. **카카오 리프레시 토큰 만료(`invalid_grant`)** → 명확한 `KakaoAuthError`("python -m nasdaq_signal.kakao_auth 로 재인증") 로그, 종료코드 ≠ 0. (Task 6 테스트)
4. **경고가 전부 붙은 날 메시지가 200자 초과** → 정해진 우선순위로 줄을 빼서 200자 이하 유지, 절대 잘린 문장 발송 금지. (Task 7 테스트)
5. **같은 날 두 번 실행** → `history.csv`에 같은 날짜 행이 중복되지 않고 덮어씀. (Task 8 테스트)

---

## File Structure

```
trade/
  pyproject.toml              # 의존성, pytest 설정
  config.yaml                 # 지표·가중치·기간·발송 설정
  .env.example                # 비밀값 키 목록
  .gitignore
  nasdaq_signal/
    __init__.py
    config.py                 # load_config()
    data/
      __init__.py             # collect_all(), MarketData
      prices.py               # yfinance
      fred.py                 # FRED + 발표 지연 시프트
      extras.py               # GPR, Fear&Greed, NQ 선물
    indicators.py             # compute_raw(), to_signals()
    scoring.py                # score_frame(), Calibration, arrow()
    backtest.py               # 백테스트 CLI → state/calibration.json
    kakao.py                  # TokenStore, FileTokenStore, KakaoClient
    kakao_auth.py             # 최초 인증 CLI
    message.py                # Report, format_messages()
    main.py                   # 일일 실행 CLI
  scripts/register_task.ps1   # 작업 스케줄러 09:00 등록
  tests/ ...                  # 모듈별 test_*.py
```

---

### Task 1: 프로젝트 골격 + 설정 로더

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `config.yaml`, `nasdaq_signal/__init__.py`, `nasdaq_signal/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path = "config.yaml") -> dict` — YAML 로드 후 검증. 검증 실패 시 `ConfigError(ValueError)`.

`config.yaml` 내용(값 고정):

```yaml
horizons:
  short: {days: 10,  up: 0.015, down: -0.015}
  mid:   {days: 63,  up: 0.04,  down: -0.04}
  long:  {days: 252, up: 0.08,  down: -0.08}
category_weights:          # 카테고리 × 기간, 기간별 합 = 1
  market:    {short: 0.45, mid: 0.30, long: 0.15}
  sentiment: {short: 0.30, mid: 0.20, long: 0.10}
  geo:       {short: 0.10, mid: 0.15, long: 0.15}
  macro:     {short: 0.15, mid: 0.35, long: 0.60}
indicators:                # transform: percentile | contrarian ; direction: +1/-1
  trend_50:     {category: market, transform: percentile, direction: 1}
  trend_200:    {category: market, transform: percentile, direction: 1}
  ret_20:       {category: market, transform: percentile, direction: 1}
  ret_60:       {category: market, transform: percentile, direction: 1}
  drawdown_52w: {category: market, transform: percentile, direction: 1}
  breadth:      {category: market, transform: percentile, direction: 1}
  vix:          {category: sentiment, transform: percentile, direction: -1,
                 horizons: {mid: {transform: contrarian}, long: {transform: contrarian}}}
  vix_term:     {category: sentiment, transform: percentile, direction: -1}
  gpr:          {category: geo, transform: percentile, direction: -1}
  oil_ret_20:   {category: geo, transform: percentile, direction: -1}
  gold_ret_20:  {category: geo, transform: percentile, direction: -1}
  us10y_chg_20: {category: macro, transform: percentile, direction: -1}
  curve:        {category: macro, transform: percentile, direction: 1}
  hy_spread:    {category: macro, transform: percentile, direction: -1}
  dxy_ret_60:   {category: macro, transform: percentile, direction: -1}
  nfci:         {category: macro, transform: percentile, direction: -1}
  sahm:         {category: macro, transform: percentile, direction: -1}
percentile_window: 1260    # 5년(거래일)
percentile_min_periods: 252
contrarian_band: 0.10      # 하위/상위 10% 밖에서만 신호
low_confidence_missing: 0.3
arrow_threshold: 20
ai_adjustment_cap: 10
stale_days: 5
backtest: {start: "2010-01-01", split: "2018-01-01", bucket: 20}
fred_series:               # series_id: 발표 지연(영업일)
  DGS10: 1
  T10Y2Y: 1
  BAMLH0A0HYM2: 1
  NFCI: 5
  SAHMREALTIME: 25
kakao: {redirect_uri: "http://localhost:5000/callback", link_url: "https://finance.yahoo.com/quote/%5ENDX"}
paths: {state: "state", calibration: "state/calibration.json", history: "state/history.csv", tokens: "state/kakao_tokens.json", logs: "state/logs"}
```

- [ ] **Step 1: `git init`, `.gitignore`(`.env`, `state/`, `__pycache__/`, `.venv/`), `pyproject.toml`(위 Tech Stack 의존성 + `[tool.pytest.ini_options] testpaths=["tests"]`), `.env.example` 작성. `python -m venv .venv` 후 `pip install -e .[dev]`.**
- [ ] **Step 2: 실패 테스트 작성** — `tests/test_config.py`
  - `test_loads_default_config`: `cfg = load_config()`; `cfg["horizons"]["mid"]["days"] == 63`; `cfg["indicators"]["vix"]["horizons"]["mid"]["transform"] == "contrarian"`.
  - `test_category_weights_sum_to_one_per_horizon`: 각 기간별 합이 `pytest.approx(1.0)`.
  - `test_rejects_unknown_category`: 지표 category를 `"foo"`로 바꾼 임시 YAML → `ConfigError`.
  - `test_rejects_weights_not_summing_to_one`: short 합 0.9 → `ConfigError`.
- [ ] **Step 3: 실행 → FAIL 확인** `pytest tests/test_config.py -v`
- [ ] **Step 4: `load_config` 구현** (검증: 카테고리 ∈ category_weights 키, transform ∈ {percentile, contrarian}, 기간별 가중치 합 1±1e-6).
- [ ] **Step 5: 실행 → PASS 확인**
- [ ] **Step 6: Commit** `git commit -m "feat: project scaffold and config loader"`

---

### Task 2: 가격·FRED 수집 + `collect_all`

**Files:**
- Create: `nasdaq_signal/data/__init__.py`, `nasdaq_signal/data/prices.py`, `nasdaq_signal/data/fred.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `PRICE_TICKERS = {"ndx": "^NDX", "spy": "SPY", "rsp": "RSP", "vix": "^VIX", "vix3m": "^VIX3M", "oil": "CL=F", "gold": "GC=F", "dxy": "DX-Y.NYB"}`
  - `fetch_prices(start: str) -> pd.DataFrame` — 열 = PRICE_TICKERS 키, 일별 종가, DatetimeIndex(tz 없음). 빈 티커는 열 전체 NaN.
  - `fetch_fred(series: dict[str, int], start: str, api_key: str | None) -> pd.DataFrame` — 열 = series_id 소문자. 각 시리즈를 관측일 기준으로 받은 뒤 **지연 영업일만큼 인덱스를 앞으로 시프트**(`index + BDay(lag)`). `"."` 값은 NaN. api_key가 None이면 빈 DataFrame.
  - `class DataSourceError(RuntimeError)`
  - `@dataclass MarketData: frame: pd.DataFrame; as_of: date; missing: list[str]; extras: dict[str, float | None]`
  - `collect_all(cfg: dict, start: str, include_extras: bool = True) -> MarketData` — prices와 fred(그리고 Task 3 extras)를 ndx 거래일 인덱스에 맞춰 병합, forward-fill. `as_of` = ndx 마지막 유효일. ndx가 비면 `DataSourceError`. 소스 하나가 예외를 던지면 그 소스 열 이름을 `missing`에 넣고 계속. 열별 마지막 유효값이 `as_of`보다 `stale_days` 영업일 넘게 오래되면 마지막 행을 NaN으로 만들고 `missing`에 추가.

- [ ] **Step 1: 실패 테스트 작성** (네트워크 없이 `monkeypatch`로 `yfinance.download`/`requests.get` 대체)
  - `test_fred_shifts_by_publication_lag`: NFCI 관측일 2024-01-05 값 → 결과 인덱스 2024-01-12에 위치.
  - `test_fred_dot_values_become_nan`
  - `test_fred_without_key_returns_empty`
  - `test_collect_all_raises_when_ndx_empty`: → `DataSourceError`
  - `test_collect_all_marks_failed_source_missing`: fred가 예외 → `missing`에 `dgs10` 등 포함, frame에 ndx 존재.
  - `test_collect_all_marks_stale_column_missing` (Review Focus 1): rsp 마지막 유효값이 as_of보다 7영업일 전 → `"rsp" in missing`, `frame["rsp"].iloc[-1]` 이 NaN.
- [ ] **Step 2: 실행 → FAIL 확인** `pytest tests/test_data.py -v`
- [ ] **Step 3: 구현** — yfinance는 `yf.download(list, start=, auto_adjust=True, progress=False)["Close"]`.
- [ ] **Step 4: 실행 → PASS 확인**
- [ ] **Step 5: 실제 네트워크 스모크(수동)** `python -c "from nasdaq_signal.data import collect_all; from nasdaq_signal.config import load_config; m=collect_all(load_config(),'2024-01-01',False); print(m.as_of, m.missing, m.frame.tail(2))"` → as_of가 최근 거래일, ndx 값 존재.
- [ ] **Step 6: Commit** `git commit -m "feat: price and FRED collectors"`

---

### Task 3: 부가 소스 (GPR, Fear & Greed, NQ 선물)

**Files:**
- Create: `nasdaq_signal/data/extras.py`
- Modify: `nasdaq_signal/data/__init__.py` (`collect_all`에서 GPR은 frame 열 `gpr`로, 나머지는 `extras`로)
- Test: `tests/test_extras.py`

**Interfaces:**
- Produces:
  - `fetch_gpr() -> pd.Series` — `https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls` 다운로드, 일별 `GPRD` 열(날짜 열 이름은 파일에서 확인), 이름 `gpr`. 점수에 쓰이는 유일한 부가 소스.
  - `fetch_fear_greed() -> float | None` — `https://production.dataviz.cnn.io/index/fearandgreed/graphdata`, 일반 브라우저 User-Agent, `json["fear_and_greed"]["score"]`. 실패 시 None (예외 전파 금지).
  - `fetch_nq_futures_change() -> float | None` — `NQ=F` 1시간봉 최근 5일의 마지막 가격 / 직전 일봉 종가 - 1. 실패 시 None.
  - `MarketData.extras` 키: `"fear_greed"`, `"nq_change"`.

- [ ] **Step 1: 실패 테스트 작성**
  - `test_fetch_gpr_parses_daily_series`: 작은 가짜 xls(테스트 픽스처, `tests/fixtures/gpr_sample.xls` — `xlwt` 없으면 `read_excel`을 monkeypatch해 DataFrame 반환) → Series, 이름 `gpr`, DatetimeIndex.
  - `test_fear_greed_returns_none_on_http_error`
  - `test_fear_greed_parses_score`: `{"fear_and_greed": {"score": 72.4}}` → `72.4`
  - `test_collect_all_puts_gpr_in_frame_and_fg_in_extras`
- [ ] **Step 2: FAIL 확인** `pytest tests/test_extras.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: PASS 확인**
- [ ] **Step 5: Commit** `git commit -m "feat: GPR, Fear&Greed, NQ futures sources"`

---

### Task 4: 원지표 계산 + 기간별 신호 변환

**Files:**
- Create: `nasdaq_signal/indicators.py`
- Test: `tests/test_indicators.py`

**Interfaces:**
- Consumes: `MarketData.frame` 열 (`ndx, spy, rsp, vix, vix3m, oil, gold, dxy, gpr, dgs10, t10y2y, bamlh0a0hym2, nfci, sahmrealtime`)
- Produces:
  - `compute_raw(frame: pd.DataFrame) -> pd.DataFrame` — 열 = config 지표 이름. 정의:
    - `trend_50 = ndx/SMA50 - 1`, `trend_200 = ndx/SMA200 - 1`, `ret_20 = ndx.pct_change(20)`, `ret_60 = ndx.pct_change(60)`, `drawdown_52w = ndx/rolling_max(252) - 1`, `breadth = (rsp/spy).pct_change(60)`
    - `vix = vix`, `vix_term = vix/vix3m`
    - `gpr = gpr.rolling(30).mean()`, `oil_ret_20 = oil.pct_change(20)`, `gold_ret_20 = gold.pct_change(20)`
    - `us10y_chg_20 = dgs10.diff(20)`, `curve = t10y2y`, `hy_spread = bamlh0a0hym2`, `dxy_ret_60 = dxy.pct_change(60)`, `nfci = nfci`, `sahm = sahmrealtime`
    - 원천 열이 없으면 해당 지표 열은 NaN.
  - `to_signals(raw: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]` — 키 = 기간(short/mid/long), 값 = 날짜×지표 신호 [-1,+1]. 지표별·기간별 transform은 `cfg["indicators"][name]["horizons"][h]` 가 있으면 그것, 없으면 기본값.
    - 공통: `p` = 롤링 백분위 (창 `percentile_window`, `min_periods`), 현재 값이 과거 창 내에서 차지하는 순위 / n (해당 시점까지의 데이터만 사용).
    - `percentile`: `signal = direction * (2p - 1)`
    - `contrarian` (band b = `contrarian_band`): 극단 구간에서만 percentile 신호의 **반대** 방향으로, 크기는 0→1 선형.
      - `p > 1-b`: `s = -direction * (p-(1-b))/b`
      - `p < b`: `s = direction * (b-p)/b`
      - 그 외: `s = 0`
      - 확인: vix(direction=-1)에서 극단 고점(p→1) → +1(상승 신호), 극단 저점(p→0) → -1.

- [ ] **Step 1: 실패 테스트 작성**
  - `test_trend_200_definition`: ndx가 일정하게 상승하는 합성 시리즈 → 마지막 `trend_200 > 0`.
  - `test_missing_source_column_gives_nan_indicator`: frame에 `vix3m` 없음 → `raw["vix_term"]` 전부 NaN.
  - `test_percentile_signal_bounds`: 모든 신호 ∈ [-1, 1] (NaN 제외).
  - `test_percentile_signal_direction`: 단조 증가 시리즈, direction=1 → 마지막 신호 `== pytest.approx(1.0, abs=0.01)`; direction=-1 → `≈ -1.0`.
  - `test_contrarian_vix_extreme_high_is_bullish`: VIX 시리즈 마지막 값이 창 내 최고 → `signals["mid"]["vix"].iloc[-1] > 0.9`, `signals["short"]["vix"].iloc[-1] < -0.9`.
  - `test_contrarian_neutral_inside_band`: 중간 백분위 → mid 신호 == 0.
  - `test_no_lookahead`: 시리즈 끝에 값을 추가해도 이전 날짜의 신호가 변하지 않음.
- [ ] **Step 2: FAIL 확인** `pytest tests/test_indicators.py -v`
- [ ] **Step 3: 구현** (롤링 백분위는 `rolling(window, min_periods).apply(lambda w: (w[:-1] < w[-1]).mean() + 0.5*(w[:-1]==w[-1]).mean(), raw=True)` 수준이면 충분; 속도 문제 시 빌더 재량으로 최적화)
- [ ] **Step 4: PASS 확인**
- [ ] **Step 5: Commit** `git commit -m "feat: indicators and per-horizon signals"`

---

### Task 5: 점수 합산 + 확률 보정표 + 백테스트

**Files:**
- Create: `nasdaq_signal/scoring.py`, `nasdaq_signal/backtest.py`
- Test: `tests/test_scoring.py`, `tests/test_backtest.py`

**Interfaces:**
- Consumes: `to_signals()` 결과, `cfg`
- Produces:
  - `score_frame(signals: dict[str, pd.DataFrame], cfg: dict, ai_adjustment: dict[str, float] | None = None) -> pd.DataFrame` — 열: `{h}_score`, `{h}_missing`(누락 가중치 비율 0~1), `{h}_{category}`(카테고리 점수 -1~1) for h in short/mid/long. 
    - 카테고리 점수 = 해당 카테고리 지표 신호의 단순 평균(NaN 제외). 카테고리 내 지표가 모두 NaN이면 카테고리 NaN.
    - 기간 점수 = Σ(가중치×카테고리 점수) / Σ(유효 카테고리 가중치) × 100, 단 누락 비율 = (카테고리 내 누락 지표 비율 × 카테고리 가중치)의 합.
    - `ai_adjustment[h]`를 `±ai_adjustment_cap`으로 자른 뒤 더하고, 최종 -100~100으로 클립.
  - `arrow(score: float, threshold: float = 20) -> str` — `▲` / `▼` / `─`
  - `class Calibration`: `from_backtest(scores: pd.Series, labels: pd.Series, bucket: int) -> Calibration`, `save(path)`, `load(path) -> Calibration | None` (파일 없으면 None), `probabilities(score: float) -> tuple[int, int, int] | None` (상승/횡보/하락 %, 합 100, 해당 구간 표본 < 30이면 None). 기간별 하나씩 — 파일 형식 `{"short": {"-100": [up, flat, down, n], ...}, ...}`, 로더는 `dict[str, Calibration]` 반환하는 `load_all(path)` 제공.
  - `backtest.py`:
    - `forward_labels(ndx: pd.Series, days: int, up: float, down: float) -> pd.Series` — 값 `"up"|"flat"|"down"`, 미래 데이터 없는 끝부분 NaN.
    - `run_backtest(cfg: dict, frame: pd.DataFrame) -> dict` — `start`~ 구간 점수 계산; 학습(< split)·검증(≥ split) 각각 기간별 `hit_rate`(점수 ≥ threshold → up 예측, ≤ -threshold → down 예측, 그 사이는 flat 예측; 정답 비율), `corr`(점수와 미래수익률 스피어만), `n`. 보정표는 **검증 구간만**으로 생성해 `paths.calibration`에 저장.
    - CLI `python -m nasdaq_signal.backtest` — `collect_all(cfg, "2003-01-01", include_extras=False)` 후 실행, 결과 표 출력.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_scoring.py`
  - `test_all_positive_signals_score_100`: 모든 신호 1 → 각 `{h}_score == 100`, `{h}_missing == 0`.
  - `test_missing_category_renormalizes`: macro 전부 NaN, 나머지 신호 0.5 → `long_score == 50`, `long_missing == pytest.approx(0.60)`.
  - `test_ai_adjustment_capped`: `ai_adjustment={"short": 50}` → short 점수가 기준값 +10.
  - `test_arrow_thresholds`: `arrow(20)=="▲"`, `arrow(19.9)=="─"`, `arrow(-20)=="▼"`.
  - `test_calibration_roundtrip_and_lookup`: 합성 점수/라벨 → save/load 후 `probabilities(35)` 합 100.
  - `test_calibration_sparse_bucket_returns_none`
  - `test_calibration_missing_file_returns_none`
- [ ] **Step 2: 실패 테스트 작성** — `tests/test_backtest.py`
  - `test_forward_labels`: 10일 후 +2% → `"up"`, 끝 10개 NaN.
  - `test_calibration_uses_only_test_period`: 학습 구간 라벨 전부 up, 검증 구간 전부 down인 합성 데이터 → 저장된 보정표 probabilities의 down = 100.
- [ ] **Step 3: FAIL 확인** `pytest tests/test_scoring.py tests/test_backtest.py -v`
- [ ] **Step 4: 구현**
- [ ] **Step 5: PASS 확인**
- [ ] **Step 6: 실제 백테스트 실행** `python -m nasdaq_signal.backtest` → 기간별 학습/검증 hit_rate·corr 표 출력, `state/calibration.json` 생성. 결과 수치를 사용자에게 보고(좋든 나쁘든 그대로).
- [ ] **Step 7: Commit** `git commit -m "feat: scoring, calibration, backtest"`

---

### Task 6: 카카오 토큰 관리 + 발송 + 최초 인증 CLI

**Files:**
- Create: `nasdaq_signal/kakao.py`, `nasdaq_signal/kakao_auth.py`
- Test: `tests/test_kakao.py`

**Interfaces:**
- Produces:
  - `class TokenStore(Protocol)`: `load() -> dict | None`, `save(tokens: dict) -> None`
  - `class FileTokenStore(TokenStore)`: `__init__(path: str)` — JSON. 토큰 dict 키: `access_token`, `refresh_token`, `refresh_token_expires_at`(ISO8601 UTC).
  - `class KakaoAuthError(RuntimeError)`
  - `class KakaoClient`: `__init__(rest_api_key: str, store: TokenStore, client_secret: str | None = None)`
    - `refresh() -> None` — `POST https://kauth.kakao.com/oauth/token` (`grant_type=refresh_token, client_id, refresh_token[, client_secret]`). 응답에 `refresh_token`이 있으면 교체하고 `refresh_token_expires_in`(초)로 만료시각 갱신. 저장. 400 `invalid_grant` 또는 토큰 없음 → `KakaoAuthError("카카오 재인증 필요: python -m nasdaq_signal.kakao_auth")`.
    - `days_until_reauth() -> int`
    - `send_text(text: str, link_url: str) -> None` — `POST https://kapi.kakao.com/v2/api/talk/memo/default/send`, `Authorization: Bearer`, form `template_object=json.dumps({"object_type":"text","text":text,"link":{"web_url":link_url,"mobile_web_url":link_url}}, ensure_ascii=False)`. `len(text) > 200` 이면 `ValueError`. 응답 `result_code != 0` → `RuntimeError`.
  - `kakao_auth.py` CLI: 인가 URL(`https://kauth.kakao.com/oauth/authorize?client_id=..&redirect_uri=..&response_type=code&scope=talk_message`) 출력 → 사용자가 브라우저 로그인 후 리다이렉트된 주소 전체를 붙여넣음 → `code` 추출 → `grant_type=authorization_code`로 토큰 발급 → store 저장 → 테스트 메시지 "연결 완료" 발송.

- [ ] **Step 1: 실패 테스트 작성** (`requests.post` monkeypatch)
  - `test_refresh_rotates_refresh_token_and_expiry`
  - `test_refresh_keeps_old_refresh_token_when_not_returned`
  - `test_refresh_invalid_grant_raises_auth_error` (Review Focus 3): 메시지에 `kakao_auth` 포함.
  - `test_send_text_rejects_over_200_chars`
  - `test_send_text_posts_template_with_link`: 전송 payload의 JSON에 `object_type == "text"`, link 포함, 한글이 이스케이프 없이 들어감.
  - `test_days_until_reauth`
- [ ] **Step 2: FAIL 확인** `pytest tests/test_kakao.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: PASS 확인**
- [ ] **Step 5: 수동 연결 (사용자와 함께)** — 사용자가 developers.kakao.com에서 앱 생성, 카카오 로그인 활성화, Redirect URI `http://localhost:5000/callback` 등록, 동의항목 '카카오톡 메시지 전송' 설정, REST API 키를 `.env`에 입력. `python -m nasdaq_signal.kakao_auth` 실행 → 카톡에 "연결 완료" 수신 확인.
- [ ] **Step 6: Commit** `git commit -m "feat: kakao token management and send"`

---

### Task 7: 메시지 포맷

**Files:**
- Create: `nasdaq_signal/message.py`
- Test: `tests/test_message.py`

**Interfaces:**
- Consumes: `score_frame` 마지막 행, `Calibration`, `MarketData`
- Produces:
  - `@dataclass Report`: `as_of: date; run_date: date; market_closed: bool; scores: dict[str, float]; probs: dict[str, tuple[int,int,int] | None]; categories: dict[str, float]` (기간 무관 카테고리 표시값 = mid 기간 카테고리 점수), `key_values: dict[str, float]` (`vix`, `us10y`, `hy_spread`), `fear_greed: float | None; nq_change: float | None; missing: list[str]; low_confidence: bool; reauth_days: int | None`
  - `format_messages(report: Report, arrow_threshold: float = 20) -> list[str]` — 정확히 2개, 각 ≤ 200자.

메시지 1 형식:
```
[나스닥 방향성] 11/24(화) · 11/23 마감
단기 ▲ +32 상55 횡30 하15
중기 ─ +5 상38 횡40 하22
장기 ▲ +41 상62 횡23 하15
```
- 휴장일: 둘째 줄 앞 헤더를 `11/24(일) · 휴장, 11/21 기준`으로.
- 확률 None이면 점수까지만.
- `low_confidence`면 마지막 줄 `⚠ 신뢰도 낮음(데이터 누락)`.

메시지 2 형식 (줄 우선순위 높은 순 — 200자 넘으면 낮은 순위부터 제거, Review Focus 4):
1. `시장 +0.4 심리 +0.1 지정학 -0.2 매크로 +0.3`
2. `⚠ 재인증 필요: D-{n}` (reauth_days ≤ 14일 때)
3. `⚠ 누락: {이름들}` (너무 길면 `외 N개`로 축약)
4. `VIX 14.2 · 10년물 4.21% · HY 3.10%`
5. `월요일 NQ선물 +0.8%` (nq_change 있고 run_date가 월요일)
6. `공포탐욕 72(참고)`

- [ ] **Step 1: 실패 테스트 작성**
  - `test_message_one_layout`: 예시 Report → 메시지 1이 위 형식과 일치(문자열 비교).
  - `test_message_one_without_probabilities`
  - `test_closed_market_header`
  - `test_low_confidence_line`
  - `test_all_warnings_fit_200_chars` (Review Focus 4): 누락 10개, 재인증 D-3, NQ, F&G 전부 → 두 메시지 모두 ≤ 200자, 1번 줄(카테고리)과 재인증 경고는 반드시 포함.
  - `test_nq_line_only_on_monday`
- [ ] **Step 2: FAIL 확인** `pytest tests/test_message.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: PASS 확인**
- [ ] **Step 5: Commit** `git commit -m "feat: kakao message formatting"`

---

### Task 8: 일일 실행 진입점 + 기록 + 스케줄러 등록

**Files:**
- Create: `nasdaq_signal/main.py`, `scripts/register_task.ps1`, `README.md`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: 위 모든 모듈
- Produces:
  - `build_report(cfg: dict, md: MarketData, run_date: date, reauth_days: int | None) -> tuple[Report, dict]` — 두 번째 값은 history 행(dict: 날짜, 원지표 마지막 값, 신호, 점수, 누락).
  - `is_market_closed(as_of: date, run_date_kst: date) -> bool` — `as_of != run_date_kst - 1일` 이면 True (KST 오전 실행 = 미국 전날 세션 기준).
  - `append_history(path: str, row: dict) -> None` — `utf-8-sig` CSV, 같은 `date` 행은 덮어씀 (Review Focus 5).
  - `main(argv: list[str] | None = None) -> int` — 옵션 `--dry-run`(발송 없이 출력), `--config PATH`. 흐름: load_config → `.env` 로드 → `collect_all(cfg, start=오늘-7년)` → build_report → append_history → KakaoClient.refresh → 메시지 2개 발송. `DataSourceError` 시 `"[나스닥 방향성] 오류: 핵심 데이터 수집 실패 — 로그 확인"` 1건 발송 후 1 반환. `KakaoAuthError` 시 로그 기록 후 2 반환. 로그는 `state/logs/YYYY-MM-DD.log` (UTF-8). run_date는 `Asia/Seoul` 기준(`zoneinfo`).

- [ ] **Step 1: 실패 테스트 작성** (collect_all·KakaoClient를 가짜로 주입)
  - `test_is_market_closed`: as_of 금 11/20, run 토 11/21 → False; run 일 11/22 → True; run 월 11/23 → True.
  - `test_append_history_upserts_same_date` (Review Focus 5): 같은 날짜 두 번 → 행 1개, 두 번째 값.
  - `test_dry_run_sends_nothing`: 가짜 KakaoClient의 send 호출 0회, 반환 0.
  - `test_core_data_failure_sends_error_message`: collect_all이 DataSourceError → send 1회, 텍스트에 "오류", 반환 1.
  - `test_fred_missing_still_sends_with_warning` (Review Focus 2): FRED 열 없음 → 메시지 2개 발송, 메시지 1에 "신뢰도 낮음" 포함.
  - `test_auth_error_returns_2`
- [ ] **Step 2: FAIL 확인** `pytest tests/test_main.py -v`
- [ ] **Step 3: 구현**
- [ ] **Step 4: PASS 확인** — 전체 `pytest -v` 도 PASS.
- [ ] **Step 5: 실제 dry-run** `python -m nasdaq_signal.main --dry-run` → 메시지 2개 콘솔 출력, `state/history.csv` 행 추가.
- [ ] **Step 6: 실제 발송** `python -m nasdaq_signal.main` → 카톡 수신 확인 (사용자).
- [ ] **Step 7: `scripts/register_task.ps1`** — `Register-ScheduledTask` 이름 `NasdaqSignal`, 매일 09:00, 작업 디렉터리 = 프로젝트 루트, 실행 = `.venv\Scripts\python.exe -m nasdaq_signal.main`, `StartWhenAvailable`(PC가 꺼져 있다 켜지면 실행). README에 설치·카카오 앱 설정·인증·백테스트·스케줄 등록 순서 기록.
- [ ] **Step 8: Commit** `git commit -m "feat: daily runner, history, scheduler"`
