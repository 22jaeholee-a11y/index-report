# 나스닥 방향성 일일 알림

매일 09:00(KST) 나스닥 단기(10거래일)·중기(63거래일)·장기(252거래일) 방향성 점수(-100~+100)와
과거 검증 기반 확률을 카카오톡 '나에게 보내기'로 받습니다. 1단계는 규칙 기반 수치 모델입니다.

설계: `docs/superpowers/specs/2026-11-23-nasdaq-direction-design.md`

## 설치
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env   # 값 입력
```

## 1. FRED API 키 (무료)
https://fred.stlouisfed.org/docs/api/api_key.html 에서 발급 → `.env` 의 `FRED_API_KEY`.
없으면 금리·스프레드 등 매크로 지표가 빠지고 메시지에 "신뢰도 낮음"이 표시됩니다.

## 2. 카카오 앱 설정 (최초 1회)
1. https://developers.kakao.com → 내 애플리케이션 → 애플리케이션 추가
2. 앱 키의 **REST API 키** → `.env` 의 `KAKAO_REST_API_KEY`
3. 카카오 로그인 → 활성화 ON, Redirect URI에 `http://localhost:5000/callback` 등록
4. 동의항목 → **카카오톡 메시지 전송(talk_message)** 설정
5. (보안 → Client Secret 을 켰다면 `.env` 의 `KAKAO_CLIENT_SECRET`)
6. `.venv\Scripts\python -m nasdaq_signal.kakao_auth` 실행 → 안내에 따라 로그인 → 카톡에 "연결 완료" 수신

리프레시 토큰은 약 2개월 유효하며 매일 실행 시 자동 갱신됩니다. 만료 14일 전부터 메시지에 재인증 경고가 붙습니다.

## 3. 방향 튜닝 + 백테스트 (확률표 생성)
```powershell
.venv\Scripts\python -m nasdaq_signal.tune       # 학습 구간(2010~2017)만으로 지표별 방향/제외 → tuning.yaml
.venv\Scripts\python -m nasdaq_signal.backtest
```
학습(2010~2017)/검증(2018~) 구간별 적중률·상관을 출력하고, 검증 구간 결과로 `state/calibration.json`을 만듭니다.
`tune`은 학습 구간을 앞/뒤 절반으로 나눠 두 절반 모두 같은 방향으로 |IC|≥0.03인 지표만 쓰고, 음수면 방향을 뒤집습니다. 검증 구간은 보지 않습니다.
확률표가 있으면 화살표는 가장 가능성 높은 결과를 따르고, 점수는 '평소 대비 우호도'를 뜻합니다.
이 파일이 없으면 메시지에 확률이 표시되지 않습니다. 가중치(`config.yaml`)를 바꾸면 다시 실행하세요.

## 4. 실행
```powershell
.venv\Scripts\python -m nasdaq_signal.main --dry-run   # 발송 없이 출력
.venv\Scripts\python -m nasdaq_signal.main             # 카톡 발송
powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1   # 매일 09:00 등록
```
기록: `state/history.csv`(일별 지표·점수), `state/logs/`(실행 로그).

## 종료 코드
0 정상 · 1 핵심 데이터(나스닥 가격) 수집 실패(오류 메시지 발송) · 2 카카오 인증 문제
