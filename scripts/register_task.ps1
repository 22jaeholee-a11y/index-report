# 매일 09:00 나스닥 방향성 알림을 Windows 작업 스케줄러에 등록합니다.
# 실행: powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "가상환경이 없습니다: $python" }

$action = New-ScheduledTaskAction -Execute $python -Argument "-m nasdaq_signal.main" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At 9:00am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
Register-ScheduledTask -TaskName "NasdaqSignal" -Action $action -Trigger $trigger -Settings $settings -Description "나스닥 방향성 카카오톡 알림" -Force | Out-Null
Write-Host "등록 완료: 작업 이름 NasdaqSignal, 매일 09:00 (PC가 꺼져 있었으면 켜진 뒤 실행)"
