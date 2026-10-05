# BATCH_001 다운로드 실패 진단 및 인증 재개 (2026-10-05)

## 확인된 기록
- preprocessing_results_2c72f72fa5c24bbebb9331c82dd5e028.csv 및 247 checkpoint의 status/failure_reason 일치.
- SUCCESS 15, NEEDS_MANUAL_REVIEW 1, ERROR 231. 첫 ERROR는 17번째이며 이후 231개 모두 FileURLRetrievalError. ERROR의 VAD1 count는 모두 비어 있음.
- partial 231개 모두 0 byte; 최종 WAV/JSON 파일 중 0 byte 없음.
- gdown은 use_cookies=False, resume=False. 예외가 recording 단위로 격리되지만 download retry/backoff는 없었음.
- Google 오류 문자열은 권한/과다 접근 양쪽을 포함하므로 특정 quota/IP 제한이라고 확정 불가. 초기 16개 성공 이후 연속 실패는 제한 가능성을 뒷받침하지만 서로 다른 파일의 권한 문제를 배제하지 못함.

## 기존 resume 구조
- 기존 batch_binding.json으로 plan/core/입출력 경로 확인 및 실행 lock.
- 각 recording의 마지막 attempt_*/result.json을 읽음.
- --resume --retry-errors는 SUCCESS/manual을 건너뛰고 ERROR만 새 attempt 디렉터리에서 재처리.
- SUCCESS artifact 헤더/segment index/텍스트 해시 검증. manual recording은 원래 로직에 따라 usable에 포함 안 함.
- 정상 크기의 최종 source 재사용, 크기 불일치/0 byte 최종 source는 덮어쓰기 없이 거부.
- 기존 partial은 삭제/덮어쓰기/이어쓰기하지 않음. 새 partial로 재시도; 성공 시 정확한 크기 확인 후 rename. recording checkpoint resume이며 byte-offset resume은 아님.
- 다운로드 → pairing → 기존 core 처리 순서가 recording 단위로 수행됨. core는 변경 없음.

## 최소 수정
--download-backend drive-api 추가(기존 gdown은 선택 가능). 인증된 Drive API files.get metadata + alt=media를 exact file ID로 호출.
ID/name/size/canDownload 및 제공되는 MD5 검증. 스트리밍 중 기존 byte cap/100GB reserve 보호. 네트워크/429/5xx/특정 rate-limit 403은 exponential backoff+jitter로 최대 3회 추가 재시도. 권한/파일 없음/quota 등 영구 오류는 억지 우회 안 함.
부분 다운로드는 보존하고 새 partial 사용. 기존 최종 source/checkpoints/artifacts는 보존.
--authorize-drive는 사용자가 명시 실행하는 별도 OAuth 단계이며 WAV/JSON 다운로드 및 preprocessing을 실행하지 않음.
Drive API 실행 전 token 로드/필요 시 refresh; token 파일은 덮어쓰지 않음. dry-run은 인증/refresh/API 호출도 하지 않음.

## 사용자 준비 단계 (이번 작업에서 수행하지 않음)
1. Google Cloud 프로젝트에서 Google Drive API 활성화.
2. OAuth 동의 화면 설정. 테스트 상태이면 실제 폴더 접근 계정을 test user로 추가. 외부 앱 테스트 상태의 refresh token 만료 등 장기 실행 제한은 Google 정책에 따라 확인.
3. OAuth client를 Desktop app 유형으로 생성하고 client JSON을 C:\ieum\credentials\drive_client_secret.json에 저장. 이 파일/발급 token은 공유하거나 Git에 넣지 말 것.
4. 아래 패키지를 지정 환경에 설치한 뒤 별도 OAuth 명령 실행. 실제 원본 폴더에 접근하는 동일 계정으로 로그인하고 Drive 읽기 권한에 동의.

```powershell
& 'C:\ieum\.venv_smoke\Scripts\python.exe' -m pip install google-auth google-auth-oauthlib requests
& 'C:\ieum\.venv_smoke\Scripts\python.exe' -B 'C:\ieum\scripts\06_batch_preprocessing\10_run_candidate2_batch.py' --authorize-drive --drive-client-secrets 'C:\ieum\credentials\drive_client_secret.json' --drive-token 'C:\ieum\credentials\drive_readonly_token.json'
```

기존 token이 있으면 덮어쓰지 않으므로 새 경로를 지정하고 이후 명령도 동일 경로 사용.
OAuth 준비 없이는 실제 다운로드 불가. Codex의 Drive connector 인증은 로컬 Python에 자동 전달되지 않음.

## 다운로드 없이 재개 대상 검증
```powershell
& 'C:\ieum\.venv_smoke\Scripts\python.exe' -B 'C:\ieum\scripts\06_batch_preprocessing\10_run_candidate2_batch.py' --batch-id BATCH_001 --output-dir 'C:\ieum\results\full_expansion\candidate2_batch_runs' --resume --retry-errors --download-backend drive-api --drive-token 'C:\ieum\credentials\drive_readonly_token.json' --dry-run
```
기대값: completed_skipped=16, pending_recordings=231. 인증 성공 여부는 이 dry-run으로 검증되지 않음.

## 인증 준비 후 사용자 실행 명령 (이번 작업에서 실행하지 않음)
```powershell
& 'C:\ieum\.venv_smoke\Scripts\python.exe' -B 'C:\ieum\scripts\06_batch_preprocessing\10_run_candidate2_batch.py' --batch-id BATCH_001 --output-dir 'C:\ieum\results\full_expansion\candidate2_batch_runs' --resume --retry-errors --download-backend drive-api --drive-token 'C:\ieum\credentials\drive_readonly_token.json' --execute
```
이 명령은 ERROR 231개의 다운로드뿐 아니라 성공적으로 다운로드/검증된 recording의 기존 preprocessing도 수행한다. 완료된 16개는 건너뛴다. BATCH_002 이후 및 cleanup은 실행하지 않는다.

## 검증
Syntax, original safety self-test, mock authenticated-download tests, 실제 checkpoint/artifact dry-run. 보호 함수 AST 및 core SHA 유지. 실제 API 인증/다운로드는 미검증이며 실행하지 않았음.

공식 근거:
- https://developers.google.com/workspace/drive/api/guides/manage-downloads
- https://developers.google.com/workspace/drive/api/guides/handle-errors
- https://developers.google.com/workspace/guides/create-credentials
