# 로컬 원본 기반 단일 배치 시험

기존 전처리와 11_extract_candidate2_usable.py의 SUCCESS 선별을 재사용합니다.
정확히 한 배치만 실행하며 다음 배치 자동 실행과 자동 삭제는 적용하지 않습니다.

1. H: mini_dataset에서 원본 WAV/JSON을 읽습니다. Drive 캐시 공간은 별도로 필요합니다.
2. D:\ieum_local_pilot\batch_runs에 전체 전처리 결과와 체크포인트를 저장합니다.
3. SUCCESS 중 1~30초 WAV/TXT만 검증해 processed_candidate2에 복사합니다.
4. 선별 결과만 archives/BATCH_010.zip으로 묶고 내부 SHA256을 검증합니다.
5. 업로드 토큰을 지정하면 기존 API 업로더로 원격 크기·MD5를 검증합니다.
   토큰이 없으면 H: 동기화 폴더에 전달하며 실제 업로드 완료·시간은 확인하지 못합니다.

원본, 중간 결과, 선별 결과와 ZIP은 자동 삭제하지 않습니다.
기존 100GB 여유 공간 보호와 전처리 결과 상한 계산을 유지합니다.
기존 ZIP과 내용이 다르면 덮어쓰지 않고 중단합니다.

프로젝트 루트에서 점검:
    .\scripts\06_batch_preprocessing\run_local_batches.ps1

실행:
    .\scripts\06_batch_preprocessing\run_local_batches.ps1 -Execute -UploadToken '실제\drive_upload_token.json'

시험 배치는 사용자가 지정한 BATCH_010입니다. 기존 작업 기록에서도 미처리인지 확인해야 합니다.
-WorkRoot로 D: 기본 작업 위치를 변경하고 -BatchIds로 정확히 한 배치를 지정합니다.
실제 실행에는 pydub가 필요합니다. API 업로드에는 google-auth, google-api-python-client 및
기존 drive.file 범위 업로드 토큰이 필요합니다. 토큰 내용은 공유하거나 Git에 넣지 않습니다.

성공한 실행의 batch_runs/BATCH_010_handoff_*.json에 전처리, 선별, ZIP 생성·검증,
업로드·검증 시간을 기록합니다. 기존 ZIP 업로드를 건너뛴 경우는 실제 업로드 측정이 아닙니다.
동기화 폴더 전달 시간도 실제 네트워크 업로드 시간이 아닙니다.
전처리에는 원본 다운로드 시간이 포함될 수 있으며 재개 실행은 완료된 녹음을 건너뜁니다.

테스트:
    python -m unittest discover -s scripts/06_batch_preprocessing -p test_local_batches.py

## 선읽기 시험

-Prefetch를 붙이면 녹음 하나를 처리하는 동안 다음 녹음의 WAV/JSON을
D:\ieum_local_pilot\staged_sources로 복사합니다. 한 작업자만 사용하며 원본을 바꾸지 않습니다.
현재 파일은 D:에서 읽고, 이미 검증된 전처리 체크포인트는 재개 시 건너뜁니다.
중단된 녹음은 별도 새 attempt에서 재처리하므로 처리 중인 녹음 하나는 다시 시작할 수 있습니다.
선읽기 파일은 자동 삭제하지 않습니다. 원본 파일 용량과 100GB 여유 보호를 유지합니다.

    .\scripts\06_batch_preprocessing\run_local_batches.ps1 -Execute -Prefetch

Prefetch의 staged는 복사 시간, wait는 아직 복사가 끝나지 않아 기다린 시간입니다.
복사 시간은 전처리와 겹치므로 두 시간을 단순 합산하면 전체 실행 시간이 아닙니다.
공간 점검은 매 녹음마다 유지하고 전체 원본 검증은 초기 점검에 수행합니다.
이후 처리 직전에는 해당 녹음의 원본/복사본 크기를 확인합니다.
개선 전후 파일 길이와 Drive 캐시 상태가 달라 같은 배치의 구간 비교만으로
인과관계나 전체 배치 단축 폭을 확정할 수 없습니다.

## 현재 실행 방식

선읽기 시험을 종료했습니다. PowerShell 실행기는 -Prefetch 옵션을 사용하지 않으며,
원본을 H:에서 직접 읽습니다. 초기 전체 원본 검증 및 녹음별 크기 검증을 유지하고,
매 녹음 직전에는 현재 여유 공간과 남은 결과 용량만 점검합니다.
기존 staged_sources 파일과 완료 결과는 삭제하지 않습니다.

## 배치 28 → 27 → 26 → 25 실행

-BatchIds BATCH_028 BATCH_027 BATCH_026 BATCH_025로 지정한 순서를 유지합니다.
선읽기를 사용하지 않고 H: 원본을 직접 읽습니다.
각 배치의 SUCCESS 선별, ZIP 생성 및 Drive 전달이 끝난 뒤 다음 배치를 판단합니다.
UploadToken을 제공하면 API 원격 업로드 크기·MD5 검증 후 다음 배치를 시작합니다.
토큰이 없으면 첫 배치의 ZIP을 동기화 폴더에 전달한 뒤 원격 완료 미확인으로 중단합니다.
이는 다음 배치의 실행 실패가 아니라 원격 업로드 확인을 기다리는 상태입니다.

## 다음 배치 진행 조건 (최신 정책)

로컬 결과를 삭제하지 않으므로 원격 업로드 완료를 기다리지 않습니다.
SUCCESS ZIP 생성·검증과 Drive 동기화 폴더 전달·검증이 끝나면 다음 배치를 시작합니다.
cloud_upload_verified=false는 실제 원격 완료를 확인하지 않았다는 기록이며 실행 중단 조건이 아닙니다.
전처리 ERROR 또는 ZIP/전달 검증 실패는 다음 배치 진행을 중단합니다.
현재 요청 범위는 27 완료 후 26 → 25이며, 24 이하를 자동 추가하지 않습니다.

## Drive 읽기 오류에서 재개

2026-10-08 배치 25는 Drive 원본 크기가 정상이어도 스트리밍 복사 도중
`OSError: [Errno 22] Invalid argument`가 발생해 중단했습니다.
`-StagingDir D:\ieum_local_pilot\staged_retry9 -RetryErrors`로 재개하면
완료 체크포인트를 검증해 건너뛰고, 남은 원본을 로컬에 복사한 후 처리합니다.
선읽기는 Errno 22에 한해 2초·4초 후 두 번 재시도합니다.
실패한 partial은 보존하고 새 partial로 복사하며, 크기·여유 공간 검증은 유지합니다.
복사 파일 경로와 10초 간격 진행량을 로그에 출력합니다.
최종 재시도도 실패하면 기존과 같이 배치를 중단합니다.
