# E1 개인화 Whisper Small 시연 서버

## 구성과 재사용 범위

`feature/e1-demo-api`는 개인화 연구 브랜치에서 분기했다. integration 전체 병합이나 Raspberry Pi 코드 복사는 하지 않았다. integration 참조 커밋 `07f55d1246c2b39e9628d1f383da9714b4fc6135`의 `src/api/main.py` 요청/응답, 화자 검증, lifespan 로딩, HTTP 오류 처리 구조와 `src/tts/piper_tts.py`의 Piper CLI 호출/UUID 출력 방식을 선택적으로 재사용했다.

기존 ModelLoader/ASRInference의 BiGRU·CTC 구현은 가져오지 않았다. `src/api/model_loader.py`와 `whisper_inference.py`는 Hugging Face Whisper Encoder–Decoder 전용이다. 기존 연구 소스, 체크포인트와 평가 파일은 변경하지 않는다.

## 서버 준비 및 실행

프로젝트 루트에서 기존 연구용 `.venv`를 사용한다. API 추가 패키지 목록은 `requirements_demo_api.txt`이다. 현재 환경에는 설치를 완료했으며 Torch/Transformers는 교체하지 않았다. 새 서버에 설치할 때는 해당 서버 환경을 확인한 뒤 아래 목록을 설치한다. 이 파일은 전체 연구 환경의 lock 파일이 아니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_demo_api.txt
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`--workers 1`을 유지한다. 각 프로세스가 두 모델을 로딩하므로 worker를 늘리면 VRAM 사용량도 증가한다. 모델은 서버 시작 시 로컬에서 한 번씩 로딩한다. 최초 경로/로딩 오류는 서버 시작 실패로 처리한다. 자동 다운로드나 다른 체크포인트로의 대체는 하지 않는다. 두 모델의 GPU 추론은 프로세스 내 lock으로 직렬화한다.

이 서버는 인증 없는 시연용이다. 팀 내부 네트워크에서 사용하고 외부 인터넷 공개용으로 배포하지 않는다. Raspberry Pi의 서버 주소는 서버 PC의 LAN IP이며 `127.0.0.1`은 Pi 자신을 가리킨다. 필요한 경우 서버 PC의 방화벽에서 시연 포트를 허용해야 한다. 이번 검증에서는 방화벽이나 공유기를 변경하지 않았다.

## 모델 설정

`configs/e1_demo_api.yaml`의 상대경로는 프로젝트 루트를 기준으로 해석한다.

| 설정 | 기본값 |
|---|---|
| General | `checkpoints/whisper_small_general_v1/model` |
| General Processor | `checkpoints/whisper_small_general_v1/processor` |
| 개인화 | `outputs/personalization/E1/KSM_M_45/best_model` |
| 개인화 Processor | 위 개인화 디렉터리 |
| 지원 화자 | `KSM_M_45` |
| 업로드 파일 상한 | 32MiB |

`IEUM_API_CONFIG`로 별도 YAML을 선택할 수 있다. `IEUM_GENERAL_MODEL`, `IEUM_GENERAL_PROCESSOR`, `IEUM_PERSONALIZED_MODEL`, `IEUM_PERSONALIZED_PROCESSOR`, `IEUM_DEVICE`, `IEUM_PRECISION` 환경변수도 지원한다. device 기본값 auto, precision 기본값 auto이다. CUDA BF16 지원 시 BF16 autocast, 그 외 CUDA에서는 FP16 autocast, CPU에서는 FP32를 사용한다. BF16 사용 불가 GPU에 명시적으로 BF16을 지정하면 오류다.

Tokenizer는 로컬 `tokenizer.json`을 사용한다. Feature Extractor는 저장된 `processor_config.json`의 중첩된 `feature_extractor` 설정으로 구성한다. mono 평균·librosa 16kHz 리샘플링·30초 패딩·attention mask를 적용한다. 빈 음성·30초 초과·NaN/Inf는 거부한다. Korean/transcribe, forced_decoder_ids=None, max_length160, beam1, sampling=False로 `generate()`한다. CTC decoding이나 Gemini 후처리는 없다.

API 출력은 디코딩 후 앞뒤 공백만 제거한 원문이다. CER/WER용 정규화는 API 출력에 적용하지 않는다. 비교 테스트에서는 기존 `normalize_text()`로 저장 Prediction과 비교한다. MP3/M4A/FLAC 확장자는 기존 계약을 유지하지만 실제 디코딩은 설치된 libsndfile 지원에 달려 있다. 팀 간 전송에는 **16kHz mono PCM16 WAV**를 사용한다.

## API 명세: Raspberry Pi 담당자 전달용

서버 주소 예: `http://<서버_PC_LAN_IP>:8000`.

| 메서드/경로 | 입력 | 성공 응답 |
|---|---|---|
| GET `/health` | 없음 | 상태·모델 로딩 여부·지원 화자·TTS 준비 여부 |
| POST `/transcribe` | multipart `audio` | `{filename, text}` (General) |
| POST `/transcribe/personalized` | multipart `audio`, Form `speaker_id` | `{speaker_id, filename, text}` |
| POST `/transcribe/compare` | multipart `audio`, Form `speaker_id` | 아래 비교 JSON |
| POST `/tts` | JSON `{text}` | `audio/wav` 파일 |

```json
{
  "speaker_id": "KSM_M_45",
  "filename": "recorded_audio.wav",
  "duration_seconds": 4.5,
  "general_text": "범용 인식 결과",
  "personalized_text": "개인화 인식 결과"
}
```

필드명은 **audio**이며 file이 아니다. 화자 문자열은 **KSM_M_45**이다. 기존 `ASRApiClient.transcribe_compare(audio_path, speaker_id)`가 사용하는 계약과 같다. 실제 장치 코드의 지원 화자 목록 등은 Raspberry Pi 담당자가 별도로 설정해야 한다. 이 브랜치는 해당 코드를 포함하지 않는다.

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/transcribe/compare -F "speaker_id=KSM_M_45" -F "audio=@C:/path/to/sample.wav;type=audio/wav"
```

입력 누락은 422, 미지원 화자·빈/손상 음성·30초 초과는 400, 파일 크기 초과는 413, Piper 미설정은 503이다. 내부 추론/합성 실패는 500이며 자세한 원인은 서버 로그에 남긴다. JSON의 `detail`을 확인하면 된다. 클라이언트에서는 업로드/추론 대기 시간을 고려해 기존 60초 timeout을 유지하고 실제 환경에서 지연을 확인한다.

## 선택적 Piper TTS

`piper_executable`, `piper_model`, `piper_config`, `tts_output`을 YAML로 설정한다. 실행 파일은 `IEUM_PIPER_EXECUTABLE`로도 지정할 수 있다. integration과 동일하게 `piper -m MODEL -c CONFIG -f OUTPUT`에 텍스트를 stdin으로 전달한다. 한국어 모델은 자동 다운로드/설치하지 않는다.

실행 파일과 음성 모델/설정이 없으면 ASR 서버는 계속 실행되고 `/health`는 `tts_ready=false`, `/tts`는 503을 반환한다. 합성은 최대 60초이며 UUID 파일을 만들고 HTTP 전송이 끝나면 해당 임시 WAV를 삭제한다. 공백 텍스트는 400이며 입력은 최대 2000자다. 실제 한국어 발음 품질과 Piper 버전 호환성은 설치 후 검증해야 한다.

## 검증 결과 (2026-11-09)

```powershell
$env:HF_HUB_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe scripts/08_demo/test_api.py
```

실제 CUDA General/E1 모델과 KSM Personal Test WAV 003250·003254를 사용한 ASGI TestClient 검증을 통과했다. `/transcribe/compare`의 두 문장은 직접 추론과 정확히 같고, 정규화한 기존 저장 Prediction과도 일치했다. 개별 `/transcribe`, `/transcribe/personalized`, `/health`도 통과했다.

| ID | General | E1 |
|---|---|---|
| 003250 | 이분달 멍멍이 온 세 알려 주세요 | 이번달 물병자리 운세 알려 줘 |
| 003254 | 이동할 때를 기다리며 히히덕거렸다 | 이번 주 물병자리 운세 알려 줘 |

파일 누락·빈 파일·손상 파일·화자 오류·확장자 오류·30초 초과·업로드 상한·stereo/44.1kHz 변환과 누락 모델 시작 실패를 확인했다. 기존 연구 소스, 모델, 분할, 예측/평가 파일 51개의 SHA256이 검증 전후 동일했다. 결과는 `outputs/e1_demo/verification/results.json`에 저장한다.

검증 환경: RTX5060, torch2.7.1+cu128, transformers5.18.0, fastapi0.162.2, uvicorn0.54.0, python-multipart0.0.33, httpx0.28.1. Transformers의 생성 인자 관련 deprecation 경고는 발생했으나 추론과 비교는 통과했다.

실제 Raspberry Pi 하드웨어, LAN HTTP 통신/방화벽, GPIO, 녹음, 오디오 재생, Piper 실제 합성은 검증하지 않았다. 서버 API는 ASGI 내부 호출로 검증했으며 외부에 서버를 계속 띄워 두지 않았다. 실제 장치에서 동일 WAV fixture 업로드 → 실제 마이크 순으로 확인한다. 이 작업에서는 Raspberry Pi 소스 복사/수정, 추가 학습, Git push, 브랜치 merge를 수행하지 않았다.
