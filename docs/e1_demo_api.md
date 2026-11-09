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

실행 파일과 음성 모델/설정이 없으면 ASR 서버는 계속 실행되고 `/health`는 `tts_ready=false`, `/tts`는 503을 반환한다. 합성은 최대 60초이며 UUID 파일을 만들고 HTTP 전송이 끝나면 해당 임시 WAV를 삭제한다. 공백 텍스트는 400이며 입력은 최대 2000자다. 2026-11-10 현재 로컬 Windows 서버의 설치·합성·실제 HTTP 검증은 완료했으며 아래 TTS 전용 절차를 따른다. 실제 한국어 발음 품질은 청취 확인이 별도로 필요하다.

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

## TTS 전용 환경과 실제 검증 (2026-11-10)

위 2026-11-09 검증 당시 미설정이던 Piper를 설치했다. ASR `.venv`와 분리된 **`.venv_tts`**에만 설치하며, 서버 자체는 계속 ASR `.venv`로 실행한다. Piper는 CPU용 ONNX Runtime을 사용하므로 Whisper의 CUDA 패키지를 바꾸지 않는다.

새 환경 재구성은 프로젝트 루트에서 다음 순서로 실행한다. 아래 명령은 설치된 로컬 환경에서 다시 실행할 필요가 없다. 다운로드 스크립트는 기존 모델을 덮어쓰지 않으므로 새 경로에 모델을 준비할 때만 실행한다.

```powershell
.\.venv\Scripts\python.exe -m venv .venv_tts
.\.venv_tts\Scripts\python.exe -m pip install -r requirements_tts.txt
.\.venv_tts\Scripts\python.exe scripts/08_demo/download_tts_voice.py
```

실제 설치 버전은 Piper1.8.0, onnxruntime1.31.1, pathvalidate3.3.1이다. 전체 의존 패키지 기록은 `outputs/e1_demo/tts_verification_20261110/tts_packages.txt`에 있다.

설정 파일은 다음과 같다. 실행 경로는 서버의 현재 디렉터리가 아닌 프로젝트 루트를 기준으로 해석한다. 서버 프로세스의 PATH에 TTS 가상환경을 추가하거나 ASR 환경을 바꿀 필요가 없다.

```yaml
piper_executable: .venv_tts/Scripts/piper.exe
piper_model: piper/ko_KR-kss-medium.onnx
piper_config: piper/ko_KR-kss-medium.onnx.json
tts_output: outputs/e1_demo/tts
```

Linux 서버에서는 별도 환경을 구성한 뒤 `IEUM_PIPER_EXECUTABLE`을 해당 서버의 절대 경로 또는 `.venv_tts/bin/piper`로 지정한다. Python 패키지 가상환경을 Windows에서 Linux로 복사하지 않는다. `IEUM_PIPER_EXECUTABLE` 등의 기존 환경변수가 있으면 YAML보다 우선하므로 오래된 설정을 확인한다. YAML은 시작 시 읽으므로 실행 중이던 서버는 재시작해야 한다.

모델은 공식 `rhasspy/piper-voices`의 한국어 KSS medium이며 ONNX 63,221,984바이트, JSON 5,232바이트다. 다운로드한 저장소 revision은 `e7229f7d33a10e145e109b886d661984201ac3c4`이며 모델·설정·모델 카드의 URL과 SHA256은 `piper/voice_provenance.json`에 보관했다. Piper 엔진은 GPL-3.0-or-later이고 모델 카드의 학습 데이터 라이선스는 CC BY-NC-SA 4.0이다. 해당 모델 카드와 라이선스를 배포·사용 범위 검토에 함께 보관한다.

모델과 가상환경은 `.gitignore`에서 제외한다. 생성 WAV와 검증 결과도 기존 `outputs/` 제외 규칙을 따른다. 모델·의존성을 Git에 넣지 않는다.

### 서버 실행 및 요청

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
curl.exe http://127.0.0.1:8000/health
```

정상 설치 시 `tts_ready`는 `true`다. `/tts`는 `speaker_id`나 오디오 파일을 받지 않으며 **UTF-8 JSON의 text**만 받는다. ASR의 `personalized_text`를 이 필드에 넣으면 된다. 음성은 KSS 합성 화자의 목소리이며 KSM_M_45 음성 복제가 아니다.

Raspberry Pi 담당자는 코드 변경 없이 다음 HTTP 요청으로 먼저 확인할 수 있다. 아래는 Pi/Linux shell 명령이며 Windows PowerShell 인용 규칙과 다르다.

```bash
curl --fail --show-error http://SERVER_PC_IP:8000/health
curl --fail --show-error -X POST http://SERVER_PC_IP:8000/tts \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data '{"text":"안녕하세요. 이음 음성 안내입니다."}' \
  --output tts_test.wav
aplay tts_test.wav
```

성공 응답은 HTTP200, `Content-Type: audio/wav`, `Content-Disposition: attachment; filename="tts_<UUID>.wav"`다. WAV는 **22,050Hz, mono, PCM16**이다. ASR 입력의 16kHz와 다르므로 수신 파일을 임의로 16kHz로 해석하지 않는다. WAV 헤더에 따라 재생한다. 응답은 스트리밍 PCM이나 JSON/base64가 아니라 WAV 파일 자체다. 서버 임시 파일은 전송 완료 후 삭제되며, 클라이언트가 저장한 파일은 유지된다. 합성은 요청마다 Piper 프로세스와 음성 모델을 로딩하는 기존 방식이므로 고빈도 실시간 스트리밍 엔진은 아니다.

공백 입력은400, 입력 text 누락/2000자 초과는422, 환경 미설정은503, 합성 실패/시간 초과는500이다. 오류 시 JSON `detail`을 확인한다. 서버 합성 제한은60초이므로 클라이언트 HTTP 대기는 연결·전송 여유를 포함해70~90초로 설정할 수 있다. 기존 클라이언트의60초 설정은 이번 약2.3초 요청에서는 충분했으나 긴 문장에서는 별도 확인한다.

### 실제 검증 결과

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe scripts/08_demo/test_tts.py
```

검증 스크립트는 실제 두 ASR 모델을 로딩하는 Uvicorn 서버를 임의의 localhost 포트로 실행하고 직접 합성·HTTP 응답을 검사한 뒤 자신이 시작한 서버만 종료한다. 기존 실행 중 서버에는 연결하거나 종료 명령을 보내지 않는다. 기존 날짜의 검증 결과는 덮어쓰지 않는다.

테스트 문장: `안녕하세요. 이음 음성 안내입니다. 오늘 저녁에 약속 있어.`

| 경로 | 형식 | 샘플레이트 | 길이 | 크기 | RMS | 소요 시간 |
|---|---|---:|---:|---:|---:|---:|
| Piper 직접 합성 | mono PCM16 WAV | 22,050Hz | 5.085초 | 224,300바이트 | 0.173861 | 2.340초 |
| 실제 HTTP `/tts` | mono PCM16 WAV | 22,050Hz | 5.016초 | 221,228바이트 | 0.172264 | 2.267초 |

Piper의 확률적 합성으로 직접/API 음성 길이와 샘플이 완전히 같을 필요는 없다. 두 파일 모두 정상 디코딩·유한한 샘플·비무음을 확인했다. `/health`의 `tts_ready=true`, HTTP200 WAV 반환, 전송 후 임시 파일 정리, 공백 입력400을 통과했다. 검증 구간에 보호한 기존 연구/모델 파일41개의 SHA256과 ASR 환경 패키지 목록이 동일했다.

결과: `outputs/e1_demo/tts_verification_20261110/results.json`. 청취용 보존 파일은 같은 폴더의 `direct_korean.wav`, `api_korean.wav`다. 스피커 청취·한국어 발음 정확성의 주관 평가·Raspberry Pi 실제 재생·LAN 통신은 아직 검증하지 않았다. 브라우저/오디오 플레이어로 파일을 열어 청취하고 팀 장치에서도 위 요청을 확인한다. Git commit/push/merge 및 라즈베리파이 소스 변경은 하지 않았다.
