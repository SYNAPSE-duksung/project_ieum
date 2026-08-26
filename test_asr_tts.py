from pathlib import Path

from raspberry_pi.api_client import ASRApiClient


# ============================================================
# 설정
# ============================================================

SERVER_URL = "http://127.0.0.1:8000"

INPUT_AUDIO = "test_input_16k.wav"
OUTPUT_AUDIO = "asr_tts_output.wav"


# ============================================================
# ASR → TTS 전체 테스트
# ============================================================

def main():

    client = ASRApiClient(
        server_url=SERVER_URL,
        timeout=60,
    )

    print("=" * 70)
    print("IEUM ASR → TTS 전체 테스트")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 서버 확인
    # --------------------------------------------------------

    print()
    print("[1] FastAPI 서버 확인")

    if not client.health_check():
        print("FastAPI server is not available.")
        return

    print("FastAPI server is running.")

    # --------------------------------------------------------
    # 2. 입력 음성 확인
    # --------------------------------------------------------

    input_path = Path(INPUT_AUDIO)

    if not input_path.exists():
        print()
        print(f"[ERROR] 음성 파일을 찾을 수 없습니다: {input_path}")
        return

    print()
    print(f"[2] 입력 음성: {input_path}")

    # --------------------------------------------------------
    # 3. ASR
    # --------------------------------------------------------

    print()
    print("[3] ASR 음성 인식 시작")

    text = client.transcribe(
        audio_path=str(input_path)
    )

    if not text:
        print("[ERROR] 음성 인식 결과가 없습니다.")
        return

    print()
    print("인식 결과:")
    print(f"  {text}")

    # --------------------------------------------------------
    # 4. TTS
    # --------------------------------------------------------

    print()
    print("[4] Piper TTS 생성 시작")

    output_path = client.synthesize(
        text=text,
        output_path=OUTPUT_AUDIO,
    )

    if not output_path:
        print("[ERROR] TTS 생성 실패.")
        return

    print()
    print(f"TTS 파일 생성 완료:")
    print(f"  {output_path}")

    # --------------------------------------------------------
    # 5. 완료
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ASR → TTS 전체 테스트 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()