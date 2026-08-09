from api_client import ASRApiClient
from record_audio import record_audio


SERVER_URL = "http://127.0.0.1:8000"
AUDIO_PATH = "recorded_audio.wav"
RECORD_DURATION = 30.0


def main():
    """
    Raspberry Pi 음성인식 클라이언트 실행

    1. FastAPI 서버 연결 확인
    2. 마이크 음성 녹음
    3. WAV 파일을 서버로 전송
    4. 음성인식 결과 출력
    """

    client = ASRApiClient(
        server_url=SERVER_URL,
    )

    # 1. 서버 연결 확인
    print("Checking FastAPI server...")

    if not client.health_check():
        print("[ERROR] FastAPI server is not available.")
        return

    print("FastAPI server is running.")

    # 2. 음성 녹음
    audio_path = record_audio(
        output_path=AUDIO_PATH,
        duration=RECORD_DURATION,
    )

    # 3. 서버로 전송
    print("Sending audio to ASR server...")

    text = client.transcribe(audio_path)

    # 4. 결과 출력
    if text is None:
        print("[ERROR] Transcription failed.")
        return

    print()
    print("===== ASR Result =====")
    print(text)
    print("======================")


if __name__ == "__main__":
    main()