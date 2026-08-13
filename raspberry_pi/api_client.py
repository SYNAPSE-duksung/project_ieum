from pathlib import Path
from typing import Optional

import requests


class ASRApiClient:
    """
    Raspberry Pi에서 IEUM FastAPI 서버로
    음성 파일을 전송하고,
    TTS 음성을 요청하는 클라이언트
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        timeout: int = 60,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    # ========================================================
    # Health Check
    # ========================================================

    def health_check(self) -> bool:
        """
        FastAPI 서버 상태 확인
        """

        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    # ========================================================
    # ASR
    # ========================================================

    def transcribe(
        self,
        audio_path: str,
    ) -> Optional[str]:
        """
        WAV 파일을 FastAPI 서버로 전송하고
        음성 인식 결과를 반환
        """

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )

        with path.open("rb") as audio_file:
            files = {
                "audio": (
                    path.name,
                    audio_file,
                    "audio/wav",
                )
            }

            try:
                response = requests.post(
                    f"{self.server_url}/transcribe",
                    files=files,
                    timeout=self.timeout,
                )

                response.raise_for_status()

            except requests.RequestException as error:
                print(f"[ERROR] ASR API request failed: {error}")
                return None

        result = response.json()

        return result.get("text")

    # ========================================================
    # TTS
    # ========================================================

    def synthesize(
        self,
        text: str,
        output_path: str = "tts_output.wav",
    ) -> Optional[str]:
        """
        텍스트를 FastAPI 서버로 전송하고
        Piper TTS가 생성한 WAV 파일을 저장한다.

        Parameters
        ----------
        text:
            음성으로 변환할 텍스트

        output_path:
            생성된 WAV 파일을 저장할 경로

        Returns
        -------
        저장된 WAV 파일 경로
        """

        if not text or not text.strip():
            raise ValueError(
                "TTS text is empty."
            )

        path = Path(output_path)

        try:
            response = requests.post(
                f"{self.server_url}/tts",
                json={
                    "text": text,
                },
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as error:
            print(f"[ERROR] TTS API request failed: {error}")
            return None

        path.write_bytes(response.content)

        return str(path)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    client = ASRApiClient(
        server_url="http://127.0.0.1:8000"
    )

    print("=" * 60)
    print("IEUM API Client 테스트")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. 서버 상태 확인
    # --------------------------------------------------------

    if client.health_check():
        print("FastAPI server is running.")
    else:
        print("FastAPI server is not available.")
        exit(1)

    # --------------------------------------------------------
    # 2. TTS 테스트
    # --------------------------------------------------------

    text = "안녕하세요. IEUM Piper TTS 테스트입니다."

    print()
    print(f"입력 텍스트: {text}")

    output = client.synthesize(
        text=text,
        output_path="tts_client_test.wav",
    )

    if output:
        print(f"TTS 파일 생성 완료: {output}")
    else:
        print("TTS 파일 생성 실패.")

    print("=" * 60)