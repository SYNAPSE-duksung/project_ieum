from pathlib import Path
from typing import Optional

import requests


class ASRApiClient:
    """
    Raspberry Pi에서 IEUM FastAPI 서버와 통신하는 클라이언트.

    현재 지원:
    1. FastAPI 서버 상태 확인
    2. 음성 파일 → ASR → 텍스트
    3. 텍스트 → TTS → WAV 파일
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
        FastAPI 서버 상태 확인.
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
        음성 인식 결과를 반환한다.
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
                print(
                    f"[ERROR] ASR API request failed: {error}"
                )
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
            음성으로 변환할 한국어 텍스트

        output_path:
            생성된 WAV 파일을 저장할 경로

        Returns
        -------
        Optional[str]
            저장된 WAV 파일 경로
        """

        if not text or not text.strip():
            raise ValueError(
                "TTS text is empty."
            )

        output = Path(output_path)

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
            print(
                f"[ERROR] TTS API request failed: {error}"
            )
            return None

        try:
            output.write_bytes(
                response.content
            )

        except OSError as error:
            print(
                f"[ERROR] Failed to save TTS WAV: {error}"
            )
            return None

        if not output.exists():
            print(
                "[ERROR] TTS WAV file was not created."
            )
            return None

        return str(output)

    # ========================================================
    # Test
    # ========================================================

    def test_tts(
        self,
        text: str = "안녕하세요. 이음 보조장치입니다.",
        output_path: str = "raspberry_pi_tts.wav",
    ) -> None:
        """
        TTS API 연결 테스트.
        """

        print("=" * 60)
        print("IEUM Raspberry Pi TTS API Test")
        print("=" * 60)

        print()
        print("[1] FastAPI 서버 확인")

        if not self.health_check():
            print(
                "[ERROR] FastAPI server is not available."
            )
            return

        print(
            "FastAPI server is running."
        )

        print()
        print("[2] TTS 요청")

        print(f"입력 텍스트: {text}")

        wav_path = self.synthesize(
            text=text,
            output_path=output_path,
        )

        if wav_path is None:
            print(
                "[ERROR] TTS 음성 생성에 실패했습니다."
            )
            return

        print()
        print("TTS 파일 생성 완료:")
        print(f"  {wav_path}")

        print()
        print("=" * 60)
        print("TTS API Test 완료")
        print("=" * 60)


if __name__ == "__main__":

    client = ASRApiClient()

    client.test_tts()