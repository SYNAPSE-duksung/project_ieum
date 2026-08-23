from pathlib import Path
from typing import Optional

import requests


class ASRApiClient:
    """
    Raspberry Pi에서 IEUM FastAPI 서버와 통신하는 클라이언트.

    HTTP 요청과 응답 처리만 담당한다.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        timeout: int = 60,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        """FastAPI 서버 상태를 확인한다."""

        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=self.timeout,
            )

            return response.status_code == 200

        except requests.RequestException:
            return False

    def transcribe(
        self,
        audio_path: str,
    ) -> Optional[str]:
        """
        음성 파일을 /transcribe로 전송하고 인식된 text를 반환한다.

        기존 raspberry_pi/run.py와의 호환성을 위해 요청 실패 시
        None을 반환한다.
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

    def transcribe_compare(
        self,
        audio_path: str | Path,
        speaker_id: str,
    ) -> dict:
        """
        동일 음성을 범용 및 개인화 ASR로 비교하고 전체 결과를 반환한다.
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
                    f"{self.server_url}/transcribe/compare",
                    files=files,
                    data={"speaker_id": speaker_id},
                    timeout=self.timeout,
                )

                response.raise_for_status()

            except requests.RequestException as error:
                raise RuntimeError(
                    "ASR comparison API request failed: "
                    f"{error}"
                ) from error

        result = response.json()

        if not isinstance(result, dict):
            raise RuntimeError(
                "ASR comparison API returned an invalid JSON response."
            )

        return result

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
    ) -> str:
        """텍스트를 /tts로 전송하고 반환된 WAV 파일을 저장한다."""

        if not text or not text.strip():
            raise ValueError("TTS text is empty.")

        output = Path(output_path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            response = requests.post(
                f"{self.server_url}/tts",
                json={"text": text},
                timeout=self.timeout,
            )

            response.raise_for_status()

        except requests.RequestException as error:
            raise RuntimeError(
                f"TTS API request failed: {error}"
            ) from error

        try:
            output.write_bytes(response.content)

        except OSError as error:
            raise RuntimeError(
                f"Failed to save TTS WAV to {output}: {error}"
            ) from error

        return str(output)


if __name__ == "__main__":
    client = ASRApiClient()

    if client.health_check():
        print("FastAPI server is running.")
    else:
        print("FastAPI server is not available.")
