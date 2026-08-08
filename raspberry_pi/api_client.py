from pathlib import Path
from typing import Optional

import requests


class ASRApiClient:
    """
    Raspberry Pi에서 IEUM FastAPI 서버로
    음성 파일을 전송하는 클라이언트
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        timeout: int = 60,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

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
                print(f"[ERROR] API request failed: {error}")
                return None

        result = response.json()

        return result.get("text")


if __name__ == "__main__":
    client = ASRApiClient()

    if client.health_check():
        print("FastAPI server is running.")
    else:
        print("FastAPI server is not available.")