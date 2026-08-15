from pathlib import Path
from typing import Optional


class AudioRecorder:
    """
    음성 녹음을 담당하는 클래스.

    현재 MacBook 테스트에서는
    미리 준비된 WAV 파일을 녹음 결과로 사용한다.

    실제 Raspberry Pi에서는 이 클래스를
    블루투스 마이크 녹음 방식으로 교체한다.
    """

    def __init__(
        self,
        test_audio_path: Optional[str] = None,
    ) -> None:

        self.test_audio_path = (
            Path(test_audio_path)
            if test_audio_path
            else None
        )

        self.is_recording = False

    def start_recording(self) -> None:
        """
        녹음을 시작한다.
        """

        if self.is_recording:
            print("[RECORDER] 이미 녹음 중입니다.")
            return

        self.is_recording = True

        print("[RECORDER] 녹음 시작")

    def stop_recording(self) -> str:
        """
        녹음을 종료하고
        녹음된 WAV 파일 경로를 반환한다.

        현재 MacBook 테스트에서는
        test_audio_path에 지정된 WAV 파일을 반환한다.
        """

        if not self.is_recording:
            raise RuntimeError(
                "현재 녹음 중이 아닙니다."
            )

        self.is_recording = False

        print("[RECORDER] 녹음 종료")

        if self.test_audio_path is None:
            raise RuntimeError(
                "현재 테스트용 오디오 파일이 지정되지 않았습니다."
            )

        if not self.test_audio_path.exists():
            raise FileNotFoundError(
                f"테스트 음성 파일을 찾을 수 없습니다: "
                f"{self.test_audio_path}"
            )

        return str(self.test_audio_path)