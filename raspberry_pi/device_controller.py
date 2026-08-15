from pathlib import Path
from typing import Optional

from raspberry_pi.api_client import ASRApiClient
from raspberry_pi.button import ButtonController
from raspberry_pi.recorder import AudioRecorder


class DeviceController:
    """
    IEUM Raspberry Pi 전체 동작을 제어한다.

    동작 흐름:

        녹음 버튼 1회
            ↓
        녹음 시작

        녹음 버튼 2회
            ↓
        녹음 종료
            ↓
        ASR
            ↓
        인식 결과 저장

        읽어주기 버튼
            ↓
        마지막 인식 결과 TTS
            ↓
        음성 출력
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        test_audio_path: str = "test_input_16k.wav",
        simulate_buttons: bool = True,
    ) -> None:

        self.api_client = ASRApiClient(
            server_url=server_url,
        )

        self.recorder = AudioRecorder(
            test_audio_path=test_audio_path,
        )

        self.current_text: Optional[str] = None

        self.is_recording = False

        self.tts_output_path = Path(
            "outputs/raspberry_pi_tts.wav"
        )

        self.tts_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.buttons = ButtonController(
            record_callback=self.on_record_button,
            tts_callback=self.on_tts_button,
            simulate=simulate_buttons,
        )

    # ========================================================
    # 녹음 버튼
    # ========================================================

    def on_record_button(self) -> None:
        """
        녹음 버튼이 눌렸을 때 실행된다.

        1회:
            녹음 시작

        2회:
            녹음 종료 → ASR
        """

        if not self.is_recording:
            self._start_recording()

        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """
        녹음을 시작한다.
        """

        print()
        print("=" * 60)
        print("[DEVICE] 녹음 시작")
        print("=" * 60)

        self.recorder.start_recording()

        self.is_recording = True

    def _stop_recording(self) -> None:
        """
        녹음을 종료하고 ASR을 실행한다.
        """

        print()
        print("=" * 60)
        print("[DEVICE] 녹음 종료")
        print("=" * 60)

        try:
            audio_path = self.recorder.stop_recording()

        except Exception as error:
            print(
                f"[ERROR] 녹음 종료 실패: {error}"
            )

            self.is_recording = False
            return

        self.is_recording = False

        print()
        print("[DEVICE] ASR 처리 시작")
        print(f"Audio: {audio_path}")

        text = self.api_client.transcribe(
            audio_path,
        )

        if text is None:
            print(
                "[ERROR] 음성 인식 결과가 없습니다."
            )
            return

        text = text.strip()

        if not text:
            print(
                "[ERROR] 인식된 텍스트가 비어 있습니다."
            )
            return

        self.current_text = text

        print()
        print("=" * 60)
        print("[DEVICE] 인식 결과")
        print("=" * 60)
        print(text)
        print("=" * 60)

        # 현재는 실제 디스플레이가 없으므로
        # 터미널에 자막을 표시한다.
        self._display_text(text)

    # ========================================================
    # 디스플레이
    # ========================================================

    def _display_text(
        self,
        text: str,
    ) -> None:
        """
        현재는 터미널을 디스플레이 대신 사용한다.

        실제 디스플레이가 연결되면
        이 부분을 LCD/OLED 코드로 교체한다.
        """

        print()
        print("[DISPLAY]")
        print(f"자막: {text}")
        print()

    # ========================================================
    # 읽어주기 버튼
    # ========================================================

    def on_tts_button(self) -> None:
        """
        읽어주기 버튼이 눌렸을 때 실행된다.
        """

        if not self.current_text:
            print(
                "[DEVICE] 읽어줄 자막이 없습니다."
            )
            return

        print()
        print("=" * 60)
        print("[DEVICE] TTS 시작")
        print("=" * 60)

        print(
            f"읽어줄 문장: {self.current_text}"
        )

        wav_path = self.api_client.synthesize(
            text=self.current_text,
            output_path=str(
                self.tts_output_path
            ),
        )

        if wav_path is None:
            print(
                "[ERROR] TTS 생성에 실패했습니다."
            )
            return

        print()
        print("[DEVICE] TTS 생성 완료")
        print(f"WAV: {wav_path}")

        self._play_audio(wav_path)

    # ========================================================
    # 음성 출력
    # ========================================================

    def _play_audio(
        self,
        audio_path: str,
    ) -> None:
        """
        생성된 TTS WAV를 재생한다.

        MacBook:
            afplay

        Raspberry Pi:
            aplay
        """

        import platform
        import subprocess

        system = platform.system()

        if system == "Darwin":
            command = [
                "afplay",
                audio_path,
            ]

        else:
            command = [
                "aplay",
                audio_path,
            ]

        print(
            f"[DEVICE] 음성 재생: {audio_path}"
        )

        try:
            subprocess.run(
                command,
                check=True,
            )

        except FileNotFoundError:
            print(
                f"[ERROR] 오디오 재생 프로그램을 "
                f"찾을 수 없습니다: {command[0]}"
            )

        except subprocess.CalledProcessError as error:
            print(
                f"[ERROR] 음성 재생 실패: {error}"
            )

    # ========================================================
    # MacBook 테스트
    # ========================================================

    def run_simulation(self) -> None:
        """
        MacBook에서 Raspberry Pi 버튼을
        가상으로 테스트한다.

        순서:

            녹음 버튼
            → 녹음 시작

            녹음 버튼
            → 녹음 종료 + ASR

            읽어주기 버튼
            → TTS + 음성 출력
        """

        print("=" * 60)
        print("IEUM Raspberry Pi Device Simulation")
        print("=" * 60)

        print()
        print("[1] FastAPI 서버 확인")

        if not self.api_client.health_check():
            print(
                "[ERROR] FastAPI server is not available."
            )
            return

        print(
            "FastAPI server is running."
        )

        print()
        print("[2] 녹음 버튼 1회")

        self.buttons.simulate_record_button()

        print()
        print("[3] 녹음 버튼 2회")

        self.buttons.simulate_record_button()

        print()
        print("[4] 읽어주기 버튼 1회")

        self.buttons.simulate_tts_button()

        print()
        print("=" * 60)
        print("IEUM Device Simulation 완료")
        print("=" * 60)


if __name__ == "__main__":

    controller = DeviceController(
        server_url="http://127.0.0.1:8000",
        test_audio_path="test_input_16k.wav",
        simulate_buttons=True,
    )

    controller.run_simulation()