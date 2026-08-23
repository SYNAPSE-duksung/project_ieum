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
    실제 USB 마이크 녹음 시작

    녹음 버튼 2회
        ↓
    녹음 종료
        ↓
    WAV 저장
        ↓
    FastAPI /transcribe
        ↓
    ASR 결과 저장

    읽어주기 버튼
        ↓
    현재 ASR 결과
        ↓
    FastAPI /tts
    """

    def __init__(
        self,
        server_url: str = "http://192.168.137.1:8000",
        simulate_buttons: bool = False,
    ) -> None:

        self.api_client = ASRApiClient(
            server_url=server_url,
        )

        self.recorder = AudioRecorder(
            output_path="outputs/recorded_input.wav",
            audio_device="plughw:3,0",
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
            record_pin=17,
            tts_pin=27,
            simulate=simulate_buttons,
        )

    # ========================================================
    # 녹음 버튼
    # ========================================================

    def on_record_button(self) -> None:
        """
        녹음 버튼이 눌렸을 때 실행된다.

        첫 번째:
            녹음 시작

        두 번째:
            녹음 종료 → ASR
        """

        if not self.is_recording:
            self._start_recording()

        else:
            self._stop_recording()

    def _start_recording(self) -> None:

        print()
        print("=" * 60)
        print("[DEVICE] 녹음 시작")
        print("=" * 60)

        try:
            self.recorder.start_recording()

        except Exception as error:
            print(
                f"[ERROR] 녹음 시작 실패: {error}"
            )
            self.is_recording = False
            return

        self.is_recording = True

    def _stop_recording(self) -> None:

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
        print(f"[DEVICE] Audio: {audio_path}")

        try:
            text = self.api_client.transcribe(
                audio_path
            )

        except Exception as error:
            print(
                f"[ERROR] ASR 요청 실패: {error}"
            )
            return

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
        print("[DEVICE] ASR 인식 결과")
        print("=" * 60)
        print(text)
        print("=" * 60)

        self._display_text(text)

    # ========================================================
    # 디스플레이
    # ========================================================

    def _display_text(
        self,
        text: str,
    ) -> None:

        # 현재는 터미널 출력.
        # 추후 노트북 UI와 연결 가능.
        print()
        print("[DISPLAY]")
        print(f"자막: {text}")
        print()

    # ========================================================
    # 읽어주기 버튼
    # ========================================================

    def on_tts_button(self) -> None:

        if not self.current_text:
            print(
                "[DEVICE] 읽어줄 ASR 결과가 없습니다."
            )
            return

        print()
        print("=" * 60)
        print("[DEVICE] TTS 요청")
        print("=" * 60)
        print(
            f"[DEVICE] 읽어줄 문장: "
            f"{self.current_text}"
        )

        try:
            wav_path = self.api_client.synthesize(
                text=self.current_text,
                output_path=str(
                    self.tts_output_path
                ),
            )

        except Exception as error:
            print(
                f"[ERROR] TTS 요청 실패: {error}"
            )
            return

        if wav_path is None:
            print(
                "[ERROR] TTS 생성에 실패했습니다."
            )
            return

        print()
        print("[DEVICE] TTS 생성 완료")
        print(f"[DEVICE] WAV: {wav_path}")

        # 현재 최종 시연에서는
        # Raspberry Pi 스피커를 사용하지 않으므로
        # Raspberry Pi에서 aplay하지 않는다.
        #
        # 노트북 UI에서 TTS를 재생하도록
        # FastAPI/UI 부분을 별도로 연결해야 한다.


if __name__ == "__main__":
    from signal import pause

    controller = DeviceController(
        server_url="http://192.168.137.1:8000",
        simulate_buttons=False,
    )

    print("=" * 60)
    print("IEUM Raspberry Pi Controller")
    print("=" * 60)

    print()
    print("FastAPI:")
    print("http://192.168.137.1:8000")

    print()
    print("USB Mic:")
    print("plughw:3,0")

    print()
    print("녹음 버튼     : GPIO 17")
    print("읽어주기 버튼 : GPIO 27")

    print()
    print("버튼 입력 대기 중...")
    print("종료: Ctrl+C")

    try:
        pause()

    except KeyboardInterrupt:
        print()
        print("IEUM Controller 종료")