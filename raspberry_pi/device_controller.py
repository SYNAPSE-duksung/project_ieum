from __future__ import annotations

import requests

from pathlib import Path
from typing import Optional

from raspberry_pi.api_client import ASRApiClient
from raspberry_pi.button import ButtonController
from raspberry_pi.recorder import AudioRecorder


SUPPORTED_SPEAKERS = (
    "HYH_M_22",
    "SKY_M_24",
)


class DeviceController:
    """IEUM Raspberry Pi의 녹음, ASR, TTS 실행 흐름을 조정한다."""

    def __init__(
        self,
        server_url: str = "http://192.168.137.1:8000",
        speaker_id: str = "HYH_M_22",
        simulate_buttons: bool = True,
        tts_output_path: str | Path = "outputs/raspberry_pi_tts.wav",
    ) -> None:
        if speaker_id not in SUPPORTED_SPEAKERS:
            raise ValueError(
                f"Unsupported speaker_id: {speaker_id}. "
                f"Supported speakers: {list(SUPPORTED_SPEAKERS)}"
            )

        self.speaker_id = speaker_id

        self.api_client = ASRApiClient(
            server_url=server_url,
        )

        self.recorder = AudioRecorder(
            output_path="outputs/recorded_input.wav",
            audio_device="plughw:3,0",
        )

        self.current_audio_path: Optional[str] = None
        self.current_general_text: Optional[str] = None
        self.current_personalized_text: Optional[str] = None
        self.current_tts_text: Optional[str] = None

        # recorder가 30초에 자동 종료되더라도 다음 버튼 입력에서
        # stop_recording()으로 저장 경로를 회수해야 함을 기억한다.
        self._recording_cycle_pending = False

        self.tts_output_path = Path(tts_output_path)

        self.buttons = ButtonController(
            record_callback=self.on_record_button,
            tts_callback=self.on_tts_button,
            record_pin=18,
            tts_pin=27,
            simulate=simulate_buttons,
        )

    def on_record_button(self) -> None:
        """첫 입력은 녹음을 시작하고 다음 입력은 종료 후 ASR을 수행한다."""

        if self.recorder.is_recording:
            self._stop_recording_and_transcribe()
        elif self._recording_cycle_pending:
            # 최대 30초 자동 종료 후 들어온 다음 버튼 입력이다.
            self._stop_recording_and_transcribe()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        print()
        print("=" * 60)
        print("[DEVICE] 녹음 시작")
        print("=" * 60)

        try:
            self.recorder.start_recording()

        except Exception as error:
            self._recording_cycle_pending = False
            print(f"[ERROR] 녹음 시작 실패: {error}")
            return

        self._clear_current_result()
        self._recording_cycle_pending = True

        self._update_ui(
            status="recording",
            general_text="",
            personalized_text="",
        )

        print(
            "[DEVICE] 녹음 중입니다. 같은 버튼을 다시 누르면 종료됩니다."
        )
        print("[DEVICE] 최대 30초 후에는 자동으로 저장됩니다.")

    def _stop_recording_and_transcribe(self) -> None:
        print()
        print("=" * 60)
        print("[DEVICE] 녹음 종료")
        print("=" * 60)

        try:
            audio_path = self.recorder.stop_recording()

        except Exception as error:
            self._recording_cycle_pending = False
            print(f"[ERROR] 녹음 종료 실패: {error}")
            return

        self._recording_cycle_pending = False
        self.current_audio_path = audio_path

        print(f"[DEVICE] 저장된 음성: {audio_path}")
        self._transcribe_compare(audio_path)

    def _transcribe_compare(
        self,
        audio_path: str,
    ) -> None:
        print()
        print("[DEVICE] 범용 / 개인화 ASR 처리 시작")
        print(f"[DEVICE] Speaker: {self.speaker_id}")

        self._update_ui(
            status="processing",
        )

        try:
            result = self.api_client.transcribe_compare(
                audio_path=audio_path,
                speaker_id=self.speaker_id,
            )

            if (
                "general_text" not in result
                or "personalized_text" not in result
            ):
                raise RuntimeError(
                    "ASR comparison response is missing required fields."
                )

            general_text = str(
                result.get("general_text") or ""
            ).strip()
            personalized_text = str(
                result.get("personalized_text") or ""
            ).strip()

        except Exception as error:
            self.current_general_text = None
            self.current_personalized_text = None
            self.current_tts_text = None
            print(f"[ERROR] 범용 / 개인화 ASR 요청 실패: {error}")
            return

        self.current_general_text = general_text
        self.current_personalized_text = personalized_text
        self.current_tts_text = personalized_text or general_text or None

        self._update_ui(
            status="complete",
            general_text=general_text,
            personalized_text=personalized_text,
        )

        self._display_transcription_results()

        if self.current_tts_text is None:
            print("[ERROR] 범용 및 개인화 인식 결과가 모두 비어 있습니다.")
            return

        selected = "Personalized" if personalized_text else "General"
        print(f"[DEVICE] TTS 선택 결과: {selected}")

    def _display_transcription_results(self) -> None:
        print()
        print("=" * 60)
        print("[General]")
        print(self.current_general_text or "(empty)")
        print()
        print("[Personalized]")
        print(self.current_personalized_text or "(empty)")
        print("=" * 60)

    def on_tts_button(self) -> None:
        """선택된 ASR 텍스트를 TTS로 합성하고 재생한다."""

        if not self.current_tts_text:
            print("[DEVICE] 먼저 음성을 인식하세요.")
            return

        print()
        print("=" * 60)
        print("[DEVICE] TTS 시작")
        print("=" * 60)
        print(f"읽어줄 문장: {self.current_tts_text}")

        try:
            wav_path = self.api_client.synthesize(
                text=self.current_tts_text,
                output_path=self.tts_output_path,
            )

        except Exception as error:
            print(f"[ERROR] TTS 요청 실패: {error}")
            return

        print(f"[DEVICE] TTS 파일: {wav_path}")
        self._play_audio(wav_path)

    def _play_audio(
        self,
        audio_path: str | Path,
    ) -> None:
        """macOS에서는 afplay, Raspberry Pi에서는 pw-play로 재생한다."""

        import platform
        import subprocess
        import wave

        path = Path(audio_path)

        if not path.exists():
            print(f"[ERROR] 재생할 WAV 파일이 없습니다: {path}")
            return

        play_path = path

        if platform.system() != "Darwin":
            try:
                with wave.open(str(path), "rb") as src:
                    params = src.getparams()
                    frames = src.readframes(src.getnframes())

                silence_seconds = 0.7
                silence_frames = int(
                    params.framerate * silence_seconds
                )

                silence = (
                    b"\x00"
                    * silence_frames
                    * params.nchannels
                    * params.sampwidth
                )

                padded_path = path.with_name(
                    path.stem + "_padded.wav"
                )

                with wave.open(str(padded_path), "wb") as dst:
                    dst.setparams(params)
                    dst.writeframes(silence + frames)

                play_path = padded_path

            except Exception as error:
                print(f"[WARN] 앞 무음 추가 실패: {error}")
                play_path = path

        if platform.system() == "Darwin":
            command = ["afplay", str(play_path)]
        else:
            command = ["pw-play", str(play_path)]

        print(f"[DEVICE] 음성 재생: {play_path}")

        try:
            subprocess.run(
                command,
                check=True,
            )

        except FileNotFoundError:
            print(
                f"[ERROR] 오디오 재생 프로그램을 찾을 수 없습니다: "
                f"{command[0]}"
            )

        except subprocess.CalledProcessError as error:
            print(f"[ERROR] 음성 재생 실패: {error}")

        except OSError as error:
            print(f"[ERROR] 오디오 재생 실행 실패: {error}")


    def _clear_current_result(self) -> None:
        self.current_audio_path = None
        self.current_general_text = None
        self.current_personalized_text = None
        self.current_tts_text = None

    def run_simulation(self) -> None:
        """fixture WAV와 simulated button callback으로 전체 흐름을 시험한다."""

        if not self.recorder.simulation_mode:
            raise RuntimeError(
                "run_simulation() requires test_audio_path."
            )

        print("=" * 60)
        print("IEUM Raspberry Pi Device Simulation")
        print("=" * 60)

        if not self.api_client.health_check():
            print("[ERROR] FastAPI server is not available.")
            return

        print("[1] 녹음 버튼: simulation 녹음 시작")
        self.buttons.simulate_record_button()

        print("[2] 녹음 버튼: fixture WAV로 ASR 비교")
        self.buttons.simulate_record_button()

        print("[3] TTS 버튼: 선택 결과 합성 및 재생")
        self.buttons.simulate_tts_button()

    def _update_ui(
        self,
        status: str,
        general_text: str | None = None,
        personalized_text: str | None = None,
    ) -> None:
        payload = {
            "status": status,
        }

        if general_text is not None:
            payload["general_text"] = general_text

        if personalized_text is not None:
            payload["personalized_text"] = personalized_text

        try:
            requests.post(
                "http://127.0.0.1:5000/api/device/state",
                json=payload,
                timeout=2,
            )
        except requests.RequestException as error:
            print(f"[UI] 상태 전달 실패: {error}")

if __name__ == "__main__":
    from signal import pause

    controller = DeviceController(
        server_url="http://192.168.137.1:8000",
        speaker_id="HYH_M_22",
        simulate_buttons=False,
    )

    print("=" * 60)
    print("IEUM Raspberry Pi Controller")
    print("=" * 60)
    print(f"개인화 화자   : {controller.speaker_id}")
    print("녹음 버튼     : GPIO 18")
    print("읽어주기 버튼 : GPIO 27")
    print("버튼 입력 대기 중...")
    print("종료: Ctrl+C")

    try:
        pause()

    except KeyboardInterrupt:
        print()
        print("IEUM Controller 종료")
