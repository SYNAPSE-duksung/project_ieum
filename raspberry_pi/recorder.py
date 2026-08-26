from pathlib import Path
import signal
import subprocess
import wave
from typing import Optional


class AudioRecorder:
    """
    Raspberry Pi에 연결된 USB 마이크로 실제 음성을 녹음한다.

    녹음 버튼 1회:
        arecord 실행 → 녹음 시작

    녹음 버튼 2회:
        arecord 종료 → WAV 파일 저장
    """

    def __init__(
        self,
        output_path: str = "outputs/recorded_input.wav",
        audio_device: str = "plughw:3,0",
    ) -> None:

        self.output_path = Path(output_path)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.audio_device = audio_device

        self.process: Optional[subprocess.Popen] = None
        self.is_recording = False

    
    def start_recording(self) -> None:
        """
        USB 마이크 녹음을 시작한다.
        """

        if self.is_recording:
            print("[RECORDER] 이미 녹음 중입니다.")
            return

        # 이전 녹음 파일 제거
        if self.output_path.exists():
            self.output_path.unlink()

        print("[RECORDER] 녹음 시작")
        print(f"[RECORDER] Device: {self.audio_device}")
        print(f"[RECORDER] Output: {self.output_path}")

        try:
            self.process = subprocess.Popen(
            [
                "arecord",
                "-D",
                self.audio_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-d",
                "29",
                str(self.output_path),
            ]
        )

        except FileNotFoundError as error:
            raise RuntimeError(
                "arecord 명령을 찾을 수 없습니다."
            ) from error

        except Exception as error:
            raise RuntimeError(
                f"녹음을 시작하지 못했습니다: {error}"
            ) from error

        self.is_recording = True

    def _repair_wav_header(self) -> None:
        """
        arecord가 SIGINT로 종료될 때 남을 수 있는
        잘못된 WAV 길이 정보를 실제 PCM 데이터 크기에 맞게 복구한다.
        """

        if not self.output_path.exists():
            return

        raw = self.output_path.read_bytes()

        if len(raw) <= 44:
            raise RuntimeError(
                "녹음된 WAV 파일에 유효한 오디오 데이터가 없습니다."
            )

        # arecord가 생성한 표준 WAV 헤더 44바이트 뒤의
        # PCM16 오디오 데이터만 가져온다.
        pcm_data = raw[44:]

        temp_path = self.output_path.with_suffix(".fixed.wav")

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(pcm_data)

        temp_path.replace(self.output_path)


    def stop_recording(self) -> str:
        """
        녹음을 종료하고 생성된 WAV 파일 경로를 반환한다.
        """

        if not self.is_recording:
            raise RuntimeError(
                "현재 녹음 중이 아닙니다."
            )

        if self.process is None:
            raise RuntimeError(
                "녹음 프로세스를 찾을 수 없습니다."
            )

        print("[RECORDER] 녹음 종료")

        try:
            # 아직 arecord가 실행 중이면 버튼 입력으로 종료
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5)
            else:
                # -d 29에 의해 이미 자동 종료된 경우
                self.process.wait()

        except subprocess.TimeoutExpired:
            print(
                "[RECORDER] 정상 종료되지 않아 강제로 종료합니다."
            )
            self.process.terminate()
            self.process.wait()

        finally:
            self.process = None
            self.is_recording = False

        self._repair_wav_header()

        if not self.output_path.exists():
            raise FileNotFoundError(
                f"녹음 파일을 찾을 수 없습니다: "
                f"{self.output_path}"
            )

        if self.output_path.stat().st_size == 0:
            raise RuntimeError(
                "녹음된 WAV 파일이 비어 있습니다."
            )

        print(
            f"[RECORDER] 녹음 파일 생성 완료: "
            f"{self.output_path}"
        )

        return str(self.output_path)