from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Timer, current_thread
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


RECORD_SAMPLE_RATE = 44_100
OUTPUT_SAMPLE_RATE = 16_000
CHANNELS = 1
MAX_DURATION = 30.0
MIN_DURATION = 0.1


class AudioRecorder:
    """
    Raspberry Pi USB 마이크용 streaming recorder.

    test_audio_path가 지정되면 실제 마이크 대신 준비된 WAV 파일을
    사용하는 simulation mode로 동작한다.
    """

    def __init__(
        self,
        output_path: str | Path = "recorded_audio.wav",
        test_audio_path: Optional[str | Path] = None,
        device: int | str | None = 1,
        max_duration: float = MAX_DURATION,
        min_duration: float = MIN_DURATION,
    ) -> None:
        if max_duration <= 0 or max_duration > MAX_DURATION:
            raise ValueError(
                f"max_duration must be between 0 and {MAX_DURATION}."
            )

        if min_duration <= 0 or min_duration > max_duration:
            raise ValueError(
                "min_duration must be greater than 0 and no greater "
                "than max_duration."
            )

        self.output_path = Path(output_path)
        self.test_audio_path = (
            Path(test_audio_path)
            if test_audio_path is not None
            else None
        )
        self.device = device
        self.max_duration = float(max_duration)
        self.min_duration = float(min_duration)

        self._lock = Lock()
        self._finalized = Event()
        self._finalized.set()

        self._stream: sd.InputStream | None = None
        self._timer: Timer | None = None
        self._chunks: list[np.ndarray] = []
        self._recorded_frames = 0
        self._is_recording = False
        self._is_finalizing = False
        self._last_output_path: str | None = None
        self._finalize_error: BaseException | None = None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._is_recording

    @property
    def simulation_mode(self) -> bool:
        return self.test_audio_path is not None

    def start_recording(self) -> None:
        """실제 input stream 또는 simulation recording을 시작한다."""

        with self._lock:
            if self._is_recording or self._is_finalizing:
                raise RuntimeError("이미 녹음 중이거나 종료 처리 중입니다.")

            self._chunks = []
            self._recorded_frames = 0
            self._last_output_path = None
            self._finalize_error = None
            self._is_recording = True
            self._finalized.clear()

        stream: sd.InputStream | None = None

        try:
            if not self.simulation_mode:
                stream = sd.InputStream(
                    samplerate=RECORD_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    device=self.device,
                    callback=self._audio_callback,
                )
                stream.start()

                with self._lock:
                    self._stream = stream

            timer = Timer(
                self.max_duration,
                self._auto_stop,
            )
            timer.daemon = True

            with self._lock:
                self._timer = timer

            timer.start()

        except Exception:
            if stream is not None:
                stream.close()

            with self._lock:
                self._is_recording = False
                self._stream = None
                self._timer = None
                self._finalized.set()
            raise

        mode = "simulation" if self.simulation_mode else "USB microphone"
        print(f"[RECORDER] 녹음 시작 ({mode}, 44.1 kHz mono)")

    def stop_recording(self) -> str:
        """
        녹음을 종료하고 16 kHz mono PCM16 WAV 파일 경로를 반환한다.

        최대 시간에 의해 이미 자동 종료된 경우에는 자동 저장된 파일
        경로를 반환한다.
        """

        should_finalize = False

        with self._lock:
            if self._is_recording:
                self._is_recording = False
                self._is_finalizing = True
                should_finalize = True
            elif self._is_finalizing:
                pass
            elif self._last_output_path is not None:
                return self._last_output_path
            elif self._finalize_error is not None:
                raise RuntimeError(
                    f"녹음 종료 처리에 실패했습니다: {self._finalize_error}"
                ) from self._finalize_error
            else:
                raise RuntimeError("현재 녹음 중이 아닙니다.")

        if should_finalize:
            self._finalize_recording()
        else:
            self._finalized.wait()

        with self._lock:
            if self._finalize_error is not None:
                raise RuntimeError(
                    f"녹음 종료 처리에 실패했습니다: {self._finalize_error}"
                ) from self._finalize_error

            if self._last_output_path is None:
                raise RuntimeError("녹음 파일이 생성되지 않았습니다.")

            return self._last_output_path

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status: sd.CallbackFlags,
    ) -> None:
        """InputStream callback: audio chunk 복사와 누적만 수행한다."""

        with self._lock:
            if not self._is_recording:
                return

            max_frames = int(
                self.max_duration * RECORD_SAMPLE_RATE
            )
            remaining_frames = max_frames - self._recorded_frames

            if remaining_frames <= 0:
                return

            chunk = indata[:remaining_frames].copy()
            self._chunks.append(chunk)
            self._recorded_frames += int(chunk.shape[0])

    def _auto_stop(self) -> None:
        """Timer thread에서 최대 녹음 시간 도달 시 종료·저장한다."""

        with self._lock:
            if not self._is_recording:
                return

            self._is_recording = False
            self._is_finalizing = True

        print(f"[RECORDER] 최대 {self.max_duration:.1f}초 도달, 자동 종료")
        self._finalize_recording()

    def _finalize_recording(self) -> None:
        """Stream을 닫고 callback 밖에서 resampling과 저장을 수행한다."""

        with self._lock:
            stream = self._stream
            timer = self._timer
            self._stream = None
            self._timer = None

        if timer is not None and timer is not current_thread():
            timer.cancel()

        try:
            if stream is not None:
                stream.stop()
                stream.close()

            if self.simulation_mode:
                saved_path = self._finalize_simulation()
            else:
                saved_path = self._resample_and_save()

            with self._lock:
                self._last_output_path = saved_path

            print(f"[RECORDER] 녹음 종료: {saved_path}")

        except BaseException as error:
            with self._lock:
                self._finalize_error = error

        finally:
            with self._lock:
                self._chunks = []
                self._recorded_frames = 0
                self._is_finalizing = False
                self._finalized.set()

    def _resample_and_save(self) -> str:
        with self._lock:
            chunks = list(self._chunks)

        if not chunks:
            raise RuntimeError("녹음된 오디오 데이터가 없습니다.")

        audio = np.concatenate(
            chunks,
            axis=0,
        )

        if audio.ndim != 2 or audio.shape[1] != CHANNELS:
            raise RuntimeError(
                f"지원하지 않는 녹음 shape입니다: {audio.shape}"
            )

        duration = audio.shape[0] / RECORD_SAMPLE_RATE

        if duration < self.min_duration:
            raise ValueError(
                f"녹음이 너무 짧습니다: {duration:.3f}초. "
                f"최소 {self.min_duration:.1f}초가 필요합니다."
            )

        resampled_audio = resample_poly(
            audio,
            OUTPUT_SAMPLE_RATE,
            RECORD_SAMPLE_RATE,
            axis=0,
        ).astype(np.float32, copy=False)

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sf.write(
            self.output_path,
            resampled_audio,
            OUTPUT_SAMPLE_RATE,
            subtype="PCM_16",
        )

        return str(self.output_path)

    def _finalize_simulation(self) -> str:
        path = self.test_audio_path

        if path is None:
            raise RuntimeError("Simulation audio path is not configured.")

        if not path.exists():
            raise FileNotFoundError(
                f"테스트 음성 파일을 찾을 수 없습니다: {path}"
            )

        info = sf.info(path)

        if info.duration < self.min_duration:
            raise ValueError(
                f"테스트 음성이 너무 짧습니다: {info.duration:.3f}초. "
                f"최소 {self.min_duration:.1f}초가 필요합니다."
            )

        if info.duration > self.max_duration:
            raise ValueError(
                f"테스트 음성이 최대 길이를 초과합니다: "
                f"{info.duration:.3f}초."
            )

        return str(path)
