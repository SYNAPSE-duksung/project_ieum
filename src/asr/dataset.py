from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
import torch
import torchaudio

from torch import Tensor
from torch.utils.data import Dataset


class IEUMDataset(Dataset):
    """
    IEUM 구음장애 음성인식 학습용 Dataset.

    Word Alignment CSV의 단어 단위 행을
    parent_wav + segment_id 기준으로 묶어
    하나의 음성 구간과 하나의 정답 문장으로 구성한다.

    같은 segment_id라도 parent_wav가 다르면
    서로 다른 학습 샘플로 유지한다.
    """

    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path,
        *,
        split: str | None = None,
        audio_filename_column: str = "parent_wav",
        audio_path_column: str = "parent_wav_path",
        segment_id_column: str = "segment_id",
        transcript_column: str = "segment_text",
        split_column: str = "split",
        speaker_column: str = "speaker_id",
        segment_start_column: str = "start_sec",
        segment_end_column: str = "end_sec",
        sample_rate: int = 16000,
        min_audio_seconds: float | None = 0.1,
        max_audio_seconds: float | None = 30.0,
        load_audio: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.audio_root = Path(audio_root)

        self.requested_split = split

        self.audio_filename_column = audio_filename_column
        self.audio_path_column = audio_path_column
        self.segment_id_column = segment_id_column
        self.transcript_column = transcript_column
        self.split_column = split_column
        self.speaker_column = speaker_column
        self.segment_start_column = segment_start_column
        self.segment_end_column = segment_end_column

        self.sample_rate = int(sample_rate)
        self.min_audio_seconds = min_audio_seconds
        self.max_audio_seconds = max_audio_seconds
        self.load_audio = load_audio

        self._validate_csv_path()

        dataframe = self._load_dataframe()

        self.samples, self.filter_summary = self._build_samples(
            dataframe
        )

        # CSV에 저장된 경로가 유효하지 않을 때
        # audio_root 아래를 파일명 기준으로 검색하기 위한 인덱스
        self.audio_index: dict[str, Path] | None = None

    def _validate_csv_path(self) -> None:
        """CSV 파일 경로가 존재하는지 확인한다."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                "입력 CSV 파일을 찾을 수 없습니다.\n"
                f"확인한 경로: {self.csv_path}"
            )

    def _load_dataframe(self) -> pd.DataFrame:
        """
        CSV를 읽고 학습에 필요한 컬럼과 결측치를 검사한다.
        """
        dataframe = pd.read_csv(self.csv_path)

        if dataframe.empty:
            raise ValueError(
                "입력 CSV에 데이터가 없습니다."
            )

        required_columns = [
            self.audio_filename_column,
            self.segment_id_column,
            self.transcript_column,
            self.split_column,
            self.speaker_column,
            self.segment_start_column,
            self.segment_end_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "입력 CSV에 필요한 컬럼이 없습니다.\n"
                f"누락 컬럼: {missing_columns}\n"
                f"현재 컬럼: {list(dataframe.columns)}"
            )

        # 시간 정보는 숫자형으로 강제 변환
        dataframe[self.segment_start_column] = pd.to_numeric(
            dataframe[self.segment_start_column],
            errors="coerce",
        )

        dataframe[self.segment_end_column] = pd.to_numeric(
            dataframe[self.segment_end_column],
            errors="coerce",
        )

        essential_columns = [
            self.audio_filename_column,
            self.segment_id_column,
            self.transcript_column,
            self.split_column,
            self.speaker_column,
            self.segment_start_column,
            self.segment_end_column,
        ]

        missing_counts = (
            dataframe[essential_columns]
            .isna()
            .sum()
        )

        invalid_missing_counts = missing_counts[
            missing_counts > 0
        ]

        if not invalid_missing_counts.empty:
            raise ValueError(
                "학습에 필요한 컬럼에 결측치가 있습니다.\n"
                f"{invalid_missing_counts.to_dict()}"
            )

        if self.requested_split is not None:
            available_splits = sorted(
                dataframe[self.split_column]
                .astype(str)
                .unique()
                .tolist()
            )

            dataframe = dataframe[
                dataframe[self.split_column].astype(str)
                == str(self.requested_split)
            ].copy()

            if dataframe.empty:
                raise ValueError(
                    f"split='{self.requested_split}' 데이터가 없습니다.\n"
                    f"사용 가능한 split: {available_splits}"
                )

        return dataframe

    @staticmethod
    def _get_single_value(
        group: pd.DataFrame,
        column: str,
        sample_id: str,
    ) -> Any:
        """
        하나의 segment 안에서 특정 메타데이터 값이
        하나로 일관적인지 확인한다.
        """
        unique_values = (
            group[column]
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_values) != 1:
            raise ValueError(
                f"하나의 segment 안에서 '{column}' 값이 "
                "일관적이지 않습니다.\n"
                f"샘플: {sample_id}\n"
                f"확인된 값: {unique_values[:5]}"
            )

        return unique_values[0]

    def _build_samples(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """
        parent_wav + segment_id 단위로 단어 행을 묶어
        실제 학습 샘플 메타데이터를 생성한다.
        """
        records: list[dict[str, Any]] = []

        group_columns = [
            self.audio_filename_column,
            self.segment_id_column,
        ]

        grouped = dataframe.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )

        for (audio_filename, segment_id), group in grouped:
            audio_filename = str(audio_filename)
            segment_id = str(segment_id)

            sample_id = (
                f"{audio_filename}::{segment_id}"
            )

            transcript = str(
                self._get_single_value(
                    group,
                    self.transcript_column,
                    sample_id,
                )
            ).strip()

            speaker_id = str(
                self._get_single_value(
                    group,
                    self.speaker_column,
                    sample_id,
                )
            )

            split_value = str(
                self._get_single_value(
                    group,
                    self.split_column,
                    sample_id,
                )
            )

            if not transcript:
                raise ValueError(
                    "빈 정답 문장이 존재합니다.\n"
                    f"샘플: {sample_id}"
                )

            segment_start_sec = float(
                group[
                    self.segment_start_column
                ].min()
            )

            segment_end_sec = float(
                group[
                    self.segment_end_column
                ].max()
            )

            duration_seconds = (
                segment_end_sec
                - segment_start_sec
            )

            if segment_start_sec < 0:
                raise ValueError(
                    "음성 구간 시작 시간이 음수입니다.\n"
                    f"샘플: {sample_id}\n"
                    f"시작 시간: {segment_start_sec}"
                )

            if duration_seconds <= 0:
                raise ValueError(
                    "음성 구간 길이가 0 이하입니다.\n"
                    f"샘플: {sample_id}\n"
                    f"시작: {segment_start_sec}\n"
                    f"종료: {segment_end_sec}"
                )

            stored_audio_path: str | None = None

            if self.audio_path_column in group.columns:
                path_values = (
                    group[self.audio_path_column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(path_values) > 1:
                    raise ValueError(
                        "하나의 segment에 여러 "
                        "parent_wav_path가 연결되어 있습니다.\n"
                        f"샘플: {sample_id}\n"
                        f"경로 예시: {path_values[:5]}"
                    )

                if path_values:
                    stored_audio_path = (
                        path_values[0]
                    )

            records.append(
                {
                    "sample_id": sample_id,
                    "audio_filename": (
                        audio_filename
                    ),
                    "stored_audio_path": (
                        stored_audio_path
                    ),
                    "segment_id": segment_id,
                    "segment_start_sec": (
                        segment_start_sec
                    ),
                    "segment_end_sec": (
                        segment_end_sec
                    ),
                    "duration_seconds": (
                        duration_seconds
                    ),
                    "transcript": transcript,
                    "speaker_id": speaker_id,
                    "split": split_value,
                    "source_row_count": int(
                        len(group)
                    ),
                }
            )

        samples = pd.DataFrame(records)

        total_before_filter = len(samples)

        too_short_mask = pd.Series(
            False,
            index=samples.index,
        )

        too_long_mask = pd.Series(
            False,
            index=samples.index,
        )

        if self.min_audio_seconds is not None:
            too_short_mask = (
                samples["duration_seconds"]
                < float(
                    self.min_audio_seconds
                )
            )

        if self.max_audio_seconds is not None:
            too_long_mask = (
                samples["duration_seconds"]
                > float(
                    self.max_audio_seconds
                )
            )

        keep_mask = ~(
            too_short_mask
            | too_long_mask
        )

        filtered_samples = (
            samples.loc[keep_mask]
            .reset_index(drop=True)
        )

        filter_summary = {
            "total_before_filter": int(
                total_before_filter
            ),
            "removed_too_short": int(
                too_short_mask.sum()
            ),
            "removed_too_long": int(
                too_long_mask.sum()
            ),
            "total_after_filter": int(
                len(filtered_samples)
            ),
        }

        if filtered_samples.empty:
            raise ValueError(
                "음성 길이 필터 적용 후 "
                "남은 데이터가 없습니다."
            )

        return (
            filtered_samples,
            filter_summary,
        )

    def _create_audio_index(
        self,
    ) -> dict[str, Path]:
        """
        audio_root 아래 WAV 파일을 파일명 기준으로 인덱싱한다.

        CSV에 저장된 경로가 현재 환경에서 유효하지 않을 경우
        파일명으로 실제 WAV 경로를 찾기 위해 사용한다.
        """
        if not self.audio_root.exists():
            raise FileNotFoundError(
                "오디오 루트 폴더를 찾을 수 없습니다.\n"
                f"확인한 경로: {self.audio_root}"
            )

        audio_index: dict[str, Path] = {}
        duplicate_paths: dict[
            str,
            list[Path],
        ] = {}

        for audio_path in self.audio_root.rglob(
            "*.wav"
        ):
            filename = audio_path.name

            if filename in audio_index:
                duplicate_paths.setdefault(
                    filename,
                    [audio_index[filename]],
                ).append(audio_path)

            else:
                audio_index[filename] = (
                    audio_path
                )

        if duplicate_paths:
            duplicate_name = next(
                iter(duplicate_paths)
            )

            example_paths = (
                duplicate_paths[
                    duplicate_name
                ]
            )

            raise ValueError(
                "audio_root 아래 동일한 파일명의 "
                "WAV가 여러 개 존재합니다.\n"
                f"중복 파일명: {duplicate_name}\n"
                f"경로 예시: "
                f"{[str(path) for path in example_paths[:5]]}"
            )

        return audio_index

    def _resolve_audio_path(
        self,
        stored_audio_path: str | None,
        audio_filename: str,
    ) -> Path:
        """
        실제로 존재하는 부모 WAV 경로를 찾는다.

        우선순위:
        1. CSV의 parent_wav_path
        2. audio_root / audio_filename
        3. audio_root 재귀 검색 결과
        """
        if stored_audio_path:
            stored_path = Path(
                stored_audio_path
            )

            if stored_path.exists():
                return stored_path

        direct_path = (
            self.audio_root
            / audio_filename
        )

        if direct_path.exists():
            return direct_path

        if self.audio_index is None:
            self.audio_index = (
                self._create_audio_index()
            )

        indexed_path = self.audio_index.get(
            audio_filename
        )

        if indexed_path is not None:
            return indexed_path

        raise FileNotFoundError(
            "음성 파일을 찾을 수 없습니다.\n"
            f"파일명: {audio_filename}\n"
            f"CSV 저장 경로: "
            f"{stored_audio_path}\n"
            f"오디오 루트: "
            f"{self.audio_root}"
        )

    def _load_audio_segment(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> Tensor:
        """
        부모 WAV에서 필요한 segment 구간만 읽는다.

        soundfile을 사용하여 전체 WAV를 메모리에 올리지 않고
        필요한 구간만 로드한다.
        """
        with sf.SoundFile(
            str(audio_path)
        ) as audio_file:
            original_sample_rate = (
                audio_file.samplerate
            )

            total_frames = len(
                audio_file
            )

            start_frame = int(
                round(
                    start_sec
                    * original_sample_rate
                )
            )

            end_frame = int(
                round(
                    end_sec
                    * original_sample_rate
                )
            )

            num_frames = (
                end_frame
                - start_frame
            )

            if num_frames <= 0:
                raise ValueError(
                    "유효하지 않은 음성 구간입니다.\n"
                    f"파일: {audio_path}\n"
                    f"시작: {start_sec}\n"
                    f"종료: {end_sec}"
                )

            if start_frame >= total_frames:
                raise ValueError(
                    "segment 시작 위치가 실제 음성 "
                    "길이를 초과합니다.\n"
                    f"파일: {audio_path}\n"
                    f"start_frame: {start_frame}\n"
                    f"전체 frame: {total_frames}"
                )

            # 종료 지점이 실제 WAV보다 길면
            # 파일 끝까지만 읽는다.
            num_frames = min(
                num_frames,
                total_frames - start_frame,
            )

            audio_file.seek(
                start_frame
            )

            waveform_np = (
                audio_file.read(
                    frames=num_frames,
                    dtype="float32",
                    always_2d=True,
                )
            )

        if waveform_np.size == 0:
            raise ValueError(
                "불러온 음성 구간이 비어 있습니다.\n"
                f"파일: {audio_path}\n"
                f"구간: "
                f"{start_sec:.4f}"
                f"~{end_sec:.4f}"
            )

        # soundfile:
        # [time, channel]
        #
        # PyTorch:
        # [channel, time]
        waveform = torch.from_numpy(
            waveform_np.T
        )

        # stereo → mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # 목표 sampling rate와 다르면
        # resampling
        if (
            original_sample_rate
            != self.sample_rate
        ):
            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    orig_freq=(
                        original_sample_rate
                    ),
                    new_freq=(
                        self.sample_rate
                    ),
                )
            )

        waveform = (
            waveform
            .squeeze(0)
            .contiguous()
        )

        if waveform.numel() == 0:
            raise ValueError(
                "전처리 후 waveform이 비어 있습니다.\n"
                f"파일: {audio_path}"
            )

        return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        row = self.samples.iloc[index]

        sample: dict[str, Any] = {
            "sample_id": row[
                "sample_id"
            ],
            "audio_filename": row[
                "audio_filename"
            ],
            "segment_id": row[
                "segment_id"
            ],
            "segment_start_sec": float(
                row[
                    "segment_start_sec"
                ]
            ),
            "segment_end_sec": float(
                row[
                    "segment_end_sec"
                ]
            ),
            "duration_seconds": float(
                row[
                    "duration_seconds"
                ]
            ),
            "transcript": row[
                "transcript"
            ],
            "speaker_id": row[
                "speaker_id"
            ],
            "split": row[
                "split"
            ],
            "source_row_count": int(
                row[
                    "source_row_count"
                ]
            ),
        }

        # 메타데이터 검사만 할 때는
        # 실제 WAV를 읽지 않는다.
        if not self.load_audio:
            sample[
                "stored_audio_path"
            ] = row[
                "stored_audio_path"
            ]

            return sample

        audio_path = (
            self._resolve_audio_path(
                stored_audio_path=row[
                    "stored_audio_path"
                ],
                audio_filename=row[
                    "audio_filename"
                ],
            )
        )

        waveform = (
            self._load_audio_segment(
                audio_path=audio_path,
                start_sec=float(
                    row[
                        "segment_start_sec"
                    ]
                ),
                end_sec=float(
                    row[
                        "segment_end_sec"
                    ]
                ),
            )
        )

        sample.update(
            {
                "audio_path": str(
                    audio_path
                ),
                "waveform": waveform,
                "num_samples": int(
                    waveform.numel()
                ),
                "sample_rate": (
                    self.sample_rate
                ),
                "loaded_duration_seconds": (
                    waveform.numel()
                    / self.sample_rate
                ),
            }
        )

        return sample

    def get_metadata(
        self,
    ) -> pd.DataFrame:
        """
        segment 단위 메타데이터 복사본을 반환한다.
        """
        return self.samples.copy()

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Dataset 기본 통계를 반환한다.
        """
        duration = (
            self.samples[
                "duration_seconds"
            ]
        )

        return {
            "csv_path": str(
                self.csv_path
            ),
            "audio_root": str(
                self.audio_root
            ),
            "requested_split": (
                self.requested_split
            ),
            "sample_count": int(
                len(self.samples)
            ),
            "speaker_count": int(
                self.samples[
                    "speaker_id"
                ].nunique()
            ),
            "split_counts": (
                self.samples[
                    "split"
                ]
                .value_counts()
                .to_dict()
            ),
            "duration_min": float(
                duration.min()
            ),
            "duration_mean": float(
                duration.mean()
            ),
            "duration_median": float(
                duration.median()
            ),
            "duration_max": float(
                duration.max()
            ),
            "sample_rate": (
                self.sample_rate
            ),
            "filter_summary": (
                self.filter_summary
            ),
        }