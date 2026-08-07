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

    기본 원칙
    ----------
    1. CSV의 기존 train / valid / test split을 그대로 사용한다.
    2. parent_wav + segment_id를 원래 segment로 본다.
    3. 30초 이하 segment는 그대로 사용한다.
    4. 30초 초과 segment는 단어 start_sec 기준으로
       최대 30초 이하의 여러 chunk로 나눈다.
    5. abnormal_duration=True인 단어도 정답에서 삭제하지 않는다.
       단, 비정상적으로 긴 end_sec 때문에 chunk가 30초를
       초과하지 않도록 제한한다.
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
        word_column: str = "word",
        split_column: str = "split",
        speaker_column: str = "speaker_id",
        segment_start_column: str = "start_sec",
        segment_end_column: str = "end_sec",
        abnormal_duration_column: str = "abnormal_duration",
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
        load_audio: bool = True,
    ) -> None:

        self.csv_path = Path(csv_path)
        self.audio_root = Path(audio_root)

        self.requested_split = split

        self.audio_filename_column = audio_filename_column
        self.audio_path_column = audio_path_column
        self.segment_id_column = segment_id_column
        self.transcript_column = transcript_column
        self.word_column = word_column
        self.split_column = split_column
        self.speaker_column = speaker_column
        self.segment_start_column = segment_start_column
        self.segment_end_column = segment_end_column
        self.abnormal_duration_column = abnormal_duration_column

        self.sample_rate = int(sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)
        self.load_audio = load_audio

        self._validate_csv_path()

        dataframe = self._load_dataframe()

        self.samples, self.chunk_summary = self._build_samples(
            dataframe
        )

        self.audio_index: dict[str, Path] | None = None

    # ============================================================
    # CSV
    # ============================================================

    def _validate_csv_path(self) -> None:

        if not self.csv_path.exists():
            raise FileNotFoundError(
                "입력 CSV 파일을 찾을 수 없습니다.\n"
                f"경로: {self.csv_path}"
            )

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """
        CSV에서 bool이 문자열 등으로 읽혀도
        안전하게 True/False로 변환한다.
        """

        if pd.isna(value):
            return False

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        normalized = str(value).strip().lower()

        return normalized in {
            "true",
            "1",
            "yes",
            "y",
        }

    def _load_dataframe(self) -> pd.DataFrame:

        dataframe = pd.read_csv(
            self.csv_path
        )

        if dataframe.empty:
            raise ValueError(
                "입력 CSV가 비어 있습니다."
            )

        required_columns = [
            self.audio_filename_column,
            self.segment_id_column,
            self.transcript_column,
            self.word_column,
            self.split_column,
            self.speaker_column,
            self.segment_start_column,
            self.segment_end_column,
            self.abnormal_duration_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "CSV에 필요한 컬럼이 없습니다.\n"
                f"누락 컬럼: {missing_columns}"
            )

        dataframe[
            self.segment_start_column
        ] = pd.to_numeric(
            dataframe[self.segment_start_column],
            errors="coerce",
        )

        dataframe[
            self.segment_end_column
        ] = pd.to_numeric(
            dataframe[self.segment_end_column],
            errors="coerce",
        )

        essential_columns = [
            self.audio_filename_column,
            self.segment_id_column,
            self.word_column,
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

        invalid = missing_counts[
            missing_counts > 0
        ]

        if not invalid.empty:
            raise ValueError(
                "필수 컬럼에 결측치가 있습니다.\n"
                f"{invalid.to_dict()}"
            )

        dataframe[
            self.abnormal_duration_column
        ] = dataframe[
            self.abnormal_duration_column
        ].apply(
            self._to_bool
        )

        # 기존 split을 그대로 사용
        if self.requested_split is not None:

            available_splits = sorted(
                dataframe[
                    self.split_column
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            dataframe = dataframe[
                dataframe[
                    self.split_column
                ].astype(str)
                == str(self.requested_split)
            ].copy()

            if dataframe.empty:
                raise ValueError(
                    f"split='{self.requested_split}' 데이터가 없습니다.\n"
                    f"사용 가능한 split: {available_splits}"
                )

        return dataframe

    # ============================================================
    # 메타데이터
    # ============================================================

    @staticmethod
    def _get_single_value(
        group: pd.DataFrame,
        column: str,
        sample_id: str,
    ) -> Any:

        values = (
            group[column]
            .dropna()
            .unique()
            .tolist()
        )

        if len(values) != 1:
            raise ValueError(
                f"하나의 segment 안에서 '{column}' 값이 "
                "일관되지 않습니다.\n"
                f"sample: {sample_id}\n"
                f"values: {values[:5]}"
            )

        return values[0]

    def _get_audio_path_value(
        self,
        group: pd.DataFrame,
        sample_id: str,
    ) -> str | None:

        if self.audio_path_column not in group.columns:
            return None

        values = (
            group[
                self.audio_path_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(values) > 1:
            raise ValueError(
                "하나의 segment에 여러 parent_wav_path가 있습니다.\n"
                f"sample: {sample_id}"
            )

        if not values:
            return None

        return values[0]

    # ============================================================
    # Chunk 생성
    # ============================================================

    def _sort_words(
        self,
        group: pd.DataFrame,
    ) -> pd.DataFrame:

        # 원래 단어 순서가 있으면 우선 사용
        if "segment_word_index" in group.columns:

            return (
                group
                .sort_values(
                    [
                        "segment_word_index",
                        self.segment_start_column,
                    ]
                )
                .reset_index(drop=True)
            )

        return (
            group
            .sort_values(
                [
                    self.segment_start_column,
                    self.segment_end_column,
                ]
            )
            .reset_index(drop=True)
        )

    def _split_segment_into_chunks(
        self,
        group: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        """
        word start_sec를 기준으로 30초 window 안에
        들어가는 단어들을 같은 chunk에 넣는다.

        비정상적으로 긴 word end_sec는 chunk 구분 기준으로
        사용하지 않는다.
        """

        group = self._sort_words(
            group
        )

        chunks: list[pd.DataFrame] = []

        current_indices: list[int] = []
        chunk_start: float | None = None

        for row_index, row in group.iterrows():

            word_start = float(
                row[self.segment_start_column]
            )

            if not current_indices:

                current_indices = [
                    row_index
                ]

                chunk_start = (
                    word_start
                )

                continue

            assert chunk_start is not None

            # 현재 단어의 시작 시점이 현재 chunk 시작에서
            # 30초 이상 떨어져 있으면 새로운 chunk 시작
            if (
                word_start
                - chunk_start
                >= self.max_audio_seconds
            ):

                chunks.append(
                    group.loc[
                        current_indices
                    ].copy()
                )

                current_indices = [
                    row_index
                ]

                chunk_start = (
                    word_start
                )

            else:

                current_indices.append(
                    row_index
                )

        if current_indices:

            chunks.append(
                group.loc[
                    current_indices
                ].copy()
            )

        return chunks

    def _calculate_chunk_times(
        self,
        chunk: pd.DataFrame,
    ) -> tuple[float, float]:
        """
        실제 WAV에서 읽을 chunk 시작/종료 시간을 계산한다.

        abnormal_duration=True인 단어는 지나치게 긴 end_sec가
        chunk 길이를 비정상적으로 늘리지 않도록 30초에서 자른다.
        """

        chunk_start = float(
            chunk[
                self.segment_start_column
            ].min()
        )

        max_allowed_end = (
            chunk_start
            + self.max_audio_seconds
        )

        normal_rows = chunk[
            ~chunk[
                self.abnormal_duration_column
            ]
        ]

        # 정상 alignment 단어가 있으면
        # 정상 단어의 가장 마지막 end를 기본 종료점으로 사용
        if not normal_rows.empty:

            chunk_end = float(
                normal_rows[
                    self.segment_end_column
                ].max()
            )

        else:

            # 전부 abnormal이면 raw end를 사용하되
            # 반드시 30초 이내로 제한
            chunk_end = float(
                chunk[
                    self.segment_end_column
                ].max()
            )

        # abnormal 단어도 audio 범위에 포함되도록
        # 최소한 마지막 word start까지 포함
        latest_word_start = float(
            chunk[
                self.segment_start_column
            ].max()
        )

        chunk_end = max(
            chunk_end,
            latest_word_start,
        )

        # raw abnormal end가 30초 이상이어도
        # Whisper 최대 길이를 절대 넘기지 않는다.
        chunk_end = min(
            chunk_end,
            max_allowed_end,
        )

        # 극단적으로 end == start인 경우 최소 길이 확보
        if chunk_end <= chunk_start:

            chunk_end = min(
                chunk_start + 0.02,
                max_allowed_end,
            )

        return (
            chunk_start,
            chunk_end,
        )

    def _build_samples(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        dict[str, Any],
    ]:

        records: list[
            dict[str, Any]
        ] = []

        grouped = dataframe.groupby(
            [
                self.audio_filename_column,
                self.segment_id_column,
            ],
            sort=False,
            dropna=False,
        )

        original_segment_count = 0
        chunked_original_count = 0
        abnormal_word_count = 0

        original_split_counts: dict[
            str,
            int,
        ] = {}

        for (
            audio_filename,
            segment_id,
        ), group in grouped:

            original_segment_count += 1

            audio_filename = str(
                audio_filename
            )

            segment_id = str(
                segment_id
            )

            original_sample_id = (
                f"{audio_filename}::{segment_id}"
            )

            speaker_id = str(
                self._get_single_value(
                    group,
                    self.speaker_column,
                    original_sample_id,
                )
            )

            split_value = str(
                self._get_single_value(
                    group,
                    self.split_column,
                    original_sample_id,
                )
            )

            original_transcript = str(
                self._get_single_value(
                    group,
                    self.transcript_column,
                    original_sample_id,
                )
            ).strip()

            stored_audio_path = (
                self._get_audio_path_value(
                    group,
                    original_sample_id,
                )
            )

            original_split_counts[
                split_value
            ] = (
                original_split_counts.get(
                    split_value,
                    0,
                )
                + 1
            )

            abnormal_word_count += int(
                group[
                    self.abnormal_duration_column
                ].sum()
            )

            chunks = (
                self._split_segment_into_chunks(
                    group
                )
            )

            if len(chunks) > 1:
                chunked_original_count += 1

            for (
                chunk_index,
                chunk,
            ) in enumerate(chunks):

                chunk_start, chunk_end = (
                    self._calculate_chunk_times(
                        chunk
                    )
                )

                duration = (
                    chunk_end
                    - chunk_start
                )

                if (
                    duration
                    > self.max_audio_seconds
                    + 1e-6
                ):
                    raise ValueError(
                        "30초 초과 chunk가 생성되었습니다.\n"
                        f"sample: {original_sample_id}\n"
                        f"duration: {duration}"
                    )

                words = (
                    chunk[
                        self.word_column
                    ]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                words = [
                    word
                    for word in words
                    if word
                ]

                if not words:
                    raise ValueError(
                        "chunk에 정답 단어가 없습니다.\n"
                        f"sample: {original_sample_id}"
                    )

                chunk_transcript = (
                    " ".join(words)
                )

                if len(chunks) == 1:

                    sample_id = (
                        original_sample_id
                    )

                else:

                    sample_id = (
                        f"{original_sample_id}"
                        f"::chunk{chunk_index}"
                    )

                records.append(
                    {
                        "sample_id": (
                            sample_id
                        ),
                        "original_sample_id": (
                            original_sample_id
                        ),
                        "audio_filename": (
                            audio_filename
                        ),
                        "stored_audio_path": (
                            stored_audio_path
                        ),
                        "segment_id": (
                            segment_id
                        ),
                        "chunk_index": (
                            int(chunk_index)
                        ),
                        "chunk_count": (
                            int(len(chunks))
                        ),
                        "segment_start_sec": (
                            chunk_start
                        ),
                        "segment_end_sec": (
                            chunk_end
                        ),
                        "duration_seconds": (
                            duration
                        ),
                        "transcript": (
                            chunk_transcript
                        ),
                        "original_transcript": (
                            original_transcript
                        ),
                        "speaker_id": (
                            speaker_id
                        ),
                        "split": (
                            split_value
                        ),
                        "source_row_count": (
                            int(len(chunk))
                        ),
                        "abnormal_word_count": (
                            int(
                                chunk[
                                    self.abnormal_duration_column
                                ].sum()
                            )
                        ),
                    }
                )

        samples = pd.DataFrame(
            records
        )

        if samples.empty:
            raise ValueError(
                "학습 샘플이 생성되지 않았습니다."
            )

        duplicate_count = int(
            samples[
                "sample_id"
            ]
            .duplicated()
            .sum()
        )

        if duplicate_count > 0:
            raise ValueError(
                "중복 sample_id가 존재합니다.\n"
                f"중복 수: {duplicate_count}"
            )

        final_split_counts = (
            samples[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        chunk_summary = {
            "original_segment_count": (
                original_segment_count
            ),
            "final_chunk_count": (
                len(samples)
            ),
            "extra_chunks_created": (
                len(samples)
                - original_segment_count
            ),
            "chunked_original_segment_count": (
                chunked_original_count
            ),
            "abnormal_word_count": (
                abnormal_word_count
            ),
            "original_split_counts": (
                original_split_counts
            ),
            "final_chunk_split_counts": (
                final_split_counts
            ),
        }

        return (
            samples,
            chunk_summary,
        )

    # ============================================================
    # Audio 경로
    # ============================================================

    def _create_audio_index(
        self,
    ) -> dict[str, Path]:

        if not self.audio_root.exists():

            raise FileNotFoundError(
                "오디오 루트가 없습니다.\n"
                f"{self.audio_root}"
            )

        audio_index: dict[
            str,
            Path,
        ] = {}

        duplicates: dict[
            str,
            list[Path],
        ] = {}

        for path in (
            self.audio_root.rglob(
                "*.wav"
            )
        ):

            filename = path.name

            if filename in audio_index:

                duplicates.setdefault(
                    filename,
                    [
                        audio_index[
                            filename
                        ]
                    ],
                ).append(
                    path
                )

            else:

                audio_index[
                    filename
                ] = path

        if duplicates:

            duplicate_name = next(
                iter(duplicates)
            )

            raise ValueError(
                "동일한 WAV 파일명이 여러 개 존재합니다.\n"
                f"파일명: {duplicate_name}"
            )

        return audio_index

    def _resolve_audio_path(
        self,
        stored_audio_path: str | None,
        audio_filename: str,
    ) -> Path:

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

        indexed_path = (
            self.audio_index.get(
                audio_filename
            )
        )

        if indexed_path is not None:

            return indexed_path

        raise FileNotFoundError(
            "WAV 파일을 찾을 수 없습니다.\n"
            f"파일: {audio_filename}"
        )

    # ============================================================
    # Audio Loading
    # ============================================================

    def _load_audio_segment(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> Tensor:

        with sf.SoundFile(
            str(audio_path)
        ) as audio_file:

            original_sample_rate = (
                audio_file.samplerate
            )

            total_frames = (
                len(audio_file)
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
                    f"{audio_path}\n"
                    f"{start_sec} ~ {end_sec}"
                )

            if start_frame >= total_frames:

                raise ValueError(
                    "음성 시작 위치가 WAV 길이를 초과합니다.\n"
                    f"{audio_path}"
                )

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
                "음성 구간이 비어 있습니다.\n"
                f"{audio_path}"
            )

        waveform = (
            torch.from_numpy(
                waveform_np.T
            )
        )

        if waveform.shape[0] > 1:

            waveform = (
                waveform.mean(
                    dim=0,
                    keepdim=True,
                )
            )

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

        return (
            waveform
            .squeeze(0)
            .contiguous()
        )

    # ============================================================
    # PyTorch Dataset
    # ============================================================

    def __len__(self) -> int:

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = (
            self.samples.iloc[
                index
            ]
        )

        sample: dict[
            str,
            Any,
        ] = {
            "sample_id": (
                row[
                    "sample_id"
                ]
            ),
            "original_sample_id": (
                row[
                    "original_sample_id"
                ]
            ),
            "audio_filename": (
                row[
                    "audio_filename"
                ]
            ),
            "segment_id": (
                row[
                    "segment_id"
                ]
            ),
            "chunk_index": (
                int(
                    row[
                        "chunk_index"
                    ]
                )
            ),
            "chunk_count": (
                int(
                    row[
                        "chunk_count"
                    ]
                )
            ),
            "segment_start_sec": (
                float(
                    row[
                        "segment_start_sec"
                    ]
                )
            ),
            "segment_end_sec": (
                float(
                    row[
                        "segment_end_sec"
                    ]
                )
            ),
            "duration_seconds": (
                float(
                    row[
                        "duration_seconds"
                    ]
                )
            ),
            "transcript": (
                row[
                    "transcript"
                ]
            ),
            "speaker_id": (
                row[
                    "speaker_id"
                ]
            ),
            "split": (
                row[
                    "split"
                ]
            ),
        }

        if not self.load_audio:

            sample[
                "stored_audio_path"
            ] = row[
                "stored_audio_path"
            ]

            return sample

        audio_path = (
            self._resolve_audio_path(
                stored_audio_path=(
                    row[
                        "stored_audio_path"
                    ]
                ),
                audio_filename=(
                    row[
                        "audio_filename"
                    ]
                ),
            )
        )

        waveform = (
            self._load_audio_segment(
                audio_path=(
                    audio_path
                ),
                start_sec=(
                    sample[
                        "segment_start_sec"
                    ]
                ),
                end_sec=(
                    sample[
                        "segment_end_sec"
                    ]
                ),
            )
        )

        sample.update(
            {
                "audio_path": (
                    str(audio_path)
                ),
                "waveform": (
                    waveform
                ),
                "num_samples": (
                    int(
                        waveform.numel()
                    )
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

    # ============================================================
    # Utility
    # ============================================================

    def get_metadata(
        self,
    ) -> pd.DataFrame:

        return (
            self.samples.copy()
        )

    def summary(
        self,
    ) -> dict[str, Any]:

        duration = (
            self.samples[
                "duration_seconds"
            ]
        )

        return {
            "csv_path": (
                str(self.csv_path)
            ),
            "audio_root": (
                str(self.audio_root)
            ),
            "requested_split": (
                self.requested_split
            ),
            "sample_count": (
                int(
                    len(self.samples)
                )
            ),
            "original_segment_count": (
                self.chunk_summary[
                    "original_segment_count"
                ]
            ),
            "speaker_count": (
                int(
                    self.samples[
                        "speaker_id"
                    ].nunique()
                )
            ),
            "split_counts": (
                self.samples[
                    "split"
                ]
                .value_counts()
                .to_dict()
            ),
            "duration_min": (
                float(
                    duration.min()
                )
            ),
            "duration_mean": (
                float(
                    duration.mean()
                )
            ),
            "duration_median": (
                float(
                    duration.median()
                )
            ),
            "duration_max": (
                float(
                    duration.max()
                )
            ),
            "sample_rate": (
                self.sample_rate
            ),
            "chunk_summary": (
                self.chunk_summary
            ),
        }