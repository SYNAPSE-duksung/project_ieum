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

    Word Alignment CSV를 기반으로 학습 샘플을 생성한다.

    기본 단위:
        parent_wav + segment_id

    Whisper 입력 길이인 max_audio_seconds를 초과하는 segment는
    삭제하지 않고 word alignment 경계에서 여러 chunk로 나눈다.

    따라서 CSV의 기존 train / valid / test split과
    원래 segment를 모두 유지한다.
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
        word_column: str = "word",
        segment_start_column: str = "start_sec",
        segment_end_column: str = "end_sec",
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

        self.split_column = split_column
        self.speaker_column = speaker_column

        self.word_column = word_column

        self.segment_start_column = segment_start_column
        self.segment_end_column = segment_end_column

        self.sample_rate = int(sample_rate)
        self.max_audio_seconds = float(max_audio_seconds)

        self.load_audio = load_audio

        self._validate_csv_path()

        dataframe = self._load_dataframe()

        self.samples, self.chunk_summary = self._build_samples(
            dataframe
        )

        self.audio_index: dict[str, Path] | None = None

    # =========================================================
    # CSV
    # =========================================================

    def _validate_csv_path(self) -> None:
        """CSV 파일 존재 여부를 확인한다."""

        if not self.csv_path.exists():
            raise FileNotFoundError(
                "입력 CSV 파일을 찾을 수 없습니다.\n"
                f"확인한 경로: {self.csv_path}"
            )

    def _load_dataframe(self) -> pd.DataFrame:
        """
        CSV를 읽고 학습에 필요한 컬럼을 검사한다.
        """

        dataframe = pd.read_csv(
            self.csv_path
        )

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
            self.word_column,
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

        dataframe[
            self.segment_start_column
        ] = pd.to_numeric(
            dataframe[
                self.segment_start_column
            ],
            errors="coerce",
        )

        dataframe[
            self.segment_end_column
        ] = pd.to_numeric(
            dataframe[
                self.segment_end_column
            ],
            errors="coerce",
        )

        essential_columns = [
            self.audio_filename_column,
            self.segment_id_column,
            self.transcript_column,
            self.split_column,
            self.speaker_column,
            self.word_column,
            self.segment_start_column,
            self.segment_end_column,
        ]

        missing_counts = (
            dataframe[
                essential_columns
            ]
            .isna()
            .sum()
        )

        invalid_missing_counts = (
            missing_counts[
                missing_counts > 0
            ]
        )

        if not invalid_missing_counts.empty:
            raise ValueError(
                "학습에 필요한 컬럼에 결측치가 있습니다.\n"
                f"{invalid_missing_counts.to_dict()}"
            )

        # 기존 CSV의 split 그대로 사용
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

    # =========================================================
    # Segment / Chunk 생성
    # =========================================================

    @staticmethod
    def _get_single_value(
        group: pd.DataFrame,
        column: str,
        sample_id: str,
    ) -> Any:
        """
        하나의 segment 안에서 메타데이터 값이
        하나로 일관적인지 검사한다.
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

    def _get_audio_path_value(
        self,
        group: pd.DataFrame,
        sample_id: str,
    ) -> str | None:
        """
        segment에 대응하는 parent_wav_path를 얻는다.
        """

        if self.audio_path_column not in group.columns:
            return None

        path_values = (
            group[
                self.audio_path_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(path_values) > 1:
            raise ValueError(
                "하나의 segment에 여러 parent_wav_path가 "
                "연결되어 있습니다.\n"
                f"샘플: {sample_id}\n"
                f"경로 예시: {path_values[:5]}"
            )

        if not path_values:
            return None

        return path_values[0]

    def _split_segment_into_chunks(
        self,
        group: pd.DataFrame,
    ) -> list[pd.DataFrame]:
        """
        하나의 segment를 최대 max_audio_seconds 이내의
        word 단위 chunk로 나눈다.

        단어를 중간에서 자르지 않는다.
        """

        group = (
            group
            .sort_values(
                by=[
                    self.segment_start_column,
                    self.segment_end_column,
                ]
            )
            .reset_index(drop=True)
        )

        chunks: list[pd.DataFrame] = []

        current_indices: list[int] = []
        current_start: float | None = None

        for row_index, row in group.iterrows():

            word_start = float(
                row[
                    self.segment_start_column
                ]
            )

            word_end = float(
                row[
                    self.segment_end_column
                ]
            )

            if word_end <= word_start:
                raise ValueError(
                    "단어 alignment 시간이 올바르지 않습니다.\n"
                    f"start={word_start}, end={word_end}"
                )

            word_duration = (
                word_end - word_start
            )

            # 한 단어 자체가 Whisper 최대 길이를 넘는 경우는
            # word boundary 기반 분할로 해결할 수 없다.
            if word_duration > self.max_audio_seconds:
                raise ValueError(
                    "한 단어 alignment 구간이 "
                    "Whisper 최대 입력 길이를 초과합니다.\n"
                    f"word: {row[self.word_column]}\n"
                    f"duration: {word_duration:.4f}초"
                )

            if not current_indices:
                current_indices = [
                    row_index
                ]
                current_start = (
                    word_start
                )
                continue

            candidate_duration = (
                word_end
                - float(current_start)
            )

            # 현재 단어까지 넣으면 30초를 넘을 경우
            # 이전 단어까지 하나의 chunk로 확정
            if (
                candidate_duration
                > self.max_audio_seconds
            ):
                chunk = group.loc[
                    current_indices
                ].copy()

                chunks.append(
                    chunk
                )

                current_indices = [
                    row_index
                ]

                current_start = (
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

    def _build_samples(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        parent_wav + segment_id를 원래 segment로 보고,
        30초를 초과할 경우 word boundary 기준으로 chunk를 생성한다.
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

        original_segment_count = 0
        long_segment_count = 0

        chunked_segment_count = 0

        split_original_counts: dict[
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

            base_sample_id = (
                f"{audio_filename}::{segment_id}"
            )

            speaker_id = str(
                self._get_single_value(
                    group,
                    self.speaker_column,
                    base_sample_id,
                )
            )

            split_value = str(
                self._get_single_value(
                    group,
                    self.split_column,
                    base_sample_id,
                )
            )

            # 원본 segment_text 자체도 일관적인지 검사
            original_transcript = str(
                self._get_single_value(
                    group,
                    self.transcript_column,
                    base_sample_id,
                )
            ).strip()

            if not original_transcript:
                raise ValueError(
                    "빈 segment_text가 존재합니다.\n"
                    f"샘플: {base_sample_id}"
                )

            stored_audio_path = (
                self._get_audio_path_value(
                    group,
                    base_sample_id,
                )
            )

            split_original_counts[
                split_value
            ] = (
                split_original_counts.get(
                    split_value,
                    0,
                )
                + 1
            )

            group = (
                group
                .sort_values(
                    by=[
                        self.segment_start_column,
                        self.segment_end_column,
                    ]
                )
                .reset_index(drop=True)
            )

            segment_start = float(
                group[
                    self.segment_start_column
                ].min()
            )

            segment_end = float(
                group[
                    self.segment_end_column
                ].max()
            )

            original_duration = (
                segment_end
                - segment_start
            )

            if original_duration <= 0:
                raise ValueError(
                    "segment 길이가 0 이하입니다.\n"
                    f"샘플: {base_sample_id}"
                )

            chunks = (
                self._split_segment_into_chunks(
                    group
                )
            )

            if len(chunks) > 1:
                long_segment_count += 1

            for chunk_index, chunk in enumerate(
                chunks
            ):
                chunk_start = float(
                    chunk[
                        self.segment_start_column
                    ].min()
                )

                chunk_end = float(
                    chunk[
                        self.segment_end_column
                    ].max()
                )

                chunk_duration = (
                    chunk_end
                    - chunk_start
                )

                if (
                    chunk_duration
                    > self.max_audio_seconds
                    + 1e-6
                ):
                    raise ValueError(
                        "chunk 생성 후에도 최대 길이를 "
                        "초과했습니다.\n"
                        f"sample: {base_sample_id}\n"
                        f"duration: {chunk_duration}"
                    )

                # chunk에 포함된 alignment 단어들로
                # 정답 문장을 새로 구성
                chunk_words = (
                    chunk[
                        self.word_column
                    ]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

                chunk_words = [
                    word
                    for word in chunk_words
                    if word
                ]

                if not chunk_words:
                    raise ValueError(
                        "chunk에 정답 단어가 없습니다.\n"
                        f"sample: {base_sample_id}"
                    )

                chunk_transcript = (
                    " ".join(
                        chunk_words
                    )
                )

                if len(chunks) == 1:
                    sample_id = (
                        base_sample_id
                    )

                else:
                    sample_id = (
                        f"{base_sample_id}"
                        f"::chunk{chunk_index}"
                    )

                records.append(
                    {
                        "sample_id": (
                            sample_id
                        ),
                        "original_sample_id": (
                            base_sample_id
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
                        "chunk_index": int(
                            chunk_index
                        ),
                        "chunk_count": int(
                            len(chunks)
                        ),
                        "segment_start_sec": (
                            chunk_start
                        ),
                        "segment_end_sec": (
                            chunk_end
                        ),
                        "duration_seconds": (
                            chunk_duration
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
                        "source_row_count": int(
                            len(chunk)
                        ),
                    }
                )

                chunked_segment_count += 1

        samples = pd.DataFrame(
            records
        )

        if samples.empty:
            raise ValueError(
                "생성된 학습 샘플이 없습니다."
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
                "중복 sample_id가 생성되었습니다.\n"
                f"중복 개수: {duplicate_count}"
            )

        split_chunk_counts = (
            samples[
                "split"
            ]
            .value_counts()
            .to_dict()
        )

        chunk_summary = {
            "original_segment_count": (
                int(
                    original_segment_count
                )
            ),
            "long_segment_count": (
                int(
                    long_segment_count
                )
            ),
            "final_chunk_count": (
                int(
                    chunked_segment_count
                )
            ),
            "extra_chunks_created": (
                int(
                    chunked_segment_count
                    - original_segment_count
                )
            ),
            "original_split_counts": (
                split_original_counts
            ),
            "final_chunk_split_counts": (
                split_chunk_counts
            ),
        }

        return samples, chunk_summary

    # =========================================================
    # WAV 경로
    # =========================================================

    def _create_audio_index(
        self,
    ) -> dict[str, Path]:
        """
        audio_root 아래 WAV 파일을 파일명 기준으로 인덱싱한다.
        """

        if not self.audio_root.exists():
            raise FileNotFoundError(
                "오디오 루트 폴더를 찾을 수 없습니다.\n"
                f"확인한 경로: {self.audio_root}"
            )

        audio_index: dict[
            str,
            Path,
        ] = {}

        duplicate_paths: dict[
            str,
            list[Path],
        ] = {}

        for audio_path in (
            self.audio_root.rglob(
                "*.wav"
            )
        ):
            filename = (
                audio_path.name
            )

            if filename in audio_index:

                duplicate_paths.setdefault(
                    filename,
                    [
                        audio_index[
                            filename
                        ]
                    ],
                ).append(
                    audio_path
                )

            else:
                audio_index[
                    filename
                ] = audio_path

        if duplicate_paths:
            duplicate_name = next(
                iter(
                    duplicate_paths
                )
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
        실제 부모 WAV 경로를 찾는다.
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

        indexed_path = (
            self.audio_index.get(
                audio_filename
            )
        )

        if indexed_path is not None:
            return indexed_path

        raise FileNotFoundError(
            "음성 파일을 찾을 수 없습니다.\n"
            f"파일명: {audio_filename}\n"
            f"CSV 저장 경로: {stored_audio_path}\n"
            f"오디오 루트: {self.audio_root}"
        )

    # =========================================================
    # WAV 로딩
    # =========================================================

    def _load_audio_segment(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> Tensor:
        """
        부모 WAV에서 필요한 chunk 구간만 읽는다.
        """

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
                    f"파일: {audio_path}\n"
                    f"시작: {start_sec}\n"
                    f"종료: {end_sec}"
                )

            if start_frame >= total_frames:
                raise ValueError(
                    "chunk 시작 위치가 실제 WAV 길이를 "
                    "초과합니다.\n"
                    f"파일: {audio_path}"
                )

            num_frames = min(
                num_frames,
                total_frames
                - start_frame,
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
                f"파일: {audio_path}"
            )

        waveform = (
            torch.from_numpy(
                waveform_np.T
            )
        )

        # Stereo → Mono
        if waveform.shape[0] > 1:
            waveform = (
                waveform.mean(
                    dim=0,
                    keepdim=True,
                )
            )

        # Sampling rate 보정
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
                "전처리 후 waveform이 비어 있습니다."
            )

        return waveform

    # =========================================================
    # PyTorch Dataset
    # =========================================================

    def __len__(
        self,
    ) -> int:
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
            "chunk_index": int(
                row[
                    "chunk_index"
                ]
            ),
            "chunk_count": int(
                row[
                    "chunk_count"
                ]
            ),
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
            "source_row_count": int(
                row[
                    "source_row_count"
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
                "waveform": (
                    waveform
                ),
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

    # =========================================================
    # 검사용 메서드
    # =========================================================

    def get_metadata(
        self,
    ) -> pd.DataFrame:
        """
        학습 chunk 단위 메타데이터를 반환한다.
        """

        return self.samples.copy()

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
                str(
                    self.csv_path
                )
            ),
            "audio_root": (
                str(
                    self.audio_root
                )
            ),
            "requested_split": (
                self.requested_split
            ),
            "sample_count": int(
                len(
                    self.samples
                )
            ),
            "original_segment_count": (
                self.chunk_summary[
                    "original_segment_count"
                ]
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
            "chunk_summary": (
                self.chunk_summary
            ),
        }