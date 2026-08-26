from pathlib import Path

import pandas as pd

from src.asr.dataset import IEUMDataset


class PersonalizationIEUMDataset(IEUMDataset):
    """
    개인화 ASR 학습용 Dataset.

    기존 범용 모델의 IEUMDataset을 그대로 재사용하면서,
    특정 화자의 데이터만 선택할 수 있도록 확장한다.

    개인화 데이터셋의 train / valid / test는
    personal_split 컬럼을 사용한다.
    """

    def __init__(
        self,
        csv_path: str | Path,
        audio_root: str | Path,
        speaker_id: str,
        *,
        split: str | None = None,
        split_column: str = "personal_split",
        **kwargs,
    ) -> None:

        self.requested_speaker_id = str(
            speaker_id
        )

        super().__init__(
            csv_path=csv_path,
            audio_root=audio_root,
            split=split,
            split_column=split_column,
            **kwargs,
        )

    def _load_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        기존 IEUMDataset 방식으로 CSV를 불러온 뒤,
        개인화 대상 화자만 선택한다.
        """

        dataframe = super()._load_dataframe()

        available_speakers = sorted(
            dataframe[
                self.speaker_column
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        dataframe = dataframe[
            dataframe[
                self.speaker_column
            ].astype(str)
            == self.requested_speaker_id
        ].copy()

        if dataframe.empty:
            raise ValueError(
                f"speaker_id='{self.requested_speaker_id}' "
                "데이터가 없습니다.\n"
                f"사용 가능한 화자 수: {len(available_speakers)}"
            )

        return dataframe

    def set_sample_weights(
        self,
        sample_weights: list[float],
        ) -> None:
        """
        각 학습 chunk에 Error Profile 기반 sample weight를 저장한다.

        sample_weights의 순서는 self.samples의 순서와
        정확히 일치해야 한다.
        """

        if len(sample_weights) != len(self.samples):
            raise ValueError(
                "sample_weights 개수가 Dataset sample 수와 다릅니다.\n"
                f"Dataset samples: {len(self.samples)}\n"
                f"sample_weights: {len(sample_weights)}"
            )

        self.samples[
            "sample_weight"
        ] = [
            float(weight)
            for weight in sample_weights
        ]

    def __getitem__(
        self,
        index: int,
    ):
        """
        기존 IEUMDataset sample을 그대로 사용하면서,
        Error Profile weight가 존재하면 함께 반환한다.
        """

        sample = super().__getitem__(
            index
        )

        if (
            "sample_weight"
            in self.samples.columns
        ):
            sample[
                "sample_weight"
            ] = float(
                self.samples.iloc[
                    index
                ][
                    "sample_weight"
                ]
            )

        return sample