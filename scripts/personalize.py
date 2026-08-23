붙여넣은 마크다운(1)(20260823-002904).md
파일
결과가 나온 거 해석하고 정리해줘

Three Model Comparison Summary

시각화를 표시할 수 없습니다

Three Model Comparison 37speakers

시각화를 표시할 수 없습니다

Three Model Improvement Summary

시각화를 표시할 수 없습니다
이 결과 기준으로 세 모델 성능 비교를 노션 정리용으로 작성해줘
이렇게 있고 이제 다음 단계로 넘어가자
model_loader(2).py
Python
extended_vocab(1).json
파일
personalize(6).py
Python
model_loader(4).py
Python

좋아. 올려준 세 파일을 기준으로 보면 FastAPI용 개인화 모델 로더를 구현하는 데 필요한 핵심 정보는 확보됐어. 별도로 최종 학습 ipynb을 받을 필요는 없어.

확인된 구조는 정확히 이거야.

개인화 모델
Whisper Small Encoder
        ↓
BiGRU CTC
  hidden_size = 512
  num_layers = 2
  dropout = 0.1
        ↓
CTC Classifier
  vocab_size = 1095

개인화 학습에서는 범용 모델의 기존 vocab 824개를 1095개로 확장하고, 기존 classifier의 0~823번 가중치는 범용 모델에서 복사한 뒤 새로 추가된 824~1094번 출력만 초기화해서 학습했어. 실제 extended_vocab.json도 ID가 0~1094, 즉 1095개 vocab인 것을 확인할 수 있어.

그리고 최종 개인화 best_model.pt에는 model_state_dict가 저장되며, 학습이 끝난 뒤에도 best_checkpoint["model_state_dict"]를 그대로 다시 모델에 로드해서 Test 평가를 수행했어. 따라서 FastAPI에서도 동일한 모델 구조를 먼저 만든 다음 개인화 best_model.pt의 model_state_dict를 그대로 로드하면 돼.

여기서 중요한 차이가 하나 있어. 지금 올려준 src/personalization/model_loader.py의 load_general_model()은 개인화 학습 시작용 함수야. 범용 824 vocab checkpoint를 읽고 → 1095 classifier를 새로 만든 다음 → 일부 weight를 복사하는 함수야.

FastAPI 추론에서는 이 과정을 다시 할 필요가 없어. 이미 학습이 끝난 개인화 checkpoint가 있으니까 새로운 함수를 추가하는 게 가장 깔끔해.

load_personalized_model(...)

역할은 이렇게 하면 돼.

best_model.pt
      ↓
extended_vocab.json (1095)
      ↓
WhisperEncoder
      ↓
BiGRUCTC(vocab_size=1095)
      ↓
CTCASRModel
      ↓
checkpoint["model_state_dict"] 전체 로드
      ↓
model.eval()

그리고 HYH와 SKY는 같은 extended_vocab.json을 사용했으므로 구조는 이렇게 유지하면 돼.

models/
├── final/
│   ├── best_model.pt
│   ├── vocab.json
│   └── extended_vocab.json       ← 공통 사용
│
└── personalized/
    ├── HYH_M_22/
    │   ├── best_model.pt
    │   ├── run_config.json
    │   └── test_summary.json
    │
    └── SKY_M_24/
        ├── best_model.pt
        ├── run_config.json
        └── test_summary.json

이 구조면 아주 좋아.

또 하나 중요한 점은 두 개인화 모델 모두 학습할 때 encoder_mode="freeze"였다는 거야. 다만 추론 시에는 requires_grad 여부가 결과에 영향을 주지 않기 때문에, FastAPI에서는 그냥 freeze 상태로 생성하고 eval()로 두면 돼. 개인화 학습 자체도 encoder_mode를 load_general_model()로 전달해서 모델을 생성하는 구조야.

이제 수정 순서

다음부터는 한 파일씩 수정하는 게 안전해.

1단계 — src/personalization/model_loader.py

여기에 load_personalized_model()을 추가해서 HYH/SKY의 .pt가 정말 정상적으로 복원되는지 먼저 확인.

2단계 — FastAPI 기존 model_loader.py

현재 범용 모델 하나만 로드하는 구조라면 다음처럼 바꿔.

General model
HYH_M_22 personalized model
SKY_M_24 personalized model

서버 시작 시 3개를 한 번만 로드하도록 만들기.

3단계 — inference.py

같은 입력 음성을

음성
 ├─→ General → 범용 결과
 └─→ 선택된 Personalized → 개인화 결과

두 모델에 넣고 각각 decode.

4단계 — FastAPI endpoint

예를 들어 요청에서

speaker_id = HYH_M_22

를 받으면 응답을

{
    "speaker_id": "HYH_M_22",
    "general_text": "...",
    "personalized_text": "..."
}

형태로 반환하게 하면 돼.

이렇게 구현하면 나중에 라즈베리파이에서도 한 번 녹음 → 범용 인식 결과 + 해당 화자 개인화 결과를 동시에 보여줄 수 있어.

그리고 현재 model_loader.py를 보니 우리가 걱정했던 824 → 1095 classifier 확장 방식도 명확하게 확인됐기 때문에 이제 추측해서 코드를 짤 필요가 없어.

다음은 바로 1단계부터 하면 돼. src/personalization/model_loader.py에 load_personalized_model()을 추가하는 코드부터 만들고, HYH/SKY checkpoint가 정상 로드되는지 VS Code에서 테스트하자.

근데 threshold나 alpha 값 정해진 이후로 이걸 적어서 수정해야 하는 파일은 없어?

응. 정해진 threshold와 alpha 값은 기록·반영해야 하는 곳이 있어. 다만 FastAPI 추론 코드에 threshold/alpha를 넣는 건 아니야.

우리가 최종 Error Profile FT에서 정한 값은 학습할 때만 사용되는 하이퍼파라미터야. personalize.py를 보면 min_count, min_ratio, alpha, max_weight로 Error Profile을 필터링하고 sample weight를 만든 뒤 그 weight로 학습하도록 되어 있어.

즉 흐름은:

threshold / alpha
       ↓
Error Profile 필터링
       ↓
sample weight 계산
       ↓
개인화 Fine-tuning
       ↓
best_model.pt
       ↓
FastAPI 추론

그래서 이미 만들어진 HYH_M_22 / SKY_M_24의 best_model.pt에는 그 설정으로 학습된 결과가 가중치에 반영되어 있어. FastAPI가 추론할 때 min_count나 alpha를 다시 알 필요는 없어.

다만 프로젝트 코드 정리 측면에서 확인해야 할 곳은 있어.

personalize.py의 기본값
현재 업로드한 코드에는 기본값이 min_count=2, min_ratio=0.0, alpha=0.5, max_weight=None으로 되어 있어. 최종 실험에서 선정한 값이 이것과 다르다면, 앞으로 같은 학습을 재현할 수 있도록 default를 최종값으로 바꾸거나 별도 config로 관리하는 게 좋아.
run_config.json은 반드시 유지
이건 오히려 잘해놨어. personalize.py가 실제 실행 당시의 min_count, min_ratio, alpha, max_weight 등을 run_config.json에 저장하고 있어. 그래서 HYH와 SKY 폴더에 run_config.json을 Git에 올려둔 게 중요해. .pt는 Git에서 제외해도 어떤 설정으로 학습한 모델인지 추적할 수 있어.
model_loader.py에는 넣지 않음
load_general_model()이나 앞으로 만들 load_personalized_model()에는 threshold/alpha가 필요 없어. 이 파일은 모델 구조를 만들고 checkpoint weight를 불러오는 역할만 하면 돼.
FastAPI에도 넣지 않음
FastAPI는 최종 best_model.pt를 가져와 추론만 하므로 threshold/alpha와 무관해.

그래서 다음 단계로 넘어가기 전에 딱 하나 확인하면 돼. HYH와 SKY의 run_config.json에 우리가 최종 선정했던 threshold와 alpha가 정확하게 기록되어 있는지만 보면 돼.

그 두 run_config.json을 지금 이미 VS Code에 넣어뒀으니까, 내용을 보여주면 최종 실험 설정과 맞는지 확인하고, 필요하면 personalize.py의 기본값까지 최종값으로 정리한 뒤 FastAPI 작업으로 넘어가자.

정확하게 되어 있어 run_config.json에 그럼 personalize.py 수정하고 model_loader.py 수정된 거 반영해서 전체 코드 작성해줘

수정본 전체 파일로 만들어뒀어.

수정된 personalize.py

수정된 model_loader.py

이번 personalize.py는 기존 로직은 그대로 두고 최종 실험에서 확정한 기본값을 반영했어.

min_count = 5
min_ratio = 0.2
alpha = 0.5
epochs = 20
patience = 3

원래 alpha=0.5, patience=3은 이미 같았고, 실제로 바뀐 건 min_count 2→5, min_ratio 0.0→0.2, epochs 10→20이야. CLI에서 값을 직접 주면 여전히 그 값이 우선되기 때문에 threshold/alpha 실험용으로도 계속 사용할 수 있어. 기존 personalize.py가 Error Profile threshold와 sample weight를 학습 단계에서 적용하는 구조도 그대로 유지했어.

model_loader.py는 기존 load_general_model()을 그대로 유지하고, 새로 **load_personalized_model()**을 추가했어. 기존 함수는 범용 824-vocab checkpoint를 1095-vocab 개인화 초기 모델로 확장하는 학습 시작용 함수이고, 새 함수는 이미 학습이 끝난 HYH/SKY의 best_model.pt를 추론용으로 직접 복원하는 역할이야. 기존 개인화 학습 로더의 구조인 Whisper Small → BiGRU → extended-vocab classifier를 그대로 따르도록 했어.

새 함수 사용 예시는 이렇게 될 거야.

import torch

from src.personalization.model_loader import (
    load_personalized_model,
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model, tokenizer, checkpoint = load_personalized_model(
    checkpoint_path=(
        "models/personalized/"
        "HYH_M_22/best_model.pt"
    ),
    extended_vocab_path=(
        "models/final/extended_vocab.json"
    ),
    run_config_path=(
        "models/personalized/"
        "HYH_M_22/run_config.json"
    ),
    device=device,
)

이때 run_config.json도 선택적으로 읽어서 저장된 encoder_mode=freeze와 loader 설정이 맞는지 검증하게 했고, checkpoint classifier 출력 크기가 extended vocab의 1095개와 정확히 맞는지도 확인하게 넣었어. 개인화 checkpoint는 이미 classifier까지 학습이 끝난 모델이므로 state_dict 전체를 strict=True로 로드해. extended vocab 자체도 0~1094의 1095개 토큰으로 구성되어 있어.

VS Code에서는 각각 기존 파일에 덮어쓰면 돼.

scripts/personalize.py
        ↑
새 personalize.py

src/personalization/model_loader.py
        ↑
새 model_loader.py

그리고 commit할 때 best_model.pt는 지금 .gitignore에 걸려 있으니까 그대로 로컬에만 남겨두면 돼.

다음 단계는 이 load_personalized_model()이 HYH_M_22와 SKY_M_24의 checkpoint를 실제로 오류 없이 로드하는지 VS Code에서 smoke test하는 거야. 그 테스트가 통과한 다음 FastAPI용 model_loader.py를 수정하는 게 안전해. 

personalize.py
코드
model_loader.py
코드

라이브러리
/
이음
/
personalize.py
from __future__ import annotations

import argparse
import json
import random
import sys

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IEUM modules
# ============================================================

from src.asr.feature_extractor import (
    IEUMWhisperFeatureExtractor,
)

from src.asr.tokenizer import (
    CTCCharacterTokenizer,
)

from src.personalization.dataset import (
    PersonalizationIEUMDataset,
)

from src.personalization.model_loader import (
    load_general_model,
)

from src.personalization.trainer import (
    PersonalizationTrainer,
)

from src.personalization.error_profile import (
    build_raw_error_profile,
    analyze_vocab_unsupported_errors,
    filter_error_profile,
)

from src.personalization.weighting import (
    calculate_sample_weights,
)


# ============================================================
# Seed
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CTC target encoding
# ============================================================

def encode_targets(
    references: list[str],
    tokenizer: CTCCharacterTokenizer,
) -> dict[str, torch.Tensor]:

    encoded_targets = [
        tokenizer.encode(text)
        for text in references
    ]

    target_lengths = torch.tensor(
        [
            len(target)
            for target in encoded_targets
        ],
        dtype=torch.long,
    )

    flattened_targets = torch.tensor(
        [
            token
            for target in encoded_targets
            for token in target
        ],
        dtype=torch.long,
    )

    return {
        "targets": flattened_targets,
        "target_lengths": target_lengths,
    }


# ============================================================
# Raw CTC Collator
# ============================================================

class PersonalizationCollator:
    """
    개인화 학습용 Raw Audio Collator.

    waveform
        ↓
    Whisper Log-Mel Feature
        ↓
    CTC target encoding

    Error Profile mode에서는 sample_weight가 존재할 경우
    batch["sample_weights"]로 함께 전달한다.
    """

    def __init__(
        self,
        feature_extractor: IEUMWhisperFeatureExtractor,
        tokenizer: CTCCharacterTokenizer,
    ) -> None:

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        waveforms = [
            sample["waveform"]
            for sample in samples
        ]

        references = [
            sample["transcript"]
            for sample in samples
        ]

        feature_batch = self.feature_extractor.batch(
            waveforms
        )

        target_data = encode_targets(
            references=references,
            tokenizer=self.tokenizer,
        )

        batch = {
            "input_features": (
                feature_batch["input_features"]
            ),
            "audio_num_samples": (
                feature_batch["audio_num_samples"]
            ),
            "targets": (
                target_data["targets"]
            ),
            "target_lengths": (
                target_data["target_lengths"]
            ),
            "references": references,
        }

        # Error Profile mode에서만 존재한다.
        if all(
            "sample_weight" in sample
            for sample in samples
        ):
            batch["sample_weights"] = torch.tensor(
                [
                    float(
                        sample["sample_weight"]
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
            )

        return batch


# ============================================================
# Dataset
# ============================================================

def create_dataset(
    csv_path: Path,
    audio_root: Path,
    speaker_id: str,
    split: str,
    sample_rate: int,
    max_audio_seconds: float,
) -> PersonalizationIEUMDataset:

    return PersonalizationIEUMDataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=speaker_id,
        split=split,
        split_column="personal_split",
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
        load_audio=True,
    )


# ============================================================
# General model prediction
# ============================================================

@torch.no_grad()
def predict_train_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer: CTCCharacterTokenizer,
    device: torch.device,
    original_vocab_size: int,
) -> list[dict[str, Any]]:
    """
    개인화 Fine-tuning 전에 현재 범용 모델로
    화자의 train 데이터 prediction을 생성한다.

    중요
    ----
    개인화 모델 자체는 extended vocab 크기로 생성되어 있지만,
    Error Profile은 '개인화 전 범용 모델의 오류'를 분석하는 것이므로
    prediction에서는 기존 범용 vocab 범위의 logits만 사용한다.

    즉:
        Error Profile prediction → original vocab
        Personalization training → extended vocab
    """

    if original_vocab_size <= 0:
        raise ValueError(
            "original_vocab_size는 1 이상이어야 합니다."
        )

    model.eval()

    results: list[dict[str, Any]] = []

    for batch in data_loader:

        input_features = batch[
            "input_features"
        ].to(
            device,
            non_blocking=True,
        )

        audio_num_samples = batch[
            "audio_num_samples"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # CTCASRModel.forward()
        #
        # 반환:
        #   logits, input_lengths
        # ----------------------------------------------------

        logits, input_lengths = model(
            input_features=input_features,
            audio_num_samples=audio_num_samples,
        )

        if logits.ndim != 3:
            raise RuntimeError(
                "예상하지 못한 logits shape입니다.\n"
                f"logits.shape = {tuple(logits.shape)}"
            )

        if logits.shape[-1] < original_vocab_size:
            raise RuntimeError(
                "모델 classifier 크기가 기존 vocab보다 작습니다.\n"
                f"classifier size   : {logits.shape[-1]}\n"
                f"original vocab   : {original_vocab_size}"
            )

        # ----------------------------------------------------
        # Error Profile 생성에서는 기존 범용 vocab만 사용
        #
        # 개인화 모델:
        #   0 ~ 823     : 범용 모델에서 학습된 weight
        #   824 ~ 1094 : 새로 초기화된 weight
        #
        # 새 classifier row는 아직 학습되지 않았으므로
        # 범용 모델 Error Profile prediction에는 포함하지 않는다.
        # ----------------------------------------------------

        general_logits = logits[
            ...,
            :original_vocab_size,
        ]

        predicted_ids = torch.argmax(
            general_logits,
            dim=-1,
        )

        references = batch[
            "references"
        ]

        if len(references) != predicted_ids.shape[0]:
            raise RuntimeError(
                "Reference 수와 prediction batch 크기가 다릅니다."
            )

        # ----------------------------------------------------
        # CTC decode
        #
        # padding 영역까지 decode하지 않고
        # 실제 encoder output length까지만 사용한다.
        # ----------------------------------------------------

        for index, reference in enumerate(
            references
        ):

            length = int(
                input_lengths[
                    index
                ].item()
            )

            length = max(
                1,
                min(
                    length,
                    predicted_ids.shape[1],
                ),
            )

            token_ids = (
                predicted_ids[
                    index,
                    :length,
                ]
                .detach()
                .cpu()
                .tolist()
            )

            prediction = tokenizer.decode(
                token_ids,
                ctc_decode=True,
            )

            results.append(
                {
                    "reference_text": (
                        reference
                    ),
                    "prediction_text": (
                        prediction
                    ),
                }
            )

    return results


# ============================================================
# Error Profile preparation
# ============================================================

def prepare_error_profile(
    *,
    model: torch.nn.Module,
    train_dataset: PersonalizationIEUMDataset,
    feature_extractor: IEUMWhisperFeatureExtractor,
    tokenizer: CTCCharacterTokenizer,
    device: torch.device,
    original_vocab_size: int,
    supported_reference_syllables: set[str],
    batch_size: int,
    num_workers: int,
    min_count: int,
    min_ratio: float,
    alpha: float,
    max_weight: float | None,
    output_dir: Path,
) -> None:
    """
    train 데이터만 사용하여

    1. 범용 모델 prediction 생성
    2. Raw Error Profile 생성
    3. threshold 적용
    4. train sample별 weight 계산
    5. Dataset에 weight 연결

    을 수행한다.

    valid/test 데이터는 Error Profile 생성에 사용하지 않는다.
    """

    print()
    print("=" * 70)
    print("Error Profile 생성")
    print("=" * 70)

    print(
        f"min_count : {min_count}"
    )

    print(
        f"min_ratio : {min_ratio}"
    )

    print(
        f"alpha     : {alpha}"
    )

    print(
        f"max_weight: {max_weight}"
    )

    # --------------------------------------------------------
    # Prediction loader
    #
    # 반드시 shuffle=False.
    # Dataset index와 prediction 순서를 유지해야
    # weight를 정확한 sample에 연결할 수 있다.
    # --------------------------------------------------------

    prediction_collator = (
        PersonalizationCollator(
            feature_extractor=(
                feature_extractor
            ),
            tokenizer=tokenizer,
        )
    )

    prediction_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=prediction_collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # General model prediction
    # --------------------------------------------------------

    print()
    print(
        "범용 모델로 train 데이터 추론 중..."
    )

    prediction_rows = (
        predict_train_dataset(
            model=model,
            data_loader=prediction_loader,
            tokenizer=tokenizer,
            device=device,
            original_vocab_size=(
                original_vocab_size
            ),
        )
    )

    if (
        len(prediction_rows)
        != len(train_dataset)
    ):
        raise RuntimeError(
            "Train prediction 수와 "
            "Dataset sample 수가 다릅니다.\n"
            f"Predictions: {len(prediction_rows)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    sample_ids = (
        train_dataset.samples[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Prediction 저장
    # --------------------------------------------------------

    prediction_save_rows = []

    for sample_id, row in zip(
        sample_ids,
        prediction_rows,
    ):
        prediction_save_rows.append(
            {
                "sample_id": sample_id,
                "reference_text": (
                    row["reference_text"]
                ),
                "prediction_text": (
                    row["prediction_text"]
                ),
            }
        )

    pd.DataFrame(
        prediction_save_rows
    ).to_csv(
        output_dir
        / "train_general_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Reference / Prediction pairs
    # --------------------------------------------------------

    reference_prediction_pairs = [
        (
            row["reference_text"],
            row["prediction_text"],
        )
        for row in prediction_rows
    ]

    # --------------------------------------------------------
    # Vocabulary unsupported error analysis
    # --------------------------------------------------------

    print(
        "Vocab 미지원 오류 분석 중..."
    )

    (
        unsupported_error_rows,
        vocab_coverage_summary,
    ) = analyze_vocab_unsupported_errors(
        reference_prediction_pairs,
        supported_reference_syllables=(
            supported_reference_syllables
        ),
    )

    # 상세 오류 저장
    pd.DataFrame(
        unsupported_error_rows,
        columns=[
            "error_type",
            "reference_syllable",
            "predicted_syllable",
            "count",

        ],
    ).to_csv(
        output_dir
        / "vocab_unsupported_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 요약 통계 저장
    with (
        output_dir
        / "vocab_coverage_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            vocab_coverage_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"전체 alignment 오류 : "
        f"{vocab_coverage_summary['total_alignment_errors']}"
    )

    print(
        f"분석 가능 오류       : "
        f"{vocab_coverage_summary['speaker_analyzable_errors']}"
    )

    print(
        f"Vocab 미지원 오류    : "
        f"{vocab_coverage_summary['vocab_unsupported_errors']}"
    )

    print(
        f"Vocab 미지원 오류율  : "
        f"{vocab_coverage_summary['vocab_unsupported_error_rate']:.4%}"
    )

    # --------------------------------------------------------
    # Raw Error Profile
    # --------------------------------------------------------

    print(
        "Raw Error Profile 생성 중..."
    )

    raw_profile = (
        build_raw_error_profile(
            reference_prediction_pairs, 
            supported_reference_syllables=(
                supported_reference_syllables
            ),
        )
    )

    raw_profile_df = pd.DataFrame(
        raw_profile
    )

    raw_profile_df.to_csv(
        output_dir
        / "raw_error_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    filtered_profile = (
        filter_error_profile(
            raw_profile,
            min_count=min_count,
            min_ratio=min_ratio,
        )
    )

    filtered_profile_df = pd.DataFrame(
        filtered_profile
    )

    filtered_profile_df.to_csv(
        output_dir
        / "filtered_error_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Raw profile errors      : "
        f"{len(raw_profile)}"
    )

    print(
        f"Filtered profile errors : "
        f"{len(filtered_profile)}"
    )

    # --------------------------------------------------------
    # Sample weights
    # --------------------------------------------------------

    print(
        "Train sample weight 계산 중..."
    )

    weight_rows = (
        calculate_sample_weights(
            reference_prediction_pairs,
            filtered_profile,
            alpha=alpha,
            base_weight=1.0,
            max_weight=max_weight,
        )
    )

    if (
        len(weight_rows)
        != len(train_dataset)
    ):
        raise RuntimeError(
            "계산된 sample weight 수와 "
            "Dataset sample 수가 다릅니다.\n"
            f"Weights: {len(weight_rows)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    sample_weights = [
        float(
            row["sample_weight"]
        )
        for row in weight_rows
    ]

    # --------------------------------------------------------
    # Dataset에 weight 연결
    # --------------------------------------------------------

    train_dataset.set_sample_weights(
        sample_weights
    )

    # --------------------------------------------------------
    # Weight 결과 저장
    # --------------------------------------------------------

    weight_save_rows = []

    for sample_id, row in zip(
        sample_ids,
        weight_rows,
    ):

        weight_save_rows.append(
            {
                "sample_id": (
                    sample_id
                ),
                "reference_text": (
                    row["reference_text"]
                ),
                "prediction_text": (
                    row["prediction_text"]
                ),
                "num_profile_errors": (
                    row[
                        "num_profile_errors"
                    ]
                ),
                "profile_ratio_sum": (
                    row[
                        "profile_ratio_sum"
                    ]
                ),
                "sample_weight": (
                    row[
                        "sample_weight"
                    ]
                ),
                "matched_errors": (
                    json.dumps(
                        row[
                            "matched_errors"
                        ],
                        ensure_ascii=False,
                    )
                ),
            }
        )

    pd.DataFrame(
        weight_save_rows
    ).to_csv(
        output_dir
        / "train_sample_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Weight summary
    # --------------------------------------------------------

    weights_array = np.asarray(
        sample_weights,
        dtype=float,
    )

    weighted_sample_count = int(
        (
            weights_array > 1.0
        ).sum()
    )

    print()
    print("-" * 70)
    print("Error Profile 요약")
    print("-" * 70)

    print(
        f"Train samples    : "
        f"{len(train_dataset)}"
    )

    print(
        f"Profile errors   : "
        f"{len(filtered_profile)}"
    )

    print(
        f"Weighted samples : "
        f"{weighted_sample_count}"
        f" / {len(train_dataset)}"
    )

    print(
        f"Weight min       : "
        f"{weights_array.min():.4f}"
    )

    print(
        f"Weight mean      : "
        f"{weights_array.mean():.4f}"
    )

    print(
        f"Weight max       : "
        f"{weights_array.max():.4f}"
    )

    print("-" * 70)



# ============================================================
# Reuse precomputed Raw Error Profile
# ============================================================

def prepare_error_profile_from_saved(
    *,
    train_dataset: PersonalizationIEUMDataset,
    raw_profile_dir: Path,
    min_count: int,
    min_ratio: float,
    alpha: float,
    max_weight: float | None,
    output_dir: Path,
) -> dict[str, Any]:
    """
    37명 Raw Error Profile 생성 단계에서 저장한 결과를 재사용한다.

    필요한 파일
    -----------
    train_general_predictions.csv
    raw_error_profile.csv

    범용 모델 재추론과 Raw Error Profile 재생성은 수행하지 않는다.
    """

    raw_profile_dir = Path(raw_profile_dir)

    prediction_path = (
        raw_profile_dir
        / "train_general_predictions.csv"
    )
    raw_profile_path = (
        raw_profile_dir
        / "raw_error_profile.csv"
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "train_general_predictions.csv를 찾을 수 없습니다.\n"
            f"{prediction_path}"
        )

    if not raw_profile_path.exists():
        raise FileNotFoundError(
            "raw_error_profile.csv를 찾을 수 없습니다.\n"
            f"{raw_profile_path}"
        )

    print()
    print("=" * 70)
    print("저장된 Raw Error Profile 재사용")
    print("=" * 70)
    print(f"Source    : {raw_profile_dir}")
    print(f"min_count : {min_count}")
    print(f"min_ratio : {min_ratio}")
    print(f"alpha     : {alpha}")
    print(f"max_weight: {max_weight}")

    prediction_df = pd.read_csv(
        prediction_path
    )

    required_prediction_columns = {
        "sample_id",
        "reference_text",
        "prediction_text",
    }
    missing = (
        required_prediction_columns
        - set(prediction_df.columns)
    )
    if missing:
        raise ValueError(
            "train_general_predictions.csv 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    dataset_sample_ids = (
        train_dataset.samples["sample_id"]
        .astype(str)
        .tolist()
    )
    prediction_sample_ids = (
        prediction_df["sample_id"]
        .astype(str)
        .tolist()
    )

    if len(prediction_df) != len(train_dataset):
        raise RuntimeError(
            "저장된 prediction 수와 현재 Train Dataset 크기가 다릅니다.\n"
            f"Predictions: {len(prediction_df)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    if prediction_sample_ids != dataset_sample_ids:
        mismatch_index = next(
            (
                i
                for i, (saved_id, dataset_id)
                in enumerate(
                    zip(
                        prediction_sample_ids,
                        dataset_sample_ids,
                    )
                )
                if saved_id != dataset_id
            ),
            None,
        )
        raise RuntimeError(
            "저장된 prediction의 sample_id 순서와 현재 Dataset 순서가 다릅니다.\n"
            f"First mismatch index: {mismatch_index}"
        )

    prediction_df = prediction_df.fillna("")
    prediction_rows = (
        prediction_df[
            [
                "reference_text",
                "prediction_text",
            ]
        ]
        .to_dict(orient="records")
    )

    raw_profile_df = pd.read_csv(
        raw_profile_path
    )

    required_profile_columns = {
        "error_type",
        "reference_syllable",
        "predicted_syllable",
        "count",
        "reference_count",
        "ratio",
    }
    missing = (
        required_profile_columns
        - set(raw_profile_df.columns)
    )
    if missing:
        raise ValueError(
            "raw_error_profile.csv 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    raw_profile_df = raw_profile_df.fillna(
        {
            "reference_syllable": "",
            "predicted_syllable": "",
        }
    )
    raw_profile = raw_profile_df.to_dict(
        orient="records"
    )

    reference_prediction_pairs = [
        (
            str(row["reference_text"]),
            str(row["prediction_text"]),
        )
        for row in prediction_rows
    ]

    # filter_error_profile()의 현재 규칙:
    # substitution/deletion -> count + ratio
    # insertion             -> count only
    filtered_profile = filter_error_profile(
        raw_profile,
        min_count=min_count,
        min_ratio=min_ratio,
    )

    pd.DataFrame(
        filtered_profile,
        columns=[
            "error_type",
            "reference_syllable",
            "predicted_syllable",
            "count",
            "reference_count",
            "ratio",
        ],
    ).to_csv(
        output_dir
        / "filtered_error_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Raw profile errors      : {len(raw_profile)}"
    )
    print(
        f"Filtered profile errors : {len(filtered_profile)}"
    )

    weight_rows = calculate_sample_weights(
        reference_prediction_pairs,
        filtered_profile,
        alpha=alpha,
        base_weight=1.0,
        max_weight=max_weight,
    )

    if len(weight_rows) != len(train_dataset):
        raise RuntimeError(
            "계산된 sample weight 수와 Dataset 크기가 다릅니다.\n"
            f"Weights: {len(weight_rows)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    sample_weights = [
        float(row["sample_weight"])
        for row in weight_rows
    ]
    train_dataset.set_sample_weights(
        sample_weights
    )

    weight_save_rows = []
    for sample_id, row in zip(
        dataset_sample_ids,
        weight_rows,
    ):
        weight_save_rows.append(
            {
                "sample_id": sample_id,
                "reference_text": row["reference_text"],
                "prediction_text": row["prediction_text"],
                "num_profile_errors": row["num_profile_errors"],
                "profile_ratio_sum": row["profile_ratio_sum"],
                "sample_weight": row["sample_weight"],
                "matched_errors": json.dumps(
                    row["matched_errors"],
                    ensure_ascii=False,
                ),
            }
        )

    pd.DataFrame(
        weight_save_rows
    ).to_csv(
        output_dir
        / "train_sample_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    weights_array = np.asarray(
        sample_weights,
        dtype=float,
    )
    weighted_sample_count = int(
        (weights_array > 1.0).sum()
    )
    weighted_sample_ratio = (
        weighted_sample_count
        / len(train_dataset)
        if len(train_dataset) > 0
        else 0.0
    )

    raw_by_type = (
        raw_profile_df["error_type"]
        .value_counts()
        .to_dict()
    )
    filtered_df = pd.DataFrame(
        filtered_profile
    )
    if filtered_df.empty:
        filtered_by_type = {}
    else:
        filtered_by_type = (
            filtered_df["error_type"]
            .value_counts()
            .to_dict()
        )

    profile_summary = {
        "raw_profile_entries": len(raw_profile),
        "filtered_profile_entries": len(filtered_profile),
        "raw_profile_by_error_type": raw_by_type,
        "filtered_profile_by_error_type": filtered_by_type,
        "train_samples": len(train_dataset),
        "weighted_samples": weighted_sample_count,
        "weighted_sample_ratio": weighted_sample_ratio,
        "weight_min": float(weights_array.min()),
        "weight_mean": float(weights_array.mean()),
        "weight_max": float(weights_array.max()),
        "min_count": min_count,
        "min_ratio": min_ratio,
        "alpha": alpha,
        "max_weight": max_weight,
        "raw_profile_reused": True,
        "raw_profile_source": str(raw_profile_dir),
    }

    with (
        output_dir
        / "error_profile_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            profile_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("-" * 70)
    print("Error Profile 요약")
    print("-" * 70)
    print(f"Train samples    : {len(train_dataset)}")
    print(f"Profile errors   : {len(filtered_profile)}")
    print(
        f"Weighted samples : {weighted_sample_count}"
        f" / {len(train_dataset)}"
        f" ({weighted_sample_ratio:.2%})"
    )
    print(f"Weight min       : {weights_array.min():.4f}")
    print(f"Weight mean      : {weights_array.mean():.4f}")
    print(f"Weight max       : {weights_array.max():.4f}")
    print("Filtered by type :", filtered_by_type)
    print("-" * 70)

    return profile_summary


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "IEUM speaker personalization"
        )
    )

    # --------------------------------------------------------
    # Speaker / data
    # --------------------------------------------------------

    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # General model
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="범용 모델 vocab.json",
    )

    parser.add_argument(
        "--extended_vocab",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # Personalization
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "baseline",
            "error_profile",
        ],
        default="baseline",
    )

    parser.add_argument(
        "--encoder_mode",
        type=str,
        choices=[
            "freeze",
            "last2",
            "last4",
            "full",
        ],
        default="freeze",
    )

    # --------------------------------------------------------
    # Error Profile
    # --------------------------------------------------------

    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help=(
            "Error Profile에 포함할 "
            "최소 오류 반복 횟수. 최종 기본값=5"
        ),
    )

    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.2,
        help=(
            "Error Profile에 포함할 "
            "최소 오류 비율. 최종 기본값=0.2"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Error Profile sample "
            "weight 강도. 최종 기본값=0.5"
        ),
    )

    parser.add_argument(
        "--max_weight",
        type=float,
        default=None,
        help=(
            "sample weight 최대값. "
            "미지정 시 제한 없음"
        ),
    )

    parser.add_argument(
        "--raw_profile_dir",
        type=str,
        default=None,
        help=(
            "미리 생성한 화자별 Raw Error Profile 디렉토리. "
            "지정하면 train prediction과 raw profile을 재사용한다."
        ),
    )

    parser.add_argument(
        "--skip_test",
        action="store_true",
        help=(
            "학습 후 Test 평가를 수행하지 않는다. "
            "Threshold/alpha 선택 실험에서 사용한다."
        ),
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/personalization"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # Argument validation
    # ========================================================

    if args.min_count < 1:
        raise ValueError(
            "--min_count는 1 이상이어야 합니다."
        )

    if not (
        0.0
        <= args.min_ratio
        <= 1.0
    ):
        raise ValueError(
            "--min_ratio는 0~1 사이여야 합니다."
        )

    if args.alpha < 0:
        raise ValueError(
            "--alpha는 0 이상이어야 합니다."
        )

    if (
        args.max_weight is not None
        and args.max_weight < 1.0
    ):
        raise ValueError(
            "--max_weight는 1.0 이상이어야 합니다."
        )

    # ========================================================
    # Seed / Device
    # ========================================================

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("IEUM 개인화 학습")
    print("=" * 70)

    print(
        f"Speaker      : {args.speaker}"
    )

    print(
        f"Mode         : {args.mode}"
    )

    print(
        f"Encoder mode : {args.encoder_mode}"
    )

    print(
        f"Device       : {device}"
    )

    # ========================================================
    # Paths
    # ========================================================

    csv_path = Path(
        args.csv
    )

    audio_root = Path(
        args.audio_root
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    vocab_path = Path(
        args.vocab
    )

    extended_vocab_path = Path(
        args.extended_vocab
    )

    # ========================================================
    # Original vocab size
    #
    # Error Profile 생성에서는 범용 모델이 실제로
    # 학습했던 기존 vocab 범위만 사용한다.
    # ========================================================

    with vocab_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        original_vocab = json.load(
            file
        )

    original_vocab_size = len(
        original_vocab
    )

    # Error Profile에서 분석할 수 있는
    # 기존 범용 vocab의 한글 음절만 추출
    supported_reference_syllables = {
        token
        for token in original_vocab
        if len(token) == 1
        and "\uAC00" <= token <= "\uD7A3"
    }

    print()
    print(
        f"Original vocab size for "
        f"Error Profile: "
        f"{original_vocab_size}"
    )

    # ========================================================
    # General → Personalization model
    # ========================================================

    print()
    print(
        "범용 모델 불러오는 중..."
    )

    model, tokenizer, _ = (
        load_general_model(
            checkpoint_path=(
                checkpoint_path
            ),
            original_vocab_path=(
                vocab_path
            ),
            extended_vocab_path=(
                extended_vocab_path
            ),
            device=device,
            encoder_train_mode=(
                args.encoder_mode
            ),
            sample_rate=(
                args.sample_rate
            ),
        )
    )

    print(
        f"Personalization vocab size: "
        f"{tokenizer.vocab_size}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "화자 Dataset 생성 중..."
    )

    train_dataset = create_dataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=args.speaker,
        split="train",
        sample_rate=args.sample_rate,
        max_audio_seconds=(
            args.max_audio_seconds
        ),
    )

    valid_dataset = create_dataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=args.speaker,
        split="valid",
        sample_rate=args.sample_rate,
        max_audio_seconds=(
            args.max_audio_seconds
        ),
    )

    test_dataset = None

    if not args.skip_test:
        test_dataset = create_dataset(
            csv_path=csv_path,
            audio_root=audio_root,
            speaker_id=args.speaker,
            split="test",
            sample_rate=args.sample_rate,
            max_audio_seconds=(
                args.max_audio_seconds
            ),
        )

    print(
        f"Train chunks : "
        f"{len(train_dataset)}"
    )

    print(
        f"Valid chunks : "
        f"{len(valid_dataset)}"
    )

    if test_dataset is None:
        print("Test chunks  : SKIPPED")
    else:
        print(
            f"Test chunks  : "
            f"{len(test_dataset)}"
        )

    if len(train_dataset) == 0:
        raise ValueError(
            "Train dataset이 비어 있습니다."
        )

    if len(valid_dataset) == 0:
        raise ValueError(
            "Valid dataset이 비어 있습니다."
        )

    if (
        test_dataset is not None
        and len(test_dataset) == 0
    ):
        raise ValueError(
            "Test dataset이 비어 있습니다."
        )

    # ========================================================
    # Feature Extractor / Collator
    # ========================================================

    feature_extractor = (
        IEUMWhisperFeatureExtractor(
            model_name=(
                "openai/whisper-small"
            ),
            sample_rate=(
                args.sample_rate
            ),
            max_audio_seconds=(
                args.max_audio_seconds
            ),
        )
    )

    collator = (
        PersonalizationCollator(
            feature_extractor=(
                feature_extractor
            ),
            tokenizer=tokenizer,
        )
    )

    # ========================================================
    # Output
    # ========================================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_root
        / args.mode
        / args.speaker
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output       : {output_dir}"
    )

    # ========================================================
    # Error Profile
    #
    # 중요:
    # Fine-tuning 전에 실행한다.
    # ========================================================

    if args.mode == "error_profile":

        if args.raw_profile_dir is not None:
            prepare_error_profile_from_saved(
                train_dataset=train_dataset,
                raw_profile_dir=Path(
                    args.raw_profile_dir
                ),
                min_count=args.min_count,
                min_ratio=args.min_ratio,
                alpha=args.alpha,
                max_weight=args.max_weight,
                output_dir=output_dir,
            )

        else:
            prepare_error_profile(
                model=model,
                train_dataset=train_dataset,
                feature_extractor=(
                    feature_extractor
                ),
                tokenizer=tokenizer,
                device=device,
                original_vocab_size=(
                    original_vocab_size
                ),
                supported_reference_syllables=(
                    supported_reference_syllables
                ),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                min_count=args.min_count,
                min_ratio=args.min_ratio,
                alpha=args.alpha,
                max_weight=args.max_weight,
                output_dir=output_dir,
            )

    # ========================================================
    # Tokenizer / Config 저장
    # ========================================================

    tokenizer.save(
        output_dir
        / "extended_vocab.json"
    )

    run_config = vars(
        args
    ).copy()

    with (
        output_dir
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            run_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # DataLoader
    #
    # Error Profile mode라면 이 시점에는
    # train_dataset에 sample_weight가 이미 들어 있다.
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = None

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=(
                device.type == "cuda"
            ),
        )

    # ========================================================
    # Trainable parameters
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print()
    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # ========================================================
    # Optimizer
    #
    # Error Profile prediction은 optimizer 생성 전에
    # 수행했기 때문에 모델 parameter는 변경되지 않는다.
    # ========================================================

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = PersonalizationTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        mode=args.mode,
        use_amp=True,
        gradient_clip_norm=1.0,
    )

    # ========================================================
    # Resume key
    # ========================================================

    if args.mode == "error_profile":
        resume_key = (
            f"mc{args.min_count}"
            f"_mr{args.min_ratio}"
            f"_a{args.alpha}"
            f"_mw{args.max_weight}"
            f"_enc{args.encoder_mode}"
            f"_lr{args.lr}"
        )
    else:
        resume_key = (
            f"baseline"
            f"_enc{args.encoder_mode}"
            f"_lr{args.lr}"
        )

    # ========================================================
    # Train
    # ========================================================

    summary = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        output_dir=output_dir,
        early_stopping_patience=(
            args.patience
        ),
        resume_key=resume_key,
    )

    # ========================================================
    # Validation 결과 저장
    #
    # Threshold / alpha 탐색에서는 Test 결과를 사용하지 않는다.
    # ========================================================

    validation_summary = {
        "speaker_id": args.speaker,
        "mode": args.mode,
        "best_epoch": summary["best_epoch"],
        "best_valid_cer": summary["best_valid_cer"],
        "min_count": (
            args.min_count
            if args.mode == "error_profile"
            else None
        ),
        "min_ratio": (
            args.min_ratio
            if args.mode == "error_profile"
            else None
        ),
        "alpha": (
            args.alpha
            if args.mode == "error_profile"
            else None
        ),
        "max_weight": (
            args.max_weight
            if args.mode == "error_profile"
            else None
        ),
        "encoder_mode": args.encoder_mode,
        "lr": args.lr,
        "skip_test": args.skip_test,
    }

    with (
        output_dir
        / "validation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    if args.skip_test:
        print()
        print("=" * 70)
        print("개인화 학습 완료 - Test 평가 생략")
        print("=" * 70)
        print(f"Speaker    : {args.speaker}")
        print(f"Mode       : {args.mode}")
        print(
            f"Best epoch : "
            f"{summary['best_epoch']}"
        )
        print(
            f"Valid CER  : "
            f"{summary['best_valid_cer']:.4f}"
        )
        print("Test       : SKIPPED")
        print(f"Results    : {output_dir}")
        return

    # ========================================================
    # Best checkpoint 다시 로드
    # ========================================================

    best_model_path = (
        output_dir
        / "best_model.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            "개인화 best_model.pt가 "
            "생성되지 않았습니다."
        )

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # Test
    # ========================================================

    print()
    print("=" * 70)
    print("Best model Test 평가")
    print("=" * 70)

    if test_loader is None:
        raise RuntimeError(
            "Test 평가가 요청되었지만 test_loader가 없습니다."
        )

    test_result = trainer.evaluate(
        test_loader
    )

    print(
        f"Test CER: "
        f"{test_result['cer']:.4f}"
    )

    print(
        f"Test WER: "
        f"{test_result['wer']:.4f}"
    )

    # ========================================================
    # Test 결과 저장
    # ========================================================

    test_summary = {
        "speaker_id": args.speaker,
        "mode": args.mode,
        "best_epoch": (
            summary["best_epoch"]
        ),
        "best_valid_cer": (
            summary["best_valid_cer"]
        ),
        "test_loss": (
            test_result["loss"]
        ),
        "test_cer": (
            test_result["cer"]
        ),
        "test_wer": (
            test_result["wer"]
        ),
    }

    # Error Profile mode에서는
    # 실험 파라미터도 결과에 함께 기록한다.
    if args.mode == "error_profile":

        test_summary.update(
            {
                "min_count": (
                    args.min_count
                ),
                "min_ratio": (
                    args.min_ratio
                ),
                "alpha": (
                    args.alpha
                ),
                "max_weight": (
                    args.max_weight
                ),
            }
        )

    with (
        output_dir
        / "test_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        {
            "reference": (
                test_result[
                    "references"
                ]
            ),
            "prediction": (
                test_result[
                    "predictions"
                ]
            ),
        }
    ).to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 70)
    print("개인화 학습 및 평가 완료")
    print("=" * 70)

    print(
        f"Speaker : {args.speaker}"
    )

    print(
        f"Mode    : {args.mode}"
    )

    print(
        f"Best epoch : "
        f"{summary['best_epoch']}"
    )

    print(
        f"Valid CER  : "
        f"{summary['best_valid_cer']:.4f}"
    )

    print(
        f"Test CER   : "
        f"{test_result['cer']:.4f}"
    )

    print(
        f"Test WER   : "
        f"{test_result['wer']:.4f}"
    )

    print(
        f"Results    : "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()