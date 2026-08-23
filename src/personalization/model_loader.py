from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.asr.encoder import WhisperEncoder
from src.asr.models import BiGRUCTC
from src.asr.tokenizer import CTCCharacterTokenizer
from src.asr.trainer import CTCASRModel


def load_general_model(
    checkpoint_path: str | Path,
    original_vocab_path: str | Path,
    extended_vocab_path: str | Path,
    *,
    device: torch.device,
    model_name: str = "openai/whisper-small",
    encoder_train_mode: str = "freeze",
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    sample_rate: int = 16000,
) -> tuple[
    CTCASRModel,
    CTCCharacterTokenizer,
    dict[str, Any],
]:
    """
    범용 IEUM ASR 모델을 개인화 학습의 초기 모델로 불러온다.

    기존 범용 모델은 original_vocab을 기준으로 학습되었고,
    개인화 모델은 extended_vocab을 사용한다.

    따라서:
        1. Whisper Encoder / BiGRU 가중치는 그대로 불러온다.
        2. 기존 classifier의 출력 가중치는 그대로 유지한다.
        3. 새로 추가된 vocabulary 출력 노드만 새로 초기화한다.

    구조:
        Whisper Encoder
            ↓
        BiGRU
            ↓
        확장된 CTC classifier

    Parameters
    ----------
    checkpoint_path:
        범용 모델의 best_model.pt 경로.

    original_vocab_path:
        범용 모델 학습 당시 사용한 기존 vocab.json 경로.

    extended_vocab_path:
        개인화 데이터의 새로운 문자를 추가한
        extended_vocab.json 경로.

    encoder_train_mode:
        개인화 학습에서 Whisper Encoder의
        어느 부분을 학습할지 지정한다.

        freeze / last2 / last4 / full
    """

    checkpoint_path = Path(checkpoint_path)
    original_vocab_path = Path(original_vocab_path)
    extended_vocab_path = Path(extended_vocab_path)

    # ========================================================
    # 파일 존재 여부 확인
    # ========================================================

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "범용 모델 checkpoint를 찾을 수 없습니다.\n"
            f"경로: {checkpoint_path}"
        )

    if not original_vocab_path.exists():
        raise FileNotFoundError(
            "기존 vocabulary를 찾을 수 없습니다.\n"
            f"경로: {original_vocab_path}"
        )

    if not extended_vocab_path.exists():
        raise FileNotFoundError(
            "확장 vocabulary를 찾을 수 없습니다.\n"
            f"경로: {extended_vocab_path}"
        )

    # ========================================================
    # Original / Extended Vocabulary 불러오기
    # ========================================================

    with open(
        original_vocab_path,
        "r",
        encoding="utf-8",
    ) as f:
        original_vocab = json.load(f)

    with open(
        extended_vocab_path,
        "r",
        encoding="utf-8",
    ) as f:
        extended_vocab = json.load(f)

    original_vocab_size = len(original_vocab)
    extended_vocab_size = len(extended_vocab)

    # ========================================================
    # 기존 vocab ID가 유지됐는지 확인
    # ========================================================

    for token, original_id in original_vocab.items():
        if token not in extended_vocab:
            raise ValueError(
                "extended_vocab에 기존 token이 없습니다.\n"
                f"token: {token}"
            )

        if extended_vocab[token] != original_id:
            raise ValueError(
                "extended_vocab에서 기존 token ID가 변경되었습니다.\n"
                f"token: {token}\n"
                f"original ID: {original_id}\n"
                f"extended ID: {extended_vocab[token]}"
            )

    if extended_vocab_size < original_vocab_size:
        raise ValueError(
            "extended_vocab의 크기가 기존 vocab보다 작습니다."
        )

    # 개인화 학습에서 사용할 tokenizer
    tokenizer = CTCCharacterTokenizer.load(
        extended_vocab_path
    )

    # ========================================================
    # Whisper Encoder
    # ========================================================

    encoder = WhisperEncoder(
        model_name=model_name,
        train_mode=encoder_train_mode,
    )

    # ========================================================
    # 확장된 BiGRU CTC
    #
    # 출력 크기를 extended_vocab 기준으로 생성한다.
    # ========================================================

    downstream_model = BiGRUCTC(
        input_dim=encoder.hidden_size,
        vocab_size=extended_vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ========================================================
    # 전체 ASR Model
    # ========================================================

    model = CTCASRModel(
        encoder=encoder,
        downstream_model=downstream_model,
        sample_rate=sample_rate,
    )

    # ========================================================
    # 범용 checkpoint 불러오기
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "checkpoint에 model_state_dict가 없습니다."
        )

    general_state_dict = checkpoint[
        "model_state_dict"
    ]

    # ========================================================
    # Classifier key 확인
    # ========================================================

    classifier_weight_key = (
        "downstream_model.classifier.weight"
    )

    classifier_bias_key = (
        "downstream_model.classifier.bias"
    )

    if classifier_weight_key not in general_state_dict:
        raise KeyError(
            "checkpoint에서 classifier weight를 찾을 수 없습니다.\n"
            f"예상 key: {classifier_weight_key}"
        )

    if classifier_bias_key not in general_state_dict:
        raise KeyError(
            "checkpoint에서 classifier bias를 찾을 수 없습니다.\n"
            f"예상 key: {classifier_bias_key}"
        )

    old_classifier_weight = general_state_dict[
        classifier_weight_key
    ]

    old_classifier_bias = general_state_dict[
        classifier_bias_key
    ]

    # ========================================================
    # 기존 classifier 크기 검증
    # ========================================================

    checkpoint_vocab_size = (
        old_classifier_weight.shape[0]
    )

    if checkpoint_vocab_size != original_vocab_size:
        raise ValueError(
            "checkpoint의 classifier 크기와 "
            "original_vocab 크기가 일치하지 않습니다.\n"
            f"checkpoint classifier: {checkpoint_vocab_size}\n"
            f"original vocab: {original_vocab_size}"
        )

    # ========================================================
    # Encoder + BiGRU 가중치 불러오기
    #
    # classifier는 크기가 달라졌기 때문에 여기서는 제외한다.
    # ========================================================

    state_dict_without_classifier = {
        key: value
        for key, value in general_state_dict.items()
        if key not in {
            classifier_weight_key,
            classifier_bias_key,
        }
    }

    load_result = model.load_state_dict(
        state_dict_without_classifier,
        strict=False,
    )

    # classifier 두 개만 missing이어야 정상
    allowed_missing_keys = {
        classifier_weight_key,
        classifier_bias_key,
    }

    unexpected_missing_keys = (
        set(load_result.missing_keys)
        - allowed_missing_keys
    )

    if unexpected_missing_keys:
        raise RuntimeError(
            "예상하지 못한 model parameter가 "
            "checkpoint에서 누락되었습니다.\n"
            f"missing keys: {sorted(unexpected_missing_keys)}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "checkpoint에 현재 모델 구조에 없는 "
            "parameter가 있습니다.\n"
            f"unexpected keys: "
            f"{sorted(load_result.unexpected_keys)}"
        )

    # ========================================================
    # 기존 classifier weight / bias 복사
    #
    # 0 ~ original_vocab_size-1
    # → 기존 범용 모델의 학습된 값을 그대로 사용
    #
    # original_vocab_size ~ extended_vocab_size-1
    # → 새 Linear layer 생성 시 초기화된 값을 그대로 사용
    # ========================================================

    with torch.no_grad():
        model.downstream_model.classifier.weight[
            :original_vocab_size
        ].copy_(
            old_classifier_weight
        )

        model.downstream_model.classifier.bias[
            :original_vocab_size
        ].copy_(
            old_classifier_bias
        )

    # ========================================================
    # Device 이동
    # ========================================================

    model = model.to(device)

    # ========================================================
    # 정보 출력
    # ========================================================

    added_vocab_size = (
        extended_vocab_size
        - original_vocab_size
    )

    print("=" * 70)
    print("범용 모델 → 개인화 모델 로드 완료")
    print("=" * 70)
    print(
        f"기존 vocab 크기      : "
        f"{original_vocab_size}"
    )
    print(
        f"확장 vocab 크기      : "
        f"{extended_vocab_size}"
    )
    print(
        f"새로 추가된 문자 수  : "
        f"{added_vocab_size}"
    )
    print()
    print(
        f"기존 classifier 출력 : "
        f"0 ~ {original_vocab_size - 1}"
        " → 범용 weight 유지"
    )

    if added_vocab_size > 0:
        print(
            f"새 classifier 출력   : "
            f"{original_vocab_size} ~ "
            f"{extended_vocab_size - 1}"
            " → 새로 초기화"
        )

    return (
        model,
        tokenizer,
        checkpoint,
    )

# ============================================================
# Final personalized model loader
# ============================================================

def load_personalized_model(
    checkpoint_path: str | Path,
    extended_vocab_path: str | Path,
    *,
    device: torch.device,
    model_name: str = "openai/whisper-small",
    encoder_train_mode: str = "freeze",
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    sample_rate: int = 16000,
    run_config_path: str | Path | None = None,
) -> tuple[
    CTCASRModel,
    CTCCharacterTokenizer,
    dict[str, Any],
]:
    """
    학습이 완료된 화자별 개인화 best_model.pt를 추론용으로 로드한다.

    중요
    ----
    load_general_model()은
        범용 824-vocab checkpoint
            -> 1095-vocab 개인화 초기 모델
    을 만드는 '개인화 학습 시작용' 함수다.

    이 함수는 이미 개인화 학습이 끝난 checkpoint를 사용하므로,
    extended_vocab 크기의 모델을 만든 뒤
    checkpoint["model_state_dict"] 전체를 strict=True로 로드한다.

    최종 개인화 모델 구조
    ---------------------
    Whisper Small Encoder
        -> BiGRU CTC
        -> extended vocab classifier

    Parameters
    ----------
    checkpoint_path:
        화자별 최종 best_model.pt 경로.

    extended_vocab_path:
        개인화 학습에 사용한 models/final/extended_vocab.json 경로.

    device:
        추론 device.

    encoder_train_mode:
        최종 개인화 학습은 freeze를 사용했으므로 기본값은 "freeze".
        추론에서는 requires_grad 자체는 출력에 영향을 주지 않지만,
        학습 당시와 동일한 구조 설정을 명시적으로 유지한다.

    run_config_path:
        선택 사항. 지정하면 run_config.json을 읽어
        encoder_mode가 현재 loader 설정과 일치하는지 검증한다.
    """

    checkpoint_path = Path(checkpoint_path)
    extended_vocab_path = Path(extended_vocab_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "개인화 checkpoint를 찾을 수 없습니다.\n"
            f"경로: {checkpoint_path}"
        )

    if not extended_vocab_path.exists():
        raise FileNotFoundError(
            "확장 vocabulary를 찾을 수 없습니다.\n"
            f"경로: {extended_vocab_path}"
        )

    # --------------------------------------------------------
    # 선택적 run_config 검증
    # --------------------------------------------------------

    if run_config_path is not None:
        run_config_path = Path(run_config_path)

        if not run_config_path.exists():
            raise FileNotFoundError(
                "run_config.json을 찾을 수 없습니다.\n"
                f"경로: {run_config_path}"
            )

        with run_config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            run_config = json.load(f)

        saved_encoder_mode = run_config.get(
            "encoder_mode"
        )

        if (
            saved_encoder_mode is not None
            and saved_encoder_mode != encoder_train_mode
        ):
            raise ValueError(
                "run_config의 encoder_mode와 loader 설정이 다릅니다.\n"
                f"run_config: {saved_encoder_mode}\n"
                f"loader: {encoder_train_mode}"
            )

    # --------------------------------------------------------
    # Extended tokenizer
    # --------------------------------------------------------

    tokenizer = CTCCharacterTokenizer.load(
        extended_vocab_path
    )

    extended_vocab_size = tokenizer.vocab_size

    # --------------------------------------------------------
    # 학습 당시와 동일한 모델 구조 생성
    # --------------------------------------------------------

    encoder = WhisperEncoder(
        model_name=model_name,
        train_mode=encoder_train_mode,
    )

    downstream_model = BiGRUCTC(
        input_dim=encoder.hidden_size,
        vocab_size=extended_vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    model = CTCASRModel(
        encoder=encoder,
        downstream_model=downstream_model,
        sample_rate=sample_rate,
    )

    # --------------------------------------------------------
    # 개인화 checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "개인화 checkpoint에 model_state_dict가 없습니다."
        )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    classifier_weight_key = (
        "downstream_model.classifier.weight"
    )
    classifier_bias_key = (
        "downstream_model.classifier.bias"
    )

    if classifier_weight_key not in state_dict:
        raise KeyError(
            "개인화 checkpoint에서 classifier weight를 찾을 수 없습니다.\n"
            f"예상 key: {classifier_weight_key}"
        )

    if classifier_bias_key not in state_dict:
        raise KeyError(
            "개인화 checkpoint에서 classifier bias를 찾을 수 없습니다.\n"
            f"예상 key: {classifier_bias_key}"
        )

    checkpoint_vocab_size = int(
        state_dict[
            classifier_weight_key
        ].shape[0]
    )

    if checkpoint_vocab_size != extended_vocab_size:
        raise ValueError(
            "개인화 checkpoint classifier 크기와 "
            "extended_vocab 크기가 일치하지 않습니다.\n"
            f"checkpoint classifier: {checkpoint_vocab_size}\n"
            f"extended vocab: {extended_vocab_size}"
        )

    # 개인화 checkpoint는 이미 extended vocab 전체가 학습된
    # 최종 모델이므로 state_dict 전체를 엄격하게 로드한다.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("=" * 70)
    print("IEUM 개인화 ASR 모델 로드 완료")
    print("=" * 70)
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Extended vocab  : {extended_vocab_path}")
    print(f"Vocabulary size : {extended_vocab_size}")
    print(
        f"Best Epoch      : "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(
        f"Validation CER  : "
        f"{checkpoint.get('valid_cer', 'unknown')}"
    )
    print(
        f"Validation WER  : "
        f"{checkpoint.get('valid_wer', 'unknown')}"
    )
    print("=" * 70)

    return (
        model,
        tokenizer,
        checkpoint,
    )

