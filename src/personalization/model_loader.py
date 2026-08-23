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
model_loader.py
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

