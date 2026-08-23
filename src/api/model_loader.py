from __future__ import annotations

import gc
from pathlib import Path

import torch

from src.asr.encoder import WhisperEncoder
from src.asr.models.bigru_ctc import BiGRUCTC
from src.asr.tokenizer import CTCCharacterTokenizer
from src.asr.trainer import CTCASRModel

from src.personalization.model_loader import (
    load_personalized_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ============================================================
# Demo personalized speakers
# ============================================================

SUPPORTED_PERSONALIZED_SPEAKERS = (
    "HYH_M_22",
    "SKY_M_24",
)


class ModelLoader:

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        whisper_model_name: str = "openai/whisper-small",
        device: str | None = None,
        personalized_root: str | Path | None = None,
        extended_vocab_path: str | Path | None = None,
    ) -> None:

        # ====================================================
        # General model paths
        # ====================================================

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.vocab_path = Path(
            vocab_path
        )

        self.whisper_model_name = (
            whisper_model_name
        )

        # ====================================================
        # Device
        # ====================================================

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # ====================================================
        # Personalized model paths
        # ====================================================

        if personalized_root is None:

            self.personalized_root = (
                PROJECT_ROOT
                / "models"
                / "personalized"
            )

        else:

            self.personalized_root = Path(
                personalized_root
            )

        if extended_vocab_path is None:

            self.extended_vocab_path = (
                PROJECT_ROOT
                / "models"
                / "final"
                / "extended_vocab.json"
            )

        else:

            self.extended_vocab_path = Path(
                extended_vocab_path
            )

        # ====================================================
        # General model
        # ====================================================

        self.model: CTCASRModel | None = None

        self.tokenizer: (
            CTCCharacterTokenizer | None
        ) = None

        self.is_loaded = False

        # ====================================================
        # Personalized model cache
        #
        # 메모리 사용량을 줄이기 위해
        # 개인화 모델은 한 명만 메모리에 유지한다.
        # ====================================================

        self.personalized_model: (
            CTCASRModel | None
        ) = None

        self.personalized_tokenizer: (
            CTCCharacterTokenizer | None
        ) = None

        self.personalized_checkpoint: (
            dict | None
        ) = None

        self.loaded_personalized_speaker: (
            str | None
        ) = None


    # ========================================================
    # General model
    # ========================================================

    def load(self) -> None:
        """
        최종 범용 모델과 vocabulary를 로드한다.

        서버 시작 시 한 번만 호출하고
        이후 요청에서는 동일 모델을 재사용한다.
        """

        if self.is_loaded:
            return

        # ----------------------------------------------------
        # Path validation
        # ----------------------------------------------------

        if not self.checkpoint_path.exists():

            raise FileNotFoundError(
                "Checkpoint 파일을 찾을 수 없습니다.\n"
                f"경로: {self.checkpoint_path}"
            )

        if not self.vocab_path.exists():

            raise FileNotFoundError(
                "Vocabulary 파일을 찾을 수 없습니다.\n"
                f"경로: {self.vocab_path}"
            )

        print()
        print("=" * 70)
        print("IEUM 범용 ASR 모델 로딩")
        print("=" * 70)

        print(
            f"Device     : "
            f"{self.device}"
        )

        print(
            f"Checkpoint : "
            f"{self.checkpoint_path}"
        )

        print(
            f"Vocabulary : "
            f"{self.vocab_path}"
        )

        # ----------------------------------------------------
        # 1. CTC Vocabulary
        # ----------------------------------------------------

        self.tokenizer = (
            CTCCharacterTokenizer.load(
                self.vocab_path
            )
        )

        print(
            f"Vocabulary size: "
            f"{self.tokenizer.vocab_size}"
        )

        # ----------------------------------------------------
        # 2. Whisper Encoder
        #
        # 최종 범용 모델은 Last4 Fine-tuning 모델
        # ----------------------------------------------------

        encoder = WhisperEncoder(
            model_name=(
                self.whisper_model_name
            ),
            train_mode="last4",
        )

        # ----------------------------------------------------
        # 3. BiGRU CTC
        # ----------------------------------------------------

        downstream_model = BiGRUCTC(
            input_dim=(
                encoder.hidden_size
            ),
            vocab_size=(
                self.tokenizer.vocab_size
            ),
            hidden_size=512,
            num_layers=2,
            dropout=0.1,
        )

        # ----------------------------------------------------
        # 4. Encoder + downstream
        # ----------------------------------------------------

        model = CTCASRModel(
            encoder=encoder,
            downstream_model=(
                downstream_model
            ),
            sample_rate=16000,
        )

        # ----------------------------------------------------
        # 5. Final checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if (
            "model_state_dict"
            not in checkpoint
        ):

            raise KeyError(
                "Checkpoint에 "
                "model_state_dict가 없습니다."
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        # ----------------------------------------------------
        # 6. Inference mode
        # ----------------------------------------------------

        model.to(
            self.device
        )

        model.eval()

        self.model = model

        self.is_loaded = True

        print(
            f"Best Epoch     : "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Validation CER : "
            f"{checkpoint.get('valid_cer', 'unknown')}"
        )

        print(
            f"Validation WER : "
            f"{checkpoint.get('valid_wer', 'unknown')}"
        )

        print("=" * 70)
        print(
            "IEUM 범용 ASR 모델 로딩 완료"
        )
        print("=" * 70)


    def get_model(
        self,
    ) -> CTCASRModel:
        """
        범용 ASR 모델 반환.

        기존 inference.py와의 호환성을 위해
        기존 메서드 이름을 유지한다.
        """

        if (
            not self.is_loaded
            or self.model is None
        ):

            raise RuntimeError(
                "범용 ASR 모델이 "
                "아직 로드되지 않았습니다."
            )

        return self.model


    def get_tokenizer(
        self,
    ) -> CTCCharacterTokenizer:
        """
        범용 CTC tokenizer 반환.
        """

        if (
            not self.is_loaded
            or self.tokenizer is None
        ):

            raise RuntimeError(
                "범용 Tokenizer가 "
                "아직 로드되지 않았습니다."
            )

        return self.tokenizer


    # ========================================================
    # Personalized model
    # ========================================================

    def get_supported_personalized_speakers(
        self,
    ) -> list[str]:
        """
        FastAPI에서 현재 사용할 수 있는
        개인화 화자 목록 반환.
        """

        return list(
            SUPPORTED_PERSONALIZED_SPEAKERS
        )


    def _validate_personalized_speaker(
        self,
        speaker_id: str,
    ) -> None:
        """
        지원 화자 및 필수 파일 확인.
        """

        if (
            speaker_id
            not in SUPPORTED_PERSONALIZED_SPEAKERS
        ):

            raise ValueError(
                "지원하지 않는 개인화 화자입니다.\n"
                f"요청 화자: {speaker_id}\n"
                "지원 화자: "
                f"{list(SUPPORTED_PERSONALIZED_SPEAKERS)}"
            )

        speaker_dir = (
            self.personalized_root
            / speaker_id
        )

        checkpoint_path = (
            speaker_dir
            / "best_model.pt"
        )

        run_config_path = (
            speaker_dir
            / "run_config.json"
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "개인화 best_model.pt를 "
                "찾을 수 없습니다.\n"
                f"Speaker: {speaker_id}\n"
                f"경로: {checkpoint_path}"
            )

        if not run_config_path.exists():

            raise FileNotFoundError(
                "개인화 run_config.json을 "
                "찾을 수 없습니다.\n"
                f"Speaker: {speaker_id}\n"
                f"경로: {run_config_path}"
            )

        if not self.extended_vocab_path.exists():

            raise FileNotFoundError(
                "extended_vocab.json을 "
                "찾을 수 없습니다.\n"
                f"경로: "
                f"{self.extended_vocab_path}"
            )


    def load_personalized(
        self,
        speaker_id: str,
    ) -> None:
        """
        요청된 화자의 개인화 모델을 로드한다.

        동일 화자가 이미 메모리에 있으면
        checkpoint를 다시 읽지 않는다.

        다른 화자가 요청되면 기존 개인화 모델을
        메모리에서 제거하고 새 모델을 로드한다.
        """

        self._validate_personalized_speaker(
            speaker_id
        )

        # ----------------------------------------------------
        # Already cached
        # ----------------------------------------------------

        if (
            self.loaded_personalized_speaker
            == speaker_id
            and self.personalized_model
            is not None
            and self.personalized_tokenizer
            is not None
        ):

            print(
                f"[Personalized Cache] "
                f"{speaker_id} 모델 재사용"
            )

            return

        # ----------------------------------------------------
        # 다른 speaker model이 메모리에 있으면 제거
        # ----------------------------------------------------

        if (
            self.personalized_model
            is not None
        ):

            self.unload_personalized()

        speaker_dir = (
            self.personalized_root
            / speaker_id
        )

        checkpoint_path = (
            speaker_dir
            / "best_model.pt"
        )

        run_config_path = (
            speaker_dir
            / "run_config.json"
        )

        print()
        print("=" * 70)
        print(
            "IEUM 개인화 모델 로딩"
        )
        print("=" * 70)

        print(
            f"Speaker    : "
            f"{speaker_id}"
        )

        print(
            f"Checkpoint : "
            f"{checkpoint_path}"
        )

        # ----------------------------------------------------
        # src/personalization/model_loader.py에서
        # 이미 검증한 최종 개인화 모델 로더 재사용
        # ----------------------------------------------------

        (
            model,
            tokenizer,
            checkpoint,
        ) = load_personalized_model(
            checkpoint_path=(
                checkpoint_path
            ),
            extended_vocab_path=(
                self.extended_vocab_path
            ),
            run_config_path=(
                run_config_path
            ),
            device=self.device,

            # 최종 개인화 실험 설정
            encoder_train_mode="freeze",

            model_name=(
                self.whisper_model_name
            ),

            hidden_size=512,
            num_layers=2,
            dropout=0.1,
            sample_rate=16000,
        )

        self.personalized_model = (
            model
        )

        self.personalized_tokenizer = (
            tokenizer
        )

        self.personalized_checkpoint = (
            checkpoint
        )

        self.loaded_personalized_speaker = (
            speaker_id
        )

        print(
            f"✅ 개인화 모델 준비 완료: "
            f"{speaker_id}"
        )


    def get_personalized_model(
        self,
        speaker_id: str,
    ) -> CTCASRModel:
        """
        특정 화자의 개인화 모델 반환.

        아직 로드되지 않았다면 자동으로 로드한다.
        """

        self.load_personalized(
            speaker_id
        )

        if (
            self.personalized_model
            is None
        ):

            raise RuntimeError(
                "개인화 모델 로딩에 "
                "실패했습니다."
            )

        return (
            self.personalized_model
        )


    def get_personalized_tokenizer(
        self,
        speaker_id: str,
    ) -> CTCCharacterTokenizer:
        """
        특정 화자의 extended-vocab tokenizer 반환.
        """

        self.load_personalized(
            speaker_id
        )

        if (
            self.personalized_tokenizer
            is None
        ):

            raise RuntimeError(
                "개인화 tokenizer 로딩에 "
                "실패했습니다."
            )

        return (
            self.personalized_tokenizer
        )


    def unload_personalized(
        self,
    ) -> None:
        """
        현재 메모리에 올라온 개인화 모델 제거.

        범용 모델은 유지한다.
        """

        if (
            self.personalized_model
            is None
        ):

            self.loaded_personalized_speaker = (
                None
            )

            return

        old_speaker = (
            self.loaded_personalized_speaker
        )

        self.personalized_model = None
        self.personalized_tokenizer = None
        self.personalized_checkpoint = None
        self.loaded_personalized_speaker = None

        # Python object 정리
        gc.collect()

        # GPU 사용 환경일 경우 cache 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[Personalized Cache] "
            f"{old_speaker} 모델 해제"
        )


    # ========================================================
    # Common
    # ========================================================

    def get_device(
        self,
    ) -> torch.device:
        """
        현재 추론 device 반환.
        """

        return self.device