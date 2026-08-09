from __future__ import annotations

from pathlib import Path

import torch

from src.asr.encoder import WhisperEncoder
from src.asr.models.bigru_ctc import BiGRUCTC
from src.asr.tokenizer import CTCCharacterTokenizer
from src.asr.trainer import CTCASRModel


class ModelLoader:
    """
    IEUM 최종 범용 ASR 모델 로더.

    최종 구조:
        Whisper Small Encoder
        (Last 4 Layers Fine-tuning)
            ↓
        BiGRU
            ↓
        CTC

    서버 시작 시 모델과 tokenizer를 한 번만 로드하고
    이후 모든 추론 요청에서 재사용한다.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        whisper_model_name: str = "openai/whisper-small",
        device: str | None = None,
    ) -> None:

        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)
        self.whisper_model_name = whisper_model_name

        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model: CTCASRModel | None = None
        self.tokenizer: CTCCharacterTokenizer | None = None

        self.is_loaded = False

    def load(self) -> None:
        """
        최종 checkpoint와 vocabulary를 로드한다.
        """

        if self.is_loaded:
            return

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

        print("=" * 70)
        print("IEUM ASR 모델 로딩")
        print("=" * 70)

        print(f"Device: {self.device}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Vocabulary: {self.vocab_path}")

        # ----------------------------------------------------
        # 1. CTC Vocabulary
        # ----------------------------------------------------

        self.tokenizer = CTCCharacterTokenizer.load(
            self.vocab_path
        )

        print(
            f"Vocabulary size: "
            f"{self.tokenizer.vocab_size}"
        )

        # ----------------------------------------------------
        # 2. Whisper Encoder
        #
        # 학습 당시와 동일하게 Last4 구조로 생성
        # ----------------------------------------------------

        encoder = WhisperEncoder(
            model_name=self.whisper_model_name,
            train_mode="last4",
        )

        # ----------------------------------------------------
        # 3. BiGRU CTC
        # ----------------------------------------------------

        downstream_model = BiGRUCTC(
            input_dim=encoder.hidden_size,
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=512,
            num_layers=2,
            dropout=0.1,
        )

        # ----------------------------------------------------
        # 4. Whisper Encoder + BiGRU 결합
        # ----------------------------------------------------

        model = CTCASRModel(
            encoder=encoder,
            downstream_model=downstream_model,
            sample_rate=16000,
        )

        # ----------------------------------------------------
        # 5. 최종 checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint에 model_state_dict가 없습니다."
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------------------
        # 6. 추론 모드
        # ----------------------------------------------------

        model.to(self.device)
        model.eval()

        self.model = model

        self.is_loaded = True

        print(
            f"Best Epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"Validation CER: "
            f"{checkpoint.get('valid_cer', 'unknown')}"
        )

        print(
            f"Validation WER: "
            f"{checkpoint.get('valid_wer', 'unknown')}"
        )

        print("=" * 70)
        print("IEUM ASR 모델 로딩 완료")
        print("=" * 70)

    def get_model(self) -> CTCASRModel:
        """
        로드된 최종 ASR 모델 반환.
        """

        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "ASR 모델이 아직 로드되지 않았습니다."
            )

        return self.model

    def get_tokenizer(self) -> CTCCharacterTokenizer:
        """
        로드된 CTC tokenizer 반환.
        """

        if not self.is_loaded or self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer가 아직 로드되지 않았습니다."
            )

        return self.tokenizer

    def get_device(self) -> torch.device:
        return self.device