from pathlib import Path
from typing import Any, Optional


class ModelLoader:
    """
    IEUM ASR 모델 로더

    역할:
    - 최종 학습 모델을 서버 시작 시 한 번만 로드
    - 로드된 모델/processor를 추론 코드에서 재사용

    현재는 최종 모델 확정 전이므로
    실제 checkpoint 로딩은 구현하지 않음.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
    ):
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        self.model: Any = None
        self.processor: Any = None

        self.is_loaded = False

    def load(self) -> None:
        """
        최종 ASR 모델을 로드한다.

        TODO:
        최종 모델 확정 후 아래 내용을 구현

        예:
        1. Whisper processor 로드
        2. Whisper encoder 로드
        3. downstream model 생성
        4. checkpoint 불러오기
        5. model.eval()
        6. device 이동
        """

        if self.is_loaded:
            return

        # 아직 실제 모델이 없으므로
        # 로드 완료 상태만 설정
        self.is_loaded = True

    def get_model(self) -> Any:
        """
        로드된 ASR 모델 반환
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model has not been loaded."
            )

        return self.model

    def get_processor(self) -> Any:
        """
        로드된 processor 반환
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Processor has not been loaded."
            )

        return self.processor