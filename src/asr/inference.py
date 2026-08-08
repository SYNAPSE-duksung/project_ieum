from typing import Optional


class ASRInference:
    """
    IEUM ASR 추론 클래스

    역할:
    - FastAPI에서 전달받은 음성 데이터를 입력으로 받음
    - 이후 최종 ASR 모델과 연결
    - 음성 인식 결과 문자열 반환

    현재는 모델 연결 전이므로 임시 결과를 반환함
    """

    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self):
        """
        최종 ASR 모델 로딩

        TODO:
        최종 모델 구조가 확정되면 아래 구성 추가

        예:
        - Whisper processor
        - Whisper encoder
        - downstream model
        - checkpoint
        """

        # 아직 모델이 확정되지 않았으므로 비워둠
        pass

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> str:
        """
        음성 데이터를 받아 텍스트로 변환

        Parameters
        ----------
        audio_bytes : bytes
            FastAPI에서 전달받은 음성 파일의 바이너리 데이터

        filename : str, optional
            업로드된 음성 파일 이름

        Returns
        -------
        str
            음성 인식 결과
        """

        if not audio_bytes:
            raise ValueError("Audio data is empty.")

        # TODO:
        # 실제 모델 연결 후 아래 흐름으로 변경
        #
        # 1. audio_bytes -> waveform 변환
        # 2. 16kHz resampling 확인
        # 3. Whisper processor 입력 생성
        # 4. Whisper encoder feature 추출
        # 5. downstream model 추론
        # 6. CTC decoding
        # 7. 최종 문장 반환

        return "테스트 음성 인식 결과"