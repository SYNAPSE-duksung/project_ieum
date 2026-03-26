"""
전처리 파이프라인 모듈.

주요 역할:
- audio loading, normalization, VAD, segmentation을 순차적으로 실행
- 전처리 전체 흐름을 하나로 묶어서 관리
- notebook이나 학습 코드에서 쉽게 호출 가능하도록 구성
"""