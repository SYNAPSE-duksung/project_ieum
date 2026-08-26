| 설정            |       초기값 | 선정 근거                     | 변경 기준                 |
| ------------- | --------: | ------------------------- | --------------------- |
| sample rate   | 16,000 Hz | Whisper 입력 및 전처리 데이터 규격   | 변경하지 않음               |
| batch size    |         4 | Colab GPU 예비 실행을 위한 보수적 값 | OOM 시 감소, 여유 시 증가     |
| epoch         |        30 | Early stopping을 포함한 최대값   | 학습 곡선 확인 후 변경         |
| learning rate |      1e-4 | CTC head 예비 학습 시작점        | Linear-CTC LR 탐색 후 확정 |
| weight decay  |      0.01 | AdamW 초기 기준값              | 과적합 정도에 따라 조정         |
| patience      |         5 | 검증 성능 정체 시 조기 종료          | 변동이 크면 증가             |
