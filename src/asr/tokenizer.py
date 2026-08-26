from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor


class CTCCharacterTokenizer:
    """
    한국어 Character 단위 CTC Tokenizer.

    예:
        "나는 학교"

    →
        ["나", "는", "|", "학", "교"]

    →
        [token_id, token_id, ...]
    """

    BLANK_TOKEN = "<blank>"
    UNK_TOKEN = "<unk>"
    SPACE_TOKEN = "|"

    def __init__(
        self,
        vocab: dict[str, int],
    ) -> None:
        self.vocab = vocab

        self.id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

        required_tokens = [
            self.BLANK_TOKEN,
            self.UNK_TOKEN,
            self.SPACE_TOKEN,
        ]

        missing_tokens = [
            token
            for token in required_tokens
            if token not in vocab
        ]

        if missing_tokens:
            raise ValueError(
                "Vocabulary에 필수 토큰이 없습니다.\n"
                f"누락 토큰: {missing_tokens}"
            )

        self.blank_id = vocab[self.BLANK_TOKEN]
        self.unk_id = vocab[self.UNK_TOKEN]
        self.space_id = vocab[self.SPACE_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        불필요한 연속 공백만 정리한다.

        기존 전처리에서 만들어진 문자를 임의로
        변경하지 않는 것이 원칙이다.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"text는 str이어야 합니다: {type(text)}"
            )

        return " ".join(text.strip().split())

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
    ) -> "CTCCharacterTokenizer":
        """
        Train transcript만 사용하여 vocabulary를 만든다.
        """
        characters: set[str] = set()

        text_count = 0

        for text in texts:
            text = cls.normalize_text(text)

            if not text:
                continue

            text_count += 1

            for character in text:
                if character == " ":
                    continue

                characters.add(character)

        if text_count == 0:
            raise ValueError(
                "Vocabulary를 생성할 문장이 없습니다."
            )

        # 실험 재현성을 위해 정렬
        sorted_characters = sorted(characters)

        # CTC blank는 0번으로 고정
        vocab: dict[str, int] = {
            cls.BLANK_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.SPACE_TOKEN: 2,
        }

        for character in sorted_characters:
            if character in vocab:
                continue

            vocab[character] = len(vocab)

        return cls(vocab)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """문장을 CTC label ID sequence로 변환한다."""
        text = self.normalize_text(text)

        token_ids: list[int] = []

        for character in text:
            if character == " ":
                token_ids.append(self.space_id)
                continue

            token_ids.append(
                self.vocab.get(
                    character,
                    self.unk_id,
                )
            )

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        ctc_decode: bool = False,
    ) -> str:
        """
        token ID를 문자열로 복원한다.

        ctc_decode=True이면:
            - 연속 중복 token 제거
            - blank 제거
        """
        decoded_tokens: list[str] = []

        previous_id: int | None = None

        for token_id in token_ids:
            token_id = int(token_id)

            if ctc_decode:
                if token_id == previous_id:
                    previous_id = token_id
                    continue

                previous_id = token_id

                if token_id == self.blank_id:
                    continue

            token = self.id_to_token.get(
                token_id,
                self.UNK_TOKEN,
            )

            if token == self.BLANK_TOKEN:
                continue

            if token == self.SPACE_TOKEN:
                decoded_tokens.append(" ")
            elif token == self.UNK_TOKEN:
                decoded_tokens.append(self.UNK_TOKEN)
            else:
                decoded_tokens.append(token)

        return "".join(decoded_tokens).strip()

    def encode_tensor(
        self,
        text: str,
    ) -> Tensor:
        """문장을 LongTensor label로 변환한다."""
        return torch.tensor(
            self.encode(text),
            dtype=torch.long,
        )

    def save(
        self,
        vocab_path: str | Path,
    ) -> None:
        """Vocabulary를 JSON 파일로 저장한다."""
        vocab_path = Path(vocab_path)

        vocab_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with vocab_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.vocab,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(
        cls,
        vocab_path: str | Path,
    ) -> "CTCCharacterTokenizer":
        """저장된 vocabulary를 불러온다."""
        vocab_path = Path(vocab_path)

        if not vocab_path.exists():
            raise FileNotFoundError(
                "Vocabulary 파일을 찾을 수 없습니다.\n"
                f"경로: {vocab_path}"
            )

        with vocab_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            vocab = json.load(file)

        vocab = {
            str(token): int(token_id)
            for token, token_id in vocab.items()
        }

        return cls(vocab)