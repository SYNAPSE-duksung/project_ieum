import os

import google.generativeai as genai

from dotenv import load_dotenv


load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")


def correct(text: str):

    prompt = f"""
당신은 구음장애인의 Whisper ASR 후처리 전문가입니다.

입력 문장을 자연스럽게 수정하세요.

규칙

1. 의미 유지
2. 오인식 수정
3. 띄어쓰기 수정
4. 조사 수정
5. 새로운 내용 추가 금지
6. 설명 금지
7. 수정된 문장만 출력

입력

{text}
"""

    response = model.generate_content(prompt)

    return response.text.strip()