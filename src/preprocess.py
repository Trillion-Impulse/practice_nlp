import re
from pathlib import Path
import pandas as pd

def remove_special_characters(text: str) -> str:
    """
    특수문자 제거
    한글과 공백만 남김
    """

    text = re.sub(r"[^가-힣\s]", "", text)

    return text

def clean_whitespace(text: str) -> str:
    """
    여러 개의 공백을 하나의 공백으로 변환
    문자열 앞뒤 공백 제거
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def preprocess_text(text: str) -> str:
    """
    단일 텍스트를 전처리
    1. 특수문자 제거
        - 한글과 공백만 남김
    2. 공백 정리
    """

    text = remove_special_characters(text)
    text = clean_whitespace(text)

    return text

if __name__ == "__main__":

    sample_text = "이 영화 정말 재미있다!!!!!"

    removed = remove_special_characters(sample_text)

    print("샘플:", sample_text)
    print("특수문자 제거:", removed)