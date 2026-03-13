import re
from pathlib import Path
import pandas as pd

def remove_special_characters(text: str) -> str:
    """
    특수문자 제거
    """

    text = re.sub(r"[^가-힣\s]", "", text)

    return text

if __name__ == "__main__":

    sample_text = "이 영화 정말 재미있다!!!!!"

    removed = remove_special_characters(sample_text)

    print("샘플:", sample_text)
    print("특수문자 제거:", removed)