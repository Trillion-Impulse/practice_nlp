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

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 리뷰 컬럼 전처리
    """

    df["review"] = df["review"].apply(preprocess_text)

    return df

def save_data_to_csv(df: pd.DataFrame, save_path: Path):
    """
    데이터를 csv로 저장
    """

    df.to_csv(save_path, index=False)

if __name__ == "__main__":

    # remove_special_characters 테스트
    sample_text = "이 영화 정말 재미있다!!!!!"

    removed = remove_special_characters(sample_text)

    print("remove_special_characters 함수의 특수문자 제거 테스트")
    print("샘플:", sample_text)
    print("특수문자 제거:", removed)

    # clean_whitespace 테스트
    sample_text2 = "   이   영화    정말   재미있다   "
    whitespace_cleaned = clean_whitespace(sample_text2)
    print("\nclean_whitespace 함수의 공백 정리 테스트")
    print("샘플:", repr(sample_text2))
    print("공백 정리:", repr(whitespace_cleaned))

    # preprocess_text 테스트
    sample_text3 = "   이 영화   정말   재미있다!!!!!   "
    preprocessed = preprocess_text(sample_text3)
    print("\npreprocess_text 함수의 단일 텍스트 전처리 테스트")
    print("샘플:", repr(sample_text3))
    print("전처리:", repr(preprocessed))

    # preprocess_dataframe 테스트
    import pandas as pd
    df = pd.DataFrame({
        "review": ["이 영화   정말 재미있다!!!", "완전    시간 낭비였다...", "   연기가 어색했다 "],
        "label": [1, 0, 0]
    })
    print("\npreprocess_dataframe 함수의 데이터프레임의 리뷰 컬럼 전처리 테스트")
    print("\n샘플\n", df)
    df_processed = preprocess_dataframe(df)
    print("\n전처리\n", df_processed)

    # save_data_to_csv 테스트
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "test" / "save_test.csv"
    print("\nsave_data_to_csv 함수의 csv로 데이터 저장 테스트")
    save_data_to_csv(df, data_path)
    print("저장 경로: ", data_path)