from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from scipy.sparse import csr_matrix
from pathlib import Path
import joblib

def create_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF 벡터라이저 생성
    """

    vectorizer = TfidfVectorizer()

    return vectorizer


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: List[str]) -> TfidfVectorizer:
    """
    벡터라이저 학습
    """

    vectorizer.fit(texts)

    return vectorizer


def transform_text(vectorizer: TfidfVectorizer, texts: List[str]) -> csr_matrix:
    """
    텍스트 리스트를 숫자 벡터로 변환
    """

    vectors = vectorizer.transform(texts)

    return vectors


def save_vectorizer(vectorizer: TfidfVectorizer, save_path: Path) -> None:
    """
    벡터라이저를 파일로 저장
    """

    joblib.dump(vectorizer, save_path)


def load_vectorizer(load_path: Path) -> TfidfVectorizer:
    """
    저장된 벡터라이저 불러오기
    """

    vectorizer = joblib.load(load_path)

    return vectorizer


if __name__ == "__main__":

    # 테스트 데이터
    texts: List[str] = [
        "영화 정말 재미있다",
        "완전 시간 낭비였다",
        "연기가 훌륭했다"
    ]

    # 1. 벡터라이저 생성
    vectorizer = create_vectorizer()

    # 2. 학습
    vectorizer = fit_vectorizer(vectorizer, texts)

    # 3. 변환 (여러 개)
    vectors = transform_text(vectorizer, texts)

    print("벡터 shape:", vectors.shape)

    print("\n단어 목록")
    print(vectorizer.get_feature_names_out())

    print("\n벡터 결과")
    print(vectors.toarray())

    # 4. 단일 텍스트 테스트
    single_text = ["이 영화 최고다"]

    single_vector = transform_text(vectorizer, single_text)

    print("\n단일 문장 벡터 shape:", single_vector.shape)

    # 5. 저장
    project_root = Path(__file__).parent.parent
    save_path = project_root / "data" / "test" / "tfidf_vectorizer.pkl"

    save_vectorizer(vectorizer, save_path)

    print("\n벡터라이저 저장 완료:", save_path)

    # 6. 불러오기 테스트
    loaded_vectorizer = load_vectorizer(save_path)

    test_vector = transform_text(loaded_vectorizer, ["이 영화 정말 정말 최고다"])

    print("\n불러온 벡터라이저 테스트:", test_vector.toarray())