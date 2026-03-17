from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from scipy.sparse import csr_matrix

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