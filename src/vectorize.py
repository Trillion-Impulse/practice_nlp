from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

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