from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF 벡터라이저 생성
    """

    vectorizer = TfidfVectorizer()

    return vectorizer