from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
import joblib

def create_model() -> MultinomialNB:
    """
    모델 생성
    """
    model = MultinomialNB()
    return model


def train_model(model: MultinomialNB, X, y) -> MultinomialNB:
    """
    모델 학습
    """
    model.fit(X, y)
    return model


def save_model(model: MultinomialNB, save_path: Path) -> None:
    """
    모델 저장
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)


def load_model(load_path: Path) -> MultinomialNB:
    """
    모델 불러오기
    """
    model = joblib.load(load_path)
    return model


if __name__ == "__main__":

    from src.load_data import load_data
    from src.preprocess import preprocess_dataframe
    from src.vectorize import (
        create_vectorizer,
        fit_vectorizer,
        transform_text,
        save_vectorizer
    )
    from sklearn.model_selection import train_test_split

    # 1. raw_data 경로 설정
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw" / "raw_reviews.txt"

    # 2. 데이터 로드
    df = load_data(data_path)

    # 3. 전처리
    df = preprocess_dataframe(df)

    # 4. X, y 분리
    X_texts = df["review"].tolist()
    y = df["label"]

    # 5. train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.2, random_state=42
    )

    # 6. 벡터화 (train data)
    vectorizer = create_vectorizer()
    vectorizer = fit_vectorizer(vectorizer, X_train)
    X_train_vectors = transform_text(vectorizer, X_train)
    X_test_vectors = transform_text(vectorizer, X_test)

    # 7. 모델 생성 및 학습
    model = create_model()
    model = train_model(model, X_train_vectors, y_train)
    print("모델 학습 완료")

    # 8. 정확도 테스트
    accuracy = model.score(X_test_vectors, y_test)
    print(f"테스트 정확도: {accuracy:.2f}")

    # 9. 저장
    vectorizer_path = project_root / "data" / "test" / "test_tfidf_vectorizer.pkl"
    save_vectorizer(vectorizer, vectorizer_path)
    print("벡터라이저 저장 완료")

    model_path = project_root / "data" / "test" / "test_sentiment_model.pkl"
    save_model(model, model_path)
    print("모델 저장 완료")