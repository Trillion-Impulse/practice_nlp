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

    # 1. raw_data 경로 설정
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw" / "raw_reviews.txt"

    # 2. 데이터 로드
    df = load_data(data_path)

    # 3. 전처리
    df = preprocess_dataframe(df)

    # 4. X, y 분리
    X_texts = df["review"]
    y = df["label"]

    # 5. 벡터화
    vectorizer = create_vectorizer()
    vectorizer = fit_vectorizer(vectorizer, X_texts)
    X_vectors = transform_text(vectorizer, X_texts)

    # 6. 모델 생성 및 학습
    model = create_model()
    model = train_model(model, X_vectors, y)
    print("모델 학습 완료")

    # 7. 저장
    model_path = project_root / "data" / "test" / "sentiment_model.pkl"
    save_model(model, model_path)

    print("모델 저장 완료")