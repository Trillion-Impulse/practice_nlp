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