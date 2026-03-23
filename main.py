from pathlib import Path
from src.load_data import load_data
from src.preprocess import preprocess_dataframe
from src.vectorize import create_vectorizer, fit_vectorizer, transform_text, save_vectorizer
from src.train_model import create_model, train_model, save_model
from sklearn.model_selection import train_test_split

# 학습 함수
def train():

    print("모델 학습 시작")

    # 프로젝트 루트
    project_root = Path(__file__).parent

    # 데이터 로드
    data_path = project_root / "data" / "raw" / "raw_reviews.txt"
    df = load_data(data_path)

    # 전처리
    df = preprocess_dataframe(df)

    # X, y 분리
    X = df["review"].tolist()
    y = df["label"]

    # train/test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 벡터화
    vectorizer = create_vectorizer()
    vectorizer = fit_vectorizer(vectorizer, X_train)
    X_train_vectors = transform_text(vectorizer, X_train)
    X_test_vectors = transform_text(vectorizer, X_test)

    # 모델 생성 및 학습
    model = create_model()
    model = train_model(model, X_train_vectors, y_train)
    print("모델 학습 완료")

    # 평가
    accuracy = model.score(X_test_vectors, y_test)
    print(f"평가 정확도: {accuracy:.2f}")

    # 모델 및 벡터라이저 저장
    model_dir = project_root / "data" / "model"
    model_dir.mkdir(exist_ok=True, parents=True)

    model_path = model_dir / "sentiment_model.pkl"
    save_model(model, model_path)
    print("모델 저장 완료:", model_path)

    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"
    save_vectorizer(vectorizer, vectorizer_path)
    print("벡터라이저 저장 완료:", vectorizer_path)