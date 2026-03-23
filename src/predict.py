from typing import List
from src.preprocess import preprocess_text
from src.vectorize import transform_text

def predict_sentiment_text(text: str, model, vectorizer) -> int:
    """
    단일 텍스트의 감정 예측

    return:
        1 (긍정) 또는 0 (부정)
    """

    # 1. 전처리
    processed_text: str = preprocess_text(text)

    # 2. 리스트로 변환 (vectorizer는 리스트 입력 필요)
    text_list: List[str] = [processed_text]

    # 3. 벡터 변환
    vector = transform_text(vectorizer, text_list)

    # 4. 예측
    prediction = model.predict(vector)

    return int(prediction[0])


def predict_sentiment_list(texts: List[str], model, vectorizer) -> List[int]:
    """
    여러 문장 예측

    return:
        1 (긍정) 또는 0 (부정)
    """

    # 1. 전처리
    processed_texts: List[str] = [preprocess_text(text) for text in texts]

    # 2. 벡터화
    vectors = transform_text(vectorizer, processed_texts)

    # 3. 예측
    predictions = model.predict(vectors)

    return list(predictions)


if __name__ == "__main__":

    from pathlib import Path
    from src.train_model import load_model
    from src.vectorize import load_vectorizer

    project_root = Path(__file__).parent.parent

    # 모델 & 벡터라이저 경로
    model_path = project_root / "data" / "test" / "test_sentiment_model.pkl"
    vectorizer_path = project_root / "data" / "test" / "test_tfidf_vectorizer.pkl"

    # 불러오기
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    # 테스트
    test_sentence = "이 영화 정말 다시 보고 싶은 최고의 영화였다~!~!"

    result = predict_sentiment_text(test_sentence, model, vectorizer)

    print("입력 문장:", test_sentence)
    print("예측 결과:", result)