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