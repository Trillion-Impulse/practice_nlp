from sklearn.naive_bayes import MultinomialNB

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