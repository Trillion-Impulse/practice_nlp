from sklearn.naive_bayes import MultinomialNB

def create_model() -> MultinomialNB:
    """
    모델 생성
    """
    model = MultinomialNB()
    return model