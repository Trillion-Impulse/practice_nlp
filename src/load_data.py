import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"올바른 경로가 아니거나 파일이 없습니다: {file_path}")

    df = pd.read_csv(path)

    return df


if __name__ == "__main__":

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "raw" / "raw_reviews.txt"

    df = load_data(data_path)

    print("데이터 샘플")
    print(df.head())

    print("\n데이터 개수:", len(df))

    assert "review" in df.columns
    assert "label" in df.columns

    print("데이터셋 형식 검증 완료")