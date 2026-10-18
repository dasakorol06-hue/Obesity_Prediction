import sys
import pandas as pd

sys.path.append("/content/drive/MyDrive/repo/src")
from data_quality import check_quality
from data_preprocessing import clean
from feature_engineering import build_features


def run(batch_path, out_path):
    """Обрабатывает одну порцию."""
    df = pd.read_csv(batch_path)
    issues = check_quality(df)
    if issues:
        print("Нарушения:", issues)
    df = clean(df)
    df = build_features(df)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        run(sys.argv[1], sys.argv[2])
