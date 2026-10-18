import pandas as pd

def clean(df):
    df = df.dropna(subset=["Age", "Gender", "FAF", "TUE"]).copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["BMI"] = (df["Weight"] / df["Height"] ** 2).round(2)
    if "Obesity" in df.columns:
        df["obese_flag"] = df["Obesity"].isin(
            ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
        ).astype(int)
    return df

