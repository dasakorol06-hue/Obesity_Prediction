import pandas as pd

VALID_GENDER = {"Female", "Male"}
VALID_YESNO  = {"yes", "no"}
VALID_OBESITY = {
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
}

def check_quality(df):
    issues = []
    for col in ["Age", "Gender", "FAF", "TUE"]:
        if df[col].isna().any():
            issues.append(f"R1: пропуски в {col}: {df[col].isna().sum()}")
    if ((df["Age"] < 14) | (df["Age"] > 61)).any():
        issues.append("R2: Age вне диапазона 14–61")
    if (df[["Height", "Weight"]] <= 0).any().any():
        issues.append("R3: неположительные Height/Weight")
    if not set(df["Gender"].unique()).issubset(VALID_GENDER):
        issues.append("R4: недопустимые значения Gender")
    if "Obesity" in df.columns:
        if not set(df["Obesity"].unique()).issubset(VALID_OBESITY):
            issues.append("R5: недопустимые категории Obesity")
    if df.duplicated().any():
        issues.append(f"R6: полных дубликатов: {df.duplicated().sum()}")
    return issues

