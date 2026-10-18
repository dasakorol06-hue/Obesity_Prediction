import pandas as pd

def build_features(df):
    df = df.copy()
    df["bmi_category"] = pd.cut(
        df["BMI"], bins=[0, 18.5, 25, 30, 100],
        labels=["under", "normal", "over", "obese"]
    ).astype(str)
    df["sedentary_index"] = df["TUE"] / (df["FAF"] + 1)
    df["meal_irregularity"] = (
        (df["CAEC"].isin(["Frequently", "Always"])) & (df["NCP"] > 3)
    ).astype(int)
    df["family_risk"] = (
        (df["family_history"] == "yes") & (df["Age"] > 30)
    ).astype(int)
    df["water_per_meal"] = df["CH2O"] / df["NCP"]
    return df

