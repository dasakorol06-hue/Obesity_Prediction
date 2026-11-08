import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

NUM_COLS = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE",
            "sedentary_index", "water_per_meal"]
CAT_COLS = ["Gender", "family_history", "FAVC", "CAEC", "SMOKE",
            "SCC", "CALC", "MTRANS", "bmi_category"]

def clean(df):
    df = df.dropna(subset=["Age", "Gender", "FAF", "TUE"]).copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["BMI"] = (df["Weight"] / df["Height"] ** 2).round(2)
    df["obese_flag"] = df["Obesity"].isin(
        ["Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
    ).astype(int)
    return df

def add_features(df):
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

def make_pipeline(model):
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS)
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", prep),
        ("model", model)
    ])
