
import pandas as pd
import numpy as np


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop useless/unreliable ID-like columns, but save a HasCabin signal first
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df = df.drop(columns=["PassengerId", "Ticket", "Cabin"], errors="ignore")

    # 2. Extract Title from Name (captures status/age/gender info Sex/Age miss)
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]*)\.")
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )
    df = df.drop(columns=["Name"], errors="ignore")

    # 3. Combine SibSp + Parch into family size features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # 4. Impute Age by Title + Pclass median (more accurate than a global median)
    df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Age"] = df["Age"].fillna(df["Age"].median())  # safety net

    # 5. Impute Embarked with the mode (only 2 missing, categorical column)
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # 6. Impute Fare with median Fare per Pclass
    if df["Fare"].isna().any():
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )

    # 7. Encode Sex as binary, one-hot encode Embarked/Title for the model
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

    # 8. Scale continuous features separately later (fit only on train split to avoid leakage)

    return df


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "titanic.csv"
    raw = pd.read_csv(path)
    cleaned = clean_titanic(raw)

    print("Cleaned shape:", cleaned.shape)
    print("Missing values left:\n", cleaned.isna().sum()[cleaned.isna().sum() > 0])
    print(cleaned.head())

    cleaned.to_csv("titanic_cleaned.csv", index=False)
    print("\nSaved to titanic_cleaned.csv")