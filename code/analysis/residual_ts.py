import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import itertools
import scipy.stats
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

best_models = pd.read_csv("./data/results/glm_best_models.csv")

df = pd.read_csv("./data/processed/df_for_glm.csv", index_col=0)

f = best_models.iloc[3]["Formula"]
y, X = patsy.dmatrices(f, data=df)
exp = df["Exposure"].to_numpy()
glm = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exp)
model = glm.fit(scale="X2")

df["pred"] = model.predict()
df["pred_rate"] = df["pred"] / df["Exposure"]

df = df[["Sex", "Age_group", "Year", "Month", "Deaths", "Exposure", "pred", "pred_rate"]]

df["resid_pearson"] = model.resid_pearson

def long_to_wide(df):

    df["Date"] = pd.to_datetime(
        pd.DataFrame({
            "year": df["Year"],
            "month": df["Month"],
            "day": 15
        }),
        format="%M-%Y"
    )

    df_grouped = df.groupby(
        ["Sex", "Age_group"]
    )

    df_ts = df_grouped.apply(to_ts)
    df_ts.index.names = ["Sex", "Age_group", "Value"]

    return df_ts

def to_ts(df):
    df.drop(columns=["Year", "Month", "Sex", "Age_group"], inplace=True)
    df.set_index("Date", inplace=True)
    return df.transpose()


df_ts = long_to_wide(df)

df_ts.to_csv("./data/processed/pred_wide.csv")
