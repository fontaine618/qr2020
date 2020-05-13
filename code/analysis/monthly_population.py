import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mortality_wide = pd.read_csv("./data/processed/mortality_wide.csv")
mortality_wide_grouped = mortality_wide.groupby(["Sex", "Age_group"])


def pred_monthly_population_gam(df):
    df.drop(columns=["Sex", "Age_group", "Value"], inplace=True)

    df = df.iloc[1].transpose().reset_index()
    df.columns = ["time", "pop"]

    bs = BSplines(df.index.values, df=[12], degree=[3])

    gam = GLMGam.from_formula(
        "pop ~ 1",
        data=df,
        smoother=bs
    )
    res = gam.fit()
    return pd.Series(res.predict(), index=df["time"])


est_monthly_population = mortality_wide_grouped.apply(pred_monthly_population_gam)
est_monthly_population.reset_index(inplace=True)
est_monthly_population["Value"] = "Estimated_Population_GAM"


def pred_monthly_population_linear(df):
    df.drop(columns=["Sex", "Age_group", "Value"], inplace=True)

    df = df.iloc[1].transpose().reset_index()
    df.columns = ["time", "pop"]
    df["est_pop"] = df["pop"]

    for y in range(11):
        rows = range(12*y + 5, 12*y + 18)
        d = df.iloc[rows, 1].values
        pred = np.linspace(0, len(d), len(d)) * (d[-1] - d[0]) / len(d) + d[0]
        df.loc[rows, "est_pop"] = pred
    df.set_index("time", inplace=True)
    return df["est_pop"]


est_monthly_population_linear = mortality_wide_grouped.apply(pred_monthly_population_linear)
est_monthly_population_linear.reset_index(inplace=True)
est_monthly_population_linear["Value"] = "Estimated_Population_Linear"


mortality_wide = pd.concat([mortality_wide, est_monthly_population, est_monthly_population_linear])
mortality_wide.set_index(["Sex", "Age_group", "Value"], inplace=True)
mortality_wide.sort_index(inplace=True)



df = mortality_wide.loc[("Female", "10_14",
                         ["Population", "Estimated_Population_GAM", "Estimated_Population_Linear"]), :]

plt.figure()
plt.plot(df.T)
plt.savefig("./tmp/est_pop.pdf")


mortality_wide.to_csv("./data/processed/mortality_wide_est_pop.csv")

# beck to long
mortality_wide.reset_index(inplace=True)
mortality_long = mortality_wide.melt(
    id_vars=["Sex", "Age_group", "Value"],
    value_vars=mortality_wide.columns[3:],
    var_name="Date",
    value_name="val"
)
mortality_long.set_index(["Value", "Date", "Sex", "Age_group"], inplace=True)
mortality_long.sort_index(inplace=True)

mortality_long = mortality_long.unstack(level=0)

mortality_long.columns = ["Deaths", "Est_Pop_GAM", "Est_Pop_Lin", "Population"]

midpoints = pd.read_csv("./data/processed/midpoints.csv")
midpoints.set_index(["Date", "Sex", "Age_group"], inplace=True)

mortality_long = mortality_long.join(midpoints)

mortality_long.reset_index(inplace=True)

date = pd.to_datetime(mortality_long["Date"])

mortality_long["Year"] = date.apply(lambda x: x.year)
mortality_long["Month"] = date.apply(lambda x: x.month)

mortality_long.to_csv("./data/processed/mortality_long.csv")