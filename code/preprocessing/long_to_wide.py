import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mortality = pd.read_csv("./data/raw/pop_mort.csv")

mortality["Date"] = pd.to_datetime(
    pd.DataFrame({
        "year": mortality["Year"],
        "month": mortality["Month"],
        "day": 15
    }),
    format="%M-%Y"
)

mortality_grouped = mortality.groupby(
    ["Sex", "Age_group"]
)


def to_ts(df):
    df.drop(columns=["Year", "Month", "Sex", "Age_group"], inplace=True)
    df.set_index("Date", inplace=True)
    return df.transpose()



mortality_ts = mortality_grouped.apply(to_ts)
mortality_ts.index.names = ["Sex", "Age_group", "Value"]

mortality_ts.to_csv("./data/processed/mortality_wide.csv")

deaths = mortality_ts.loc[(slice(None), slice(None), "Deaths"), :]
plt.figure(figsize=(10, 10))
for id, ts in deaths.iterrows():
    color = ["#990000", "#000099"][0 if id[0] == "Female" else 1]
    ts.plot(color=color)
plt.savefig("./tmp/deaths_ts.pdf")


population = mortality_ts.loc[(slice(None), slice(None), "Population"), :]
plt.figure(figsize=(10, 10))
for id, ts in population.iterrows():
    color = ["#990000", "#000099"][0 if id[0] == "Female" else 1]
    ts.plot(color=color)
plt.savefig("./tmp/pop_ts.pdf")


deaths_rate = mortality_ts.loc[(slice(None), slice(None), "Deaths_Rate"), :]
plt.figure(figsize=(10, 10))
for id, ts in deaths_rate.iterrows():
    color = ["#990000", "#000099"][0 if id[0] == "Female" else 1]
    ts.plot(color=color)
plt.savefig("./tmp/deaths_rate_ts.pdf")