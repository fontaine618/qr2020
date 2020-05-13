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

pop_ysa = mortality.groupby(["Date", "Sex", "Age_group"]).agg({"Population": "mean"})
pop_ysa.reset_index(inplace=True)

pop_ys = pop_ysa.groupby(["Date", "Sex"])

gr, df = next(iter(pop_ys))



def midpoint_linear(row):
    amin = row["Age_min"]
    amax = row["Age_max"] + 1
    ages = np.array(range(amin, amax))
    est_pop = np.linspace(row["prev_pop"], row["next_pop"], num=len(ages))
    return np.sum(ages * est_pop) / sum(est_pop)


def compute_midpoints(df):
    df.reset_index(inplace=True)
    df["Age_min"] = df["Age_group"].apply(lambda x: int(x[:2]))
    df["Age_max"] = df["Age_group"].apply(lambda x: int(x[-2:]))
    df["Midpoint"] = (df["Age_max"] + df["Age_min"]) / 2
    df["prev_pop"] = [df["Population"][0], *df["Population"][:-1]]
    df["next_pop"] = [*df["Population"][1:], 0]
    df["Midpoint_linear"] = df.apply(midpoint_linear, axis=1)
    df.set_index(["Date", "Sex", "Age_group"], inplace=True)
    return df.drop(columns=["Age_min", "Age_max", "prev_pop", "next_pop"])


compute_midpoints(df)
midpoints = pd.concat([compute_midpoints(df) for _, df in pop_ys])
midpoints.drop(columns="index", inplace=True)

midpoints.groupby(["Sex", "Age_group"]).agg({"Midpoint_linear": "mean"})

midpoints.drop(columns="Population").to_csv("./data/processed/midpoints.csv")

plt.figure()
for _, df in midpoints.groupby(["Date", "Sex"]):
    plt.plot(df["Midpoint"], df["Midpoint_linear"] - df["Midpoint"])

plt.savefig("./tmp/midpoints.pdf")

