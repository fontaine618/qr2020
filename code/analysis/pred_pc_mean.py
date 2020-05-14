import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mean = pd.read_csv("./data/processed/pred_pc_mean.csv", index_col=0)
mean.index = pd.to_datetime(mean.index)
date = pd.to_datetime(mean.index)
mean["Year"] = date.map(lambda x: x.year)
mean["Month"] = date.map(lambda x: x.month)

mean_long = mean.pivot(
    index="Year",
    columns="Month"
)
mean_long.columns = range(1, 13)


mean.columns = ['Mean', 'Year', 'Month']

m_long = mean.pivot(index="Month", columns="Year", values="Mean")
d_long = mean.reset_index().pivot(index="Month", columns="Year", values="index")


color = plt.cm.coolwarm(np.linspace(0.1, 0.9, 12))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)

for i in range(12):
    ax1.plot(d_long.iloc[:, i], m_long.iloc[:, i])
ax1.grid(which="minor", axis="x", color="white", linestyle=":")
ax1.minorticks_on()

ax1.set_title("PCA Mean through Years", loc="left")
ax1.set_ylabel("log(Mortality rate)")
ax1.set_xlabel("Year")

m_long.plot(ax=ax2, legend=False, cmap="coolwarm")
ax2.grid(which="minor", axis="x", color="white", linestyle=":")
ax2.minorticks_on()
ax2.set_title("PCA Mean by Month", loc="left")

plt.tight_layout()
plt.savefig("./tmp/pred_pc_mean.pdf")