import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv("./data/processed/mortality_wide_est_pop.csv", index_col=[0, 1, 2])
days_per_month = pd.read_csv("./data/processed/days_per_month.csv", index_col=0)

deaths = df.loc[(slice(None), slice(None), "Deaths"), :]
pop = df.loc[(slice(None), slice(None), "Estimated_Population_Linear"), :]

exposure = pop.apply(lambda x: x * days_per_month["Days"], 1)

rate = deaths.to_numpy() / exposure.to_numpy()

pop = df.loc[(slice(None), slice(None), "Population"), :]
est = df.loc[(slice(None), slice(None), "Estimated_Population_Linear"), :]

pop0 = pop.iloc[0]
est0 = est.iloc[0]
x = pd.to_datetime(pop0.index)
ages = df.index.get_level_values(1)[:18]


# plt.figure(figsize=(5, 3))
#
# plt.plot(x, pop0.values/1e5, label="Yearly")
# plt.plot(x, est0.values/1e5, label="Monthly (est.)")
# plt.grid(which="minor", axis="x", color="white", linestyle=":")
# plt.minorticks_on()
#
# plt.legend(facecolor="white", framealpha=0.8, frameon=True)
#
# plt.xlabel("Year")
# plt.ylabel("Population (x100,000)")
# plt.title("Estimated Monthly Population (Female, Age 00-04)", loc="left")
#
# plt.tight_layout()
# plt.savefig("./tmp/est_pop.pdf")



color = plt.cm.winter(np.linspace(0.1, 0.9, 18))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

plt.figure(figsize=(10, 4))
ax = plt.gca()
# Females
females = rate[:18, :]
for i in range(18):
    ax.plot(x, np.log(females[i, :]), label="F" + ages[i][:2] + "-" + ages[i][-2:])

# Males
males = rate[18:, :]
for i in range(18):
    ax.plot(x, np.log(males[i, :]), label="F" + ages[i][:2] + "-" + ages[i][-2:], linestyle="--")

ax.grid(which="minor", axis="x", color="white", linestyle=":")
ax.minorticks_on()

# ax.legend(facecolor="white", framealpha=0.8, frameon=True)

ax.set_xlabel("Year")
ax.set_ylabel("log(Mortality rate)")
ax.set_title("Log Mortality Rate Time Series by Sex and Age Group", loc="left")

plt.tight_layout()
plt.savefig("./tmp/log_rates.pdf")