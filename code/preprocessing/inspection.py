import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mortality = pd.read_csv("./data/raw/pop_mort.csv")

# check missing values
mortality.isna().sum()

for name, col in mortality.items():
    print(name + "====================================")
    print(col.value_counts(dropna=False))

# no missing values
# year is complete (432 by year)
# Month is complete (432 per month)
# Sex is complete (2592 per sex)
# Age group is complete (288 per age group)

# inspect mortality

mortality["Deaths"].hist()
plt.ylabel("Deaths")
plt.savefig("./tmp/hist_deaths.pdf")

mortality["Deaths Rate"] = mortality["Deaths"] / mortality["Population"]

plt.figure()
mortality["Deaths Rate"].hist()
plt.ylabel("Deaths / Population")
plt.savefig("./tmp/hist_deaths_rate.pdf")


plt.figure()
mortality["Population"].hist()
plt.ylabel("Population")
plt.savefig("./tmp/hist_population.pdf")