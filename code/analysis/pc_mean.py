import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mean = pd.read_csv("./data/processed/pc_mean.csv", index_col=0)
mean.index = pd.to_datetime(mean.index)
date = pd.to_datetime(mean.index)
mean["Year"] = date.map(lambda x: x.year)
mean["Month"] = date.map(lambda x: x.month)

mean_long = mean.pivot(
    index="Year",
    columns="Month"
)
mean_long.columns = range(1, 13)



import patsy
from statsmodels.api import OLS

y, X = patsy.dmatrices("Mean ~ bs(Year, 5) + bs(Month, 5)", data=mean)
model = OLS(y, X).fit()
model.summary()




mean["Pred"] = model.predict()

fig, (ax1, ax2) = plt.subplots(2, 1)

mean[["Mean", "Pred"]].plot(ax=ax1)

mean_long.T.plot(ax=ax2, color="#3333AA", legend=False, alpha=0.5)

ax2.plot(range(1, 13), mean["Pred"][-12:], color="black", linewidth=4)

plt.savefig("./tmp/pc_mean.pdf")