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

df_ts = pd.read_csv("./data/processed/pred_wide.csv", index_col=[0, 1, 2])


autocorr = pd.DataFrame({
    k: resid.apply(pd.Series.autocorr, axis=1, lag=k)
    for k in range(0, 31)
})


plt.figure(figsize=(5, 3))
ax = plt.gca()
autocorr.T.plot(ax=ax, color="#3333AA", alpha=0.2, legend=False)
autocorr.mean().plot(ax=ax, color="black", linewidth=4)
plt.title("Predicted Log Mortality Rate: Residual Autocorrelation", loc="left")
plt.xlabel("Lag")
plt.xticks([0, 12, 24])
plt.tight_layout()
plt.savefig("./tmp/autocorr.pdf")