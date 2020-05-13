import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mortality_wide = pd.read_csv("./data/processed/mortality_wide_est_pop.csv")
mortality_wide.set_index(["Value", "Sex", "Age_group"], inplace=True)
mortality_wide.sort_index(inplace=True)

days_per_month = pd.read_csv("./data/processed/days_per_month.csv", index_col=0)

deaths = mortality_wide.loc[("Deaths", slice(None), slice(None)), :]
pop = mortality_wide.loc[("Estimated_Population_Linear", slice(None), slice(None)), :]

exposure = pop.apply(lambda x: x * days_per_month["Days"], 1)

rate = deaths.to_numpy() / exposure.to_numpy()

midpoints = pd.read_csv("./data/processed/midpoints.csv")
midpoints = midpoints.groupby(["Sex", "Age_group"]).agg("mean")


n_components = 4
n_vals = 3
pca = PCA()
logits = np.log(rate)
pca.fit(logits)
U_rate = pca.transform(logits)
percent_var_explained = pca.explained_variance_ratio_

pct_var_explained_remaining = pca.explained_variance_[1:].cumsum() / pca.explained_variance_[1:].sum()

fig, axs = plt.subplots(figsize=(10, 10), nrows=n_components, ncols=2, gridspec_kw={'width_ratios': [2, 1]})

for c in range(n_components):
    # plt.subplot(n_components, 1, c+1)
    # plot time series
    U = np.zeros((n_vals, pca.n_components_))
    min_val = U_rate[:, c].min()
    max_val = U_rate[:, c].max()
    lim = max(np.abs([min_val, max_val]))
    U[:, c] = np.linspace(-lim, lim, n_vals)
    X = pca.inverse_transform(U)
    for k in range(n_vals):
        axs[c, 0].plot(
            pd.to_datetime(mortality_wide.columns),
            X[k, :] - pca.mean_,
            color="#5500{}{}".format(k*4, k*4),
            label="{:.2f}".format(U[k, c])
        )
    axs[c, 0].grid(which="minor", axis="x", color="white", linestyle=":")
    axs[c, 0].minorticks_on()
    axs[c, 0].legend(facecolor="white", framealpha=0.8, frameon=True)
    axs[c, 0].set_title("Component {} ({:.5f} % variance explained)".format(c + 1, percent_var_explained[c]), loc="left")
    axs[c, 0].set_ylabel("Diff. from mean")
    # plot scatter
    axs[c, 1].plot(
        midpoints["Midpoint_linear"][18:],
        U_rate[18:, c],
        label="Male"
    )
    axs[c, 1].plot(
        midpoints["Midpoint_linear"][:18],
        U_rate[:18, c],
        label="Female"
    )
    axs[c, 1].legend(facecolor="white", framealpha=0.8, frameon=True)
    axs[c, 1].set_ylabel("Component {}".format(c + 1))
    axs[c, 1].set_xlabel("Age")

fig.tight_layout()
fig.savefig("./tmp/pca_ts_logit.pdf")

# 1: mean of ts
# 2: slope
# 3: deviation from seasonality
#     - seems to have two modes 2010-2015 and outside


pc = pd.DataFrame(U_rate, index=mortality_wide.loc[("Deaths", slice(None), slice(None)), :].index)
pc.columns = ["PC{}".format(i+1) for i in range(36)]
pc.reset_index(inplace=True)
pc.drop(columns="Value", inplace=True)
pc.set_index(["Sex", "Age_group"], inplace=True)

pc = pc.join(midpoints)
pc.reset_index(inplace=True)



pc.to_csv("./data/processed/pc.csv", index=False)

mean = pd.DataFrame(
    {"Mean": pca.mean_},
    index=pd.to_datetime(mortality_wide.columns)
)

mean.to_csv("./data/processed/pc_mean.csv")