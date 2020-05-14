import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df_ts = pd.read_csv("./data/processed/pred_wide.csv", index_col=[0, 1, 2])

log_pred_rate = df_ts.loc[(slice(None), slice(None), "pred_rate"), :].applymap(np.log)

midpoints = pd.read_csv("./data/processed/midpoints.csv")
midpoints = midpoints.groupby(["Sex", "Age_group"]).agg("mean")

n_components = 3
pca = PCA()
pca.fit(log_pred_rate)
U_rate = pca.transform(log_pred_rate)
percent_var_explained = pca.explained_variance_ratio_

pct_var_explained_remaining = pca.explained_variance_[1:].cumsum() / pca.explained_variance_[1:].sum()

fig, axs = plt.subplots(figsize=(10, 7), nrows=n_components, ncols=2,
                        gridspec_kw={'width_ratios': [2, 1]})

for c in range(n_components):
    # plt.subplot(n_components, 1, c+1)
    # plot time series
    U = np.zeros((3, pca.n_components_))
    min_val = U_rate[:, c].min()
    max_val = U_rate[:, c].max()
    U[:, c] = [min_val, 0, max_val]
    X = pca.inverse_transform(U)
    for k in range(3):
        axs[c, 0].plot(
            pd.to_datetime(df_ts.columns),
            X[k, :] - pca.mean_,
            color="#{}{}{}{}{}{}".format(*[["3", "6", "9"][k]]*6),
            label="{:.2f}".format(U[k, c])
        )
    axs[c, 0].grid(which="minor", axis="x", color="white", linestyle=":")
    axs[c, 0].minorticks_on()
    axs[c, 0].legend(facecolor="white", framealpha=0.8, frameon=True)
    axs[c, 0].set_title("Component {} ({:.3f} % variance explained)".format(c + 1, 100 * percent_var_explained[c]), loc="left")
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

fig.suptitle("Predicted Log Mortality Rate Time Series: Principal Components Analysis", fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("./tmp/pred_pca_ts_logit.pdf")

# 1: mean of ts
# 2: slope
# 3: deviation from seasonality
#     - seems to have two modes 2010-2015 and outside


pc = pd.DataFrame(U_rate, index=df_ts.loc[(slice(None), slice(None), "pred_rate"), :].index)
pc.columns = ["PC{}".format(i+1) for i in range(36)]
pc.reset_index(inplace=True)
pc.drop(columns="Value", inplace=True)
pc.set_index(["Sex", "Age_group"], inplace=True)

pc = pc.join(midpoints)
pc.reset_index(inplace=True)



pc.to_csv("./data/processed/pred_pc.csv", index=False)



mean = pd.DataFrame(
    {"Mean": pca.mean_},
    index=pd.to_datetime(df_ts.columns)
)

mean.to_csv("./data/processed/pred_pc_mean.csv")