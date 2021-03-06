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

best_models = pd.read_csv("./data/results/glm_best_models.csv")

df = pd.read_csv("./data/processed/df_for_glm.csv", index_col=0)

best_results = pd.DataFrame(
    columns=["model", "LogLik", "DF_Model", "AIC", "BIC", "MSE", "MSEL", "Formula"],
    index=pd.MultiIndex(levels=[[]]*3,
                    codes=[[]]*3,
                    names=["family", "formula", "dummy"])
)

for _, (family, name, formula) in best_models[["family", "formula", "Formula"]].iterrows():
    print("=" * 80)
    # vals = {"n_age": n_age, "n_date": n_date, "n_month": n_month}
    f = formula
    # for k, v in vals.items():
    #     f = f.replace(k, str(v))
    print(family, name, f)
    y, X = patsy.dmatrices(f, data=df)
    exp = df["Exposure"].to_numpy()
    if family == "Poisson":
        glm = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exp)
    elif family == "NB":
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), exposure=exp)
    elif family == "Tweedie":
        glm = sm.GLM(y, X, family=sm.families.Tweedie(var_power=1.5, eql=True), exposure=exp)
    elif family == "logNormal":
        fm = sm.families.Gaussian(link=sm.families.links.log())
        glm = sm.GLM(y, X, family=fm, exposure=exp)
    res = glm.fit(scale="X2")
    pred = res.predict().reshape((-1, 1))
    mse = np.mean((y - pred)**2)
    msel = np.mean((np.log(y) - np.log(pred))**2)
    print("LogLik", res.llf)
    print("DF Model", res.df_model)
    print("AIC", res.aic)
    print("BIC", res.bic)
    print("MSE", mse)
    print("MSEL", msel)
    best_results.loc[(family, name, "")] = [
        res, res.llf, res.df_model, res.aic, res.bic, mse, msel, f
    ]


def var_plot(ax, model, type="mean"):
    if type == "mean":
        pred = model.predict()
    elif type == "rate":
        pred = model.predict() / df["Exposure"].to_numpy()
    resid = model.resid_pearson / np.sqrt(model.scale)

    bins = np.logspace(np.log10(pred.min()), np.log10(pred.max()), num=25)
    counts = np.histogram(pred, bins)[0]
    centers = (bins[:-1] + bins[1:]) / 2

    d = pd.DataFrame({"pred": pred, "resid": resid})
    d.sort_values("pred", inplace=True)

    d["S"] = d["resid"].rolling(100, center=True, win_type="gaussian").std(std=400)
    d["M"] = d["resid"].rolling(100, center=True, win_type="gaussian").mean(std=400)
    d["U"] = d["M"] + 1.96 * d["S"]
    d["L"] = d["M"] - 1.96 * d["S"]

    ax.scatter(d["pred"], d["resid"], alpha=0.2)

    ax.plot(d["pred"][50:-50], d["U"][50:-50], color="black", linestyle="--")
    ax.plot(d["pred"][50:-50], d["M"][50:-50], color="black", linestyle="-")
    ax.plot(d["pred"][50:-50], d["L"][50:-50], color="black", linestyle="--")

    ax.fill_between(x=centers, y1=-10, y2=-10 + counts/100, alpha=0.2)



fig, axs = plt.subplots(1, 4, figsize=(10,3), sharey=True)

for k in range(4):
    ax = axs[k]
    row = best_results.iloc[k]
    var_plot(ax, row["model"], "mean")
    if k == 0:
        ax.set_ylabel("Normalized Pearson residuals")
    ax.set_xlabel("Fitted mean")
    ax.set_title(["Normal (log link)", "Poisson", "Negative Binomial", "Tweedie(1.5)"][k], loc="left")
    ax.set_xscale("log")

plt.tight_layout()
plt.savefig("./tmp/resid_plot.pdf")




