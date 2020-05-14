import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import patsy
import statsmodels.api as sm
import itertools
from scipy.stats import poisson
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

mortality_long = pd.read_csv("./data/processed/mortality_long.csv", index_col=0)

days_per_month = pd.read_csv("./data/processed/days_per_month.csv", index_col=0)

# merge with number of days
mortality_long = mortality_long.merge(
    days_per_month["Days"],
    how="left",
    left_on="Date",
    right_index=True
)

mortality_long["Exposure"] = mortality_long["Est_Pop_Lin"] * mortality_long["Days"]

df = mortality_long[["Deaths", "Sex", "Age_group", "Midpoint_linear", "Year", "Month", "Exposure", "Days"]]

cumdays = df.groupby(["Sex", "Age_group", "Year"]).apply(
    lambda d: (d["Days"].cumsum() - d["Days"][0] + 15) / d["Days"].sum()
)

df["FDate"] = df["Year"].values + cumdays.values
df.columns = ["Deaths", "Sex", "Age_group", "Age", "Year", "Month", "Exposure", "Days", "FDate"]

df.to_csv("./data/processed/df_for_glm.csv")


families = ["logNormal", "Poisson", "NB", "Tweedie"]

formulas = {
    "sA*(sD+S*sM)": "Deaths ~ bs(Age, 9) * (bs(FDate, 50) + C(Sex) * bs(Month, 7))",
    "cA*(sD+S*sM)": "Deaths ~ C(Age_group) * (bs(FDate, 50) + C(Sex) * bs(Month, 7))",
    "sA*(sD+S*cM)": "Deaths ~ bs(Age, 9) * (bs(FDate, 50) + C(Sex) * C(Month))",
    "cA*(sD+S*cM)": "Deaths ~ C(Age_group) * (bs(FDate, 50) + C(Sex) * C(Month))",
    "sA*(sY+S*sM)": "Deaths ~ bs(Age, 9) * (bs(Year, 7) + C(Sex) * bs(Month, 7))",
    "cA*(sY+S*sM)": "Deaths ~ C(Age_group) * (bs(Year, 7) + C(Sex) * bs(Month, 7))",
    "sA*(sY+S*cM)": "Deaths ~ bs(Age, 9) * (bs(Year, 7) + C(Sex) * C(Month))",
    "cA*(sY+S*cM)": "Deaths ~ C(Age_group) * (bs(Year, 7) + C(Sex) * C(Month))",
}

models = itertools.product(
    families, formulas.items(),
)

results = pd.DataFrame(
    columns=["model", "LogLik", "DF_Model", "AIC", "BIC", "MSE", "MSEL", "Formula"],
    index=pd.MultiIndex(levels=[[]]*3,
                    codes=[[]]*3,
                    names=["family", "formula", "dummy"])
)

for family, (name, formula) in models:
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
    res = glm.fit()
    pred = res.predict().reshape((-1, 1))
    mse = np.mean((y - pred)**2)
    msel = np.mean((np.log(y) - np.log(pred))**2)
    print("LogLik", res.llf)
    print("DF Model", res.df_model)
    print("AIC", res.aic)
    print("BIC", res.bic)
    print("MSE", mse)
    print("MSEL", msel)
    results.loc[(family, name, "")] = [
        res, res.llf, res.df_model, res.aic, res.bic, mse, msel, f
    ]

bests = results.loc[results["MSE"].le(80000) & results["MSEL"].le(0.003)]
print(bests[["AIC", "BIC", "MSE", "MSEL"]].applymap("{:.4f}".format))

results.drop(columns="model").to_csv("./data/results/glm_models.csv")


best_models = results.iloc[[2, 10, 16, 24]]
print(best_models[["MSE", "MSEL"]])
best_models.drop(columns="model").to_csv("./data/results/glm_best_models.csv")

# x = res.predict()
# pred = res.predict()
# z = np.polyfit(x, y, 2)
# p = np.polynomial.polynomial.Polynomial(z, domain=[0, 60000])
# xs = np.linspace(0, 55000)
#
# plt.figure()
#
# plt.scatter(x, pred-y)
#
# plt.plot(xs, poisson.ppf(q=0.025, mu=xs) - xs)
# plt.plot(xs, poisson.ppf(q=0.25, mu=xs) - xs)
# plt.plot(xs, poisson.ppf(q=0.75, mu=xs) - xs)
# plt.plot(xs, poisson.ppf(q=0.975, mu=xs) - xs)
#
# plt.savefig("./tmp/poisson.pdf")