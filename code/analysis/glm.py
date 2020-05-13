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

df = mortality_long[["Deaths", "Sex", "Midpoint_linear", "Year", "Month", "Exposure"]]
df.columns = ["Deaths", "Sex", "Age", "Year", "Month", "Exposure"]
df["FDate"] = df["Year"] + (df["Month"] - 1) / 12

df.to_csv("./data/processed/df_for_glm.csv")


families = ["log_normal", "poisson", "NB"]


formulas = {
    "A*(Y+S*M)": "Deaths ~ bs(Age, n_age) * (bs(FDate, n_date) + C(Sex) * bs(Month, n_month))",
    "A*(Y+S+M)": "Deaths ~ bs(Age, n_age) * (bs(FDate, n_date) + C(Sex) + bs(Month, n_month))",
    #"A*(Y+M)": "Deaths ~ bs(Age, n_age) * (bs(FDate, n_date) + bs(Month, n_month))",
}

models = itertools.product(
    families, formulas.items(), [18],  [25, 50],  [8, 12],
)

results = pd.DataFrame(
    columns=["model", "LogLik", "DF_Model", "AIC", "BIC", "MSE", "MSEL", "MSELR", "Formula"],
    index=pd.MultiIndex(levels=[[]]*5,
                    codes=[[]]*5,
                    names=["familiy", "formula", "n_age", "n_date", "n_month"])
)

for family, (name, formula), n_age, n_date, n_month in models:
    print("=" * 80)
    vals = {"n_age": n_age, "n_date": n_date, "n_month": n_month}
    f = formula
    for k, v in vals.items():
        f = f.replace(k, str(v))
    print(name, f)
    y, X = patsy.dmatrices(f, data=df)
    exp = df["Exposure"].to_numpy()
    if family == "poisson":
        glm = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exp)
    elif family == "NB":
        glm = sm.GLM(y, X, family=sm.families.NegativeBinomial(), exposure=exp)
    elif family == "log_normal":
        fm = sm.families.Gaussian(link=sm.families.links.log())
        glm = sm.GLM(y.ravel()/exp, X, family=fm, exposure=exp)
    res = glm.fit()
    pred = res.predict().reshape((-1, 1))
    if family == "log_normal":
        pred = pred * exp.reshape((-1, 1))
    mse = np.mean((y - pred)**2)
    msel = np.mean((np.log(y) - np.log(pred))**2)
    mselr = np.mean((np.log(y/exp) - np.log(pred/exp))**2)
    print("LogLik", res.llf)
    print("DF Model", res.df_model)
    print("AIC", res.aic)
    print("BIC", res.bic)
    print("MSE", mse)
    print("MSEL", msel)
    print("MSELR", mselr)
    results.loc[(family, name, n_age, n_date, n_month)] = (
        res, res.llf, res.df_model, res.aic, res.bic, mse, msel, mselr, f
    )

print(results.drop(columns=["model", "Formula"]))

results.drop(columns="model").to_csv("./data/results/NB_models.csv")

x = res.predict()
pred = res.predict()
z = np.polyfit(x, y, 2)
p = np.polynomial.polynomial.Polynomial(z, domain=[0, 60000])
xs = np.linspace(0, 55000)

plt.figure()

plt.scatter(x, pred-y)

plt.plot(xs, poisson.ppf(q=0.025, mu=xs) - xs)
plt.plot(xs, poisson.ppf(q=0.25, mu=xs) - xs)
plt.plot(xs, poisson.ppf(q=0.75, mu=xs) - xs)
plt.plot(xs, poisson.ppf(q=0.975, mu=xs) - xs)

plt.savefig("./tmp/poisson.pdf")