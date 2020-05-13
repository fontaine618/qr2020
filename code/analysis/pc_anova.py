import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, te, f, intercept, l
from sklearn.preprocessing import LabelBinarizer
from statsmodels.gam.api import BSplines, GLMGam
import patsy
from statsmodels.api import OLS
plt.style.use("seaborn")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

pc = pd.read_csv("./data/processed/pc.csv")


# encoder = LabelBinarizer()
# encoder.fit(pc["Sex"])
# pc["Sex"] = encoder.transform(pc["Sex"])




components = ["PC{}".format(i+1) for i in range(3)]

midpoints = ["Midpoint", "Midpoint_linear"]

formulas = ["1", "1s", "1a", "1s+a", "1s*a"]

n_splines = [5, 7, 9]


results = pd.DataFrame(
    columns=["model", "LogLik", "AIC", "BIC", "MSE"],
    index=pd.MultiIndex(levels=[[], [], [], []],
                    codes=[[], [], [], []],
                    names=["Component", "Midpoint", "Formula", "N Splines"])
)

models = itertools.product(components, midpoints, formulas, n_splines)
for component, midpoint, formula, n_spline in models:
    print("=" * 80)
    print(component, midpoint, formula, n_spline)
    df = pc[[component, "Sex", midpoint]]
    df.columns = ["PC", "Sex", "Age"]
    if formula == "1":
        y, X = patsy.dmatrices("PC ~ 1", data=df)
    elif formula == "1s":
        y, X = patsy.dmatrices("PC ~ 1 + C(Sex)", data=df)
    elif formula == "1a":
        y, X = patsy.dmatrices("PC ~ 1 + bs(Age, df={})".format(n_spline), data=df)
    elif formula == "1s+a":
        y, X = patsy.dmatrices("PC ~ 1 + C(Sex) + bs(Age, df={})".format(n_spline), data=df)
    elif formula == "1s*a":
        y, X = patsy.dmatrices("PC ~ 1 + C(Sex) * bs(Age, df={})".format(n_spline), data=df)
    model = OLS(y, X).fit()
    aic = model.aic
    bic = model.bic
    llk = model.llf
    mse = model.mse_resid
    print(llk, aic, bic, mse)
    results.loc[(component, midpoint, formula, n_spline)] = [model, llk, aic, bic, mse]

print(results.drop(columns="model").round(3))







