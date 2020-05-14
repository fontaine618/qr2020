import pandas as pd
import sys
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


tab = pd.read_csv("./data/results/glm_models.csv")

tab.set_index("formula", inplace=True)
groups = tab["family"]
tab.drop(columns=["Formula", "dummy", "family"], inplace=True)

tab[["LogLik", "AIC", "BIC", "MSE"]] = tab[["LogLik", "AIC", "BIC", "MSE"]].applymap("{:.0f}".format)
tab[["MSEL"]] = tab[["MSEL"]].applymap("{:.05f}".format)

tab.index = ["\\texttt{" + f +"}" for f in tab.index]
tab.index.name = "Formula"
tab.columns = ['Log likelihood', 'Df Model', 'AIC', 'BIC', 'MSE', 'MSEL']
tab = tab.groupby(groups.values)

table = pd_to_tabular(
    tab,
    title="Mortality Rate: GLM Results",
    column_format="lrrrrrr",
    group_names={
        "logNormal": "Normal (log link)",
        "Poisson": "Poisson",
        "NB": "Negative Binomial",
        "Tweedie": "Tweedie (1.5)"},
    group_name_align="l",
    escape=False
)

print(table)

with open("./report/tables/model_selection.tex", "w") as f:
    f.write(table)



tab = pd.read_csv("./data/results/glm_best_models.csv")

tab.set_index("family", inplace=True)
tab.drop(columns=["Formula", "dummy", "AIC", "BIC", "LogLik"], inplace=True)

tab[["MSE"]] = tab[["MSE"]].applymap("{:.0f}".format)
tab[["MSEL"]] = tab[["MSEL"]].applymap("{:.05f}".format)

tab.index = ["Normal (log link)", "Poisson", "Negative Binomial", "Tweedie (1.5)"]
tab.index.name = "Family"
tab.columns = ["Formula", 'Df Model', 'MSE', 'MSEL']

tab["Formula"] = ["\\texttt{" + f +"}" for f in tab["Formula"]]

table = pd_to_tabular(
    tab,
    title="Mortality Rate: GLM Results (Best model per family)",
    column_format="llrrr",
    escape=False
)

print(table)

with open("./report/tables/best_models.tex", "w") as f:
    f.write(table)