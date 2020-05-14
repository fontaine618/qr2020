import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.style.use("seaborn")
sys.path.append("/home/simon/Documents/sithon")
from sithon.latex.pd_to_latex import pd_to_tabular

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 1000)


midpoints = pd.read_csv("./data/processed/midpoints.csv")
midpoints = midpoints.groupby(["Age_group"]).agg("mean")
tab = midpoints.T

tab = tab.applymap("{:.2f}".format)

tab.columns = [age[:2] + "-" + age[-2:] for age in tab.columns]
tab.index.name = ""
tab.index = ["Midpoint", "Estimated"]


table = pd_to_tabular(
    tab.iloc[:, -10:],
    title="Age Groups Midpoint Estimation",
    column_format="l" + "c"*10
)

print(table)

with open("./report/tables/midpoints.tex", "w") as f:
    f.write(table)


# x = midpoints.iloc[:18, 3].values
# m_long = midpoints.set_index(["Date", "Sex", "Age_group"])["Midpoint_linear"].unstack(2)
#
#
# plt.figure()
# ax = plt.gca()
#
# for i in range(m_long.shape[1]):
#     ax.plot(x, m_long.iloc[i, :].values - x)
#
# plt.savefig("./tmp/midpoints.pdf")

