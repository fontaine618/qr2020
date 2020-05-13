import pandas as pd
import numpy as np
from calendar import monthrange

cal = pd.DataFrame(
    {
        "Year": np.linspace(2007, 2018.999, 144).astype(int),
        "Month": [*range(1, 13)] * 12
    }
)

cal["Days"] = cal.apply(lambda x: monthrange(x["Year"], x["Month"])[1], 1)

cal.index = pd.to_datetime(
    pd.DataFrame({
        "year": cal["Year"],
        "month": cal["Month"],
        "day": 15
    }),
    format="%M-%Y"
)

cal.to_csv("./data/processed/days_per_month.csv")