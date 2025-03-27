import pandas as pd
import geopandas as gpd
from plot_percentage_built_switzerland_overview import (
    get_percentage_built_from_data,
    change_names,
)
import os
import numpy as np
import matplotlib.pyplot as plt


def get_energy_and_installation_rate_data() -> gpd.GeoDataFrame:
    data_energy_prices = gpd.read_file("data/municipal_and_energy_data.gpkg")
    data_switzerland = gpd.read_file("out/households_5.0_KLASSE2.gpkg")
    data_switzerland = get_percentage_built_from_data(
        data_switzerland, classes_to_consider=[1, 2, 3, 4, 5]
    )
    data_boundaries = change_names(gpd.read_file("data/muni_boundaries.gpkg"))

    merged = data_boundaries.merge(
        data_switzerland,
        how="left",
        left_on="municipality",
        right_on="GDE_NAME",
    )

    mean_energy_prices = (
        data_energy_prices.groupby("municipality")["energy"]
        .mean()
        .rename("mean_energy")
    )
    # print(merged)
    # print(mean_energy_prices)
    merged = merged.merge(mean_energy_prices, how="left", on="municipality")
    merged = merged[["mean_energy", "percentage"]].dropna()
    print(merged)
    return merged


# test some correlation between energy prices and installation coverage
gpd.options.io_engine = "pyogrio"
os.environ["PYOGRIO_USE_ARROW"] = "1"

data = get_energy_and_installation_rate_data()

# ss_res = np.sum((data["mean_energy"] - data["percentage"]) ** 2)
# ss_tot = np.sum((np.mean(data["percentage"]) - data["percentage"]) ** 2)
# r_squared = 1 - (ss_res / ss_tot)
print(data["mean_energy"].corr(data["percentage"]))
plt.plot(data["mean_energy"], data["percentage"], linestyle="none", marker=".")
plt.show()
