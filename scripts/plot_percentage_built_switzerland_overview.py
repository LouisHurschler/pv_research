import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
from get_tarif_data import truncate_colormap


def get_percentage_built_from_data(
    data: gpd.GeoDataFrame, classes_to_consider: list = [1, 2, 3, 4, 5]
) -> gpd.GeoDataFrame:
    data = data[data["KLASSE2"].isin(classes_to_consider)]
    total_counts = data.groupby("GDE_NAME")["FLAECHE"].sum().rename("total")
    with_pv_counts = (
        data[data["SubCategory"] == "with_PV"]
        .groupby("GDE_NAME")["FLAECHE"]
        .sum()
        .rename("count_with")
    )

    result = pd.concat([with_pv_counts, total_counts], axis=1).fillna(0)
    result.loc[result["total"] == 0, "total"] = (
        1  # when no roofs in the specific class are available, store 0 in result (no NaN)
    )
    result["percentage"] = (result["count_with"] / result["total"]) * 100.0
    # print(result)
    return result


# some municipalities has merged or changed the name. Therefore, the name in the switzerland dataset (sonnendach) should be changed to emrge them properly.
# note that this is a source for errors but it shouldn't be that big because it only includes municipalities
# which merged later than 2016
def change_names(data_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    to_change = {
        "Albula/Alvra": "Alvaneu",
        "Arzier-Le Muids": "Arzier",
        "Belmont-Broye": "Domdidier",
        "Bergün Filisur": "Bergün/Bravuogn",
        "Bussigny": "Bussigny-près-Lausanne",
        "Calanca": "Arvigo",
        "Cheyres-Châbles": "Cheyres",
        "Crans-Montana": "Montana",
        "Crans (VD)": "Montana",
        "Domleschg": "Almens",
        "Estavayer": "Estavayer-le-Lac",
        "Gibloux": "Corpataux-Magnedens",
        "Goms": "Münster-Geschinen",
        "Jorat-Mézières": "Mézières (VD)",
        "La Grande-Béroche": "Bevaix",
        "La Grande Béroche": "Bevaix",
        "Mont-Vully": "Bas-Vully",
        "Obersaxen Mundaun": "Obersaxen",
        "Petit-Val": "Monible",
        "Prez": "Prez-vers-Noréaz",
        "Péry-La Heutte": "Péry",
        "Rheinwald": "Splügen",
        "Riviera": "Cresciano",
        "Stammheim": "Oberstammheim",
        "Surses": "Bivio",
        "Thurnen": "Kirchenthurnen",
        "Valbirse": "Bévilard",
        "Verzasca": "Brione (Verzasca)",
        "Verzasca 3)": "Brione (Verzasca)",
        "Villaz": "Villaz-Saint-Pierre",
        # "Balzers": "", all Liechtenstein
        # "Eschen": "",
        # "Gamprin": "",
        # "Mauren": "",
        # "Planken": "",
        # "Ruggell": "",
        # "Schaan": "",
        # "Schellenberg": "",
        # "Triesen": "",
        # "Triesenberg": "",
        # "Vaduz": "",
    }
    for old_name, new_name in to_change.items():
        data_boundary.loc[
            data_boundary["municipality"] == old_name, "municipality"
        ] = new_name

    return data_boundary


if __name__ == "__main__":

    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    data_boundaries = change_names(gpd.read_file("data/muni_boundaries.gpkg"))
    # data_boundaries.plot()
    # plt.show()

    data_energy_prices = gpd.read_file("data/municipal_and_energy_data.gpkg")
    data_switzerland = gpd.read_file("out/households_5.0_KLASSE2.gpkg")
    custom_cmap = truncate_colormap(cm.viridis, -0.6, 0.8)
    # munis = sorted(
    #     data_switzerland["GDE_NAME"].unique(),
    #     key=lambda x: (
    #         x is not None,
    #         x,
    #     ),  # this pushes None values to the back?
    # )
    # for m in munis:
    #     print(m)

    data_switzerland = get_percentage_built_from_data(
        data_switzerland, classes_to_consider=[1, 2, 3, 4, 5]
    )

    merged = data_boundaries.merge(
        data_switzerland,
        how="left",
        left_on="municipality",
        right_on="GDE_NAME",
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    merged.plot(
        "percentage",
        legend=True,
        cmap="viridis",
        ax=ax1,
        legend_kwds={
            "label": "PV Roof Area Coverage [%]",
            "shrink": 0.6,
            "orientation": "vertical",
            # "loc": "left",
        },
    )
    ax1.axis("off")
    # plt.title("Rooftop Area PV Coverage of Switzerland")

    mean_energy_prices = (
        data_energy_prices.groupby("municipality")["energy"]
        .mean()
        .rename("mean_energy")
    )
    merged_with_energy_prices = data_energy_prices.drop_duplicates(
        "municipality"
    ).merge(mean_energy_prices, on="municipality", how="left")
    # NOTE: fill missing values with 6.0 here to have good visual plot.
    merged_with_energy_prices = merged_with_energy_prices.fillna(8.5)

    merged_with_energy_prices.plot(
        column="mean_energy",
        cmap=custom_cmap,
        legend=True,
        ax=ax2,
        legend_kwds={
            "label": "mean energy price 2014 - 2024 [Rp./kWh]",
            "shrink": 0.6,
            "orientation": "vertical",
            # "loc": "right",
        },
        # linewidth=0.0,
    )
    # fig.suptitle(
    #     "Comparison Energy Prices vs. Roof Are Coverage of PV Installation"
    # )
    ax2.axis("off")

    list_unique_names_switzerland = sorted(data_switzerland.index.unique())
    list_unique_names_boundary = sorted(
        data_boundaries["municipality"].unique()
    )
    print("in boundary but not in switzerland:")
    for name in list_unique_names_boundary:
        if name not in list_unique_names_switzerland:
            print(name)
    plt.tight_layout()
    plt.savefig("out/merged_plot.jpg", dpi=1200)

    # print(len(data_boundaries["municipality"].unique()))
    # print(len(data_switzerland["GDE_NAME"].unique()))
