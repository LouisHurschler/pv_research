import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from plot_percentage_built_switzerland_overview import (
    get_percentage_built_from_data,
    change_names,
)

gpd.options.io_engine = "pyogrio"
os.environ["PYOGRIO_USE_ARROW"] = "1"


data_switzerland = pd.read_excel("data/je-d-21.03.01.xlsx", header=5)

# cleanup data from excel
# remove unnececary information
data_switzerland = data_switzerland[3:-16]

# replace nonnumeric values with mean
# Convert to numeric, forcing errors to NaN for non-numeric entries
muni_id = data_switzerland["Gemeindename"]
data_switzerland = data_switzerland.drop(columns=["Gemeindename"])
data_switzerland = data_switzerland.apply(pd.to_numeric, errors="coerce")

# Replace NaNs (i.e., non-numeric values) with column means
data_switzerland = data_switzerland.apply(lambda col: col.fillna(col.mean()))
data_switzerland["municipality"] = muni_id
data_switzerland = change_names(data_switzerland)

data_installation_rate = gpd.read_file("out/households_5.0_KLASSE2.gpkg")
data_installation_rate = get_percentage_built_from_data(data_installation_rate)

merged = data_installation_rate.merge(
    data_switzerland,
    how="left",
    left_on="GDE_NAME",
    right_on="municipality",
)
data_boundaries = change_names(gpd.read_file("data/muni_boundaries.gpkg"))
merged = data_boundaries.merge(
    merged,
    how="left",
    on="municipality",
)
# print(merged)


def calculate_correlation(data: pd.DataFrame, left_col: str, right_col: str):

    try:
        correlation = data[left_col].corr(data[right_col])

        if abs(correlation) > 0.3:
            print(
                f"correlation between {left_col} and {right_col} is {correlation}"
            )
            merged[left_col] = merged[left_col].astype(float)
            merged.plot(column=left_col, cmap="viridis", legend=True)
            plt.show()

    except Exception as e:
        print(e)
        print(left_col)

        # plt.plot(data[left_col], data[right_col], linestyle="none", marker=".")
        # plt.show()


for key in merged.keys():
    if key in [
        "municipality_id",
        "municipality",
        "count_with",
        "total",
        "percentage",
        "easting",
        "northing",
        "geometry",
    ]:
        continue

    calculate_correlation(merged, key, "percentage")


# tested some linear regression models, r2 ~ 0.4
# merged["percentage"] = pd.to_numeric(merged["percentage"], errors="coerce")
# print(merged)
# merged = merged.dropna(subset=["percentage"])

# percentages = merged["percentage"]


# merged.drop(
#     columns=["municipality", "count_with", "total", "percentage"], inplace=True
# )


# model = LinearRegression()
# model.fit(merged, percentages)
# y_pred = model.predict(merged)
# r_squared = model.score(merged, percentages)

# coef_df = pd.DataFrame({"feature": merged.columns, "coefficient": model.coef_})
# print(coef_df)
# print(r_squared)
# plt.plot(y_pred, percentages, linestyle="none", marker=".")
# plt.show()
