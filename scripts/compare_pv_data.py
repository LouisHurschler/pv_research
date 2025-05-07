import geopandas as gpd
import os
import matplotlib.pyplot as plt
import pandas as pd


# newest entry: 2024-10-26 00:00:00
name_old = os.path.join("data", "ch.bfe.elektrizitaetsproduktionsanlagen.gpkg")

# newest entry: 2025-03-14 00:00:00
name_new = os.path.join("data", "elektrizitaetsproduktionsanlagen_020425.gpkg")

data_old = gpd.read_file(name_old)
data_new = gpd.read_file(name_new)
unique_data_old = data_old[~data_old["xtf_id"].isin(data_new["xtf_id"])]
unique_data_new = data_new[~data_new["xtf_id"].isin(data_old["xtf_id"])]
# print("unique old: ", unique_data_old)
# print("unique new: ", unique_data_new)


def cleanup(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data = data[data["SubCategory"] == "subcat_2"].copy()
    data = data[["BeginningOfOperation", "TotalPower"]]
    data["BeginningOfOperation"] = pd.to_datetime(
        data["BeginningOfOperation"], errors="coerce"
    )
    data = data.set_index("BeginningOfOperation")

    # merge multiple same indexes
    data = data.groupby(data.index).sum()

    data = data.sort_index()
    return data


unique_data_old = cleanup(unique_data_old)
unique_data_new = cleanup(unique_data_new)

data_old = cleanup(data_old)
data_new = cleanup(data_new)
# print("newest entry old: ", data_old.index[-1])
# print("newest entry new: ", data_new.index[-1])

unique_data_old["TotalPower"] = -unique_data_old["TotalPower"]

changes = pd.concat([unique_data_old, unique_data_new])
changes = changes.sort_index()
plt.axvline(x=pd.to_datetime("2024-10-26"), color="red")
# print(changes)
plt.plot(changes["TotalPower"].cumsum(), label="difference datasets")


plt.plot(data_old["TotalPower"].cumsum(), label="data from old dataset")
plt.plot(data_new["TotalPower"].cumsum(), label="data from new dataset")
# plt.plot(unique_data_old["TotalPower"].cumsum())
# plt.plot(unique_data_new["TotalPower"].cumsum())

plt.grid()
plt.legend()
plt.show()
