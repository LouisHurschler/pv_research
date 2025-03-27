import geopandas as gpd
import os
from preprocess_data import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    name_dataset_roofs = os.path.join("data", "SOLKAT_DACH.gpkg")
    name_dataset_power = os.path.join(
        "data", "ch.bfe.elektrizitaetsproduktionsanlagen.gpkg"
    )

    data_roof = gpd.read_file(name_dataset_roofs)
    data_power = gpd.read_file(name_dataset_power)

    data_pv = drop_not_pv(data_power)
    # data_merged = join_roof_with_power(data_roof, data_pv, print_infos=True)

    # print(data_pv)
    data_pv = data_pv[data_pv["TotalPower"] < 100]
    # print(data_pv)

    power_before = data_pv["TotalPower"].sum()
    tmp_data = data_pv["TotalPower"]
    tmp_data.index = np.asarray(
        data_pv["BeginningOfOperation"], dtype="datetime64[s]"
    )
    # tmp_data.set_index("BeginningOfOperation", inplace=True)
    tmp_data = tmp_data.sort_index()
    plt.plot(tmp_data.cumsum())

    data_merged = join_roof_with_power(data_roof, data_pv, print_infos=False)
    power_after = data_merged["TotalPower"].sum()
    print(power_after / power_before)
    tmp_data_2 = data_merged["TotalPower"]
    tmp_data_2.index = np.asarray(
        data_merged["BeginningOfOperation"], dtype="datetime64[s]"
    )
    # tmp_data_2.set_index("BeginningOfOperation", inplace=True)
    tmp_data_2 = tmp_data_2.sort_index()
    plt.plot(tmp_data_2.cumsum())
    plt.plot(tmp_data.cumsum() - tmp_data_2.cumsum())
    plt.show()
