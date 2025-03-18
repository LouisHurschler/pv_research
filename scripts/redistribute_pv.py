import geopandas as gpd
import numpy as np
import time
import os
from tqdm import tqdm


def redistribute_pv_power_to_best_available_roof(
    data: gpd.GeoDataFrame,
    used_area_m2_per_kwp: float = 4.0,
    min_area_m2: float = 2.0,
    name_class: str = "KLASSE",
) -> gpd.GeoDataFrame:
    """
    This function should project the pv of one household (definded by the SB_UUID) onto the 'best' roof(s)
    It starts by calculating the area the pv plant is using as
    area_using =  Power * used_area_m2_per_kwp,
    Then it 'fills up' the roofs according to roof['FLAECHE']
    It only includes rooftops with an area of at least min_area_m2 for this projection

    If the area is not sufficient, it will distribute the plant on the whole buildding evenly, weighted by the area

    The BeginningOfOperation value is set to some random BeginningOfOperation value of this household
    TODO: make it faster
    """
    total_len = len(data["SB_UUID"].unique())
    print(total_len)
    idx = 0
    # data_with = data[data["SubCategory"] == "with_PV"]
    could_fit = 0
    could_not_fit = 0
    print(
        f"total power before redistributing: {data["TotalPower"].sum(skipna=True)}"
    )
    print("starting redistributiong pv plants")
    start_time = time.time()

    idx = 0

    print("start sorting values")
    data.sort_values(by="SB_UUID", inplace=True)
    print("values sorted")

    # assign without_pv text and 0 TotalPower at the end
    without_pv_indexes = []
    print("change to category")
    # change type to category such that no new text has to be stored which increases the efficiency
    data["BeginningOfOperation"] = data["BeginningOfOperation"].astype(
        "category"
    )
    data["SubCategory"] = data["SubCategory"].astype("category")
    print("changed to category")

    for uuid, data_household in tqdm(data.groupby("SB_UUID"), total=total_len):
        if data_household["SubCategory"].iloc[0] == "without_PV":
            continue

        total_power = data_household["TotalPower"].sum()
        area_used_for_pv = total_power * used_area_m2_per_kwp
        areas = data_household["FLAECHE"]
        total_area_available = areas[areas > min_area_m2].sum()
        beginning_of_operation = (
            data_household["BeginningOfOperation"].dropna().iloc[0]
        )

        if total_area_available < area_used_for_pv:
            # print(uuid, total_area_available, area_used_for_pv)
            individual_used_area_m2_per_kwp = (
                total_area_available / total_power
            )
            could_not_fit += 1
        else:
            individual_used_area_m2_per_kwp = used_area_m2_per_kwp
            could_fit += 1
        # TODO: add beginningOfOperation, do the same with KLASSE2 (or add an argument which type of class to use)
        # what about initial power? How should it be handled if there are different beginningOfOperations?

        for klasse in range(5, 0, -1):
            data_klasse = data_household[data_household[name_class] == klasse]

            for idx in data_klasse.index:
                if area_used_for_pv == 0:
                    without_pv_indexes.append(idx)
                    # data.loc[idx, "TotalPower"] = 0.0
                    # data.loc[idx, "SubCategory"] = "without_PV"
                    continue

                row = data.loc[idx]
                loc_area = row["FLAECHE"]

                if loc_area < min_area_m2:
                    without_pv_indexes.append(idx)
                    # data.loc[idx, "TotalPower"] = 0.0
                    # data.loc[idx, "SubCategory" = "without_PV"
                    continue
                if area_used_for_pv > loc_area:
                    # whole area of roof can be used
                    area_used_for_pv -= loc_area
                    data.loc[idx, "TotalPower"] = np.float32(
                        loc_area / individual_used_area_m2_per_kwp
                    )
                    data.loc[idx, "BeginningOfOperation"] = (
                        beginning_of_operation
                    )

                else:
                    data.loc[idx, "TotalPower"] = np.float32(
                        area_used_for_pv / individual_used_area_m2_per_kwp
                    )
                    data.loc[idx, "BeginningOfOperation"] = (
                        beginning_of_operation
                    )
                    area_used_for_pv = 0

    print(f"time to redistribute pv plants: {time.time() - start_time}")
    print(
        f"total power after redistributing: {data["TotalPower"].sum(skipna=True)}"
    )
    data.loc[without_pv_indexes, "SubCategory"] = "without_PV"
    data.loc[without_pv_indexes, "TotalPower"] = 0.0
    # TODO: how could you do this with the beginningOfOperation stuff? It changes for every entry...

    print(
        f"amount of households where it fit: {float(could_fit) /(could_fit + could_not_fit)}"
    )

    data.to_file(
        os.path.join(
            "out", f"households_{used_area_m2_per_kwp}_{name_class}.gpkg"
        )
    )
    return data
