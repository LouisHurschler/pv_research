import geopandas as gpd
import os
import time


# TODO: currently if several pv plant datapoints are on the same roof, the roof gets duplicated for each plant.
# if there is no roof available at the position of the datapoint (why? found some with pv plants from 2019, probably renovated? built new?) it gets deleted
# TODO: assign sum in these cases. We don't want to have duplicates because then the roof areas will get duplicated as well.
def join_roof_with_power(
    data_roof: gpd.GeoDataFrame,
    data_power: gpd.GeoDataFrame,
    print_infos: bool = False,
) -> gpd.GeoDataFrame:

    if print_infos:
        start_time = time.time()
    if data_roof["DF_UID"].duplicated().any():
        print("Warning: Duplicates found in DF_UID of data_roof")

    if print_infos:
        print(
            f"time to check if no duplicates: {time.time() - start_time} seconds"
        )
        print(
            f"Sum of total Power before merging: {data_power["TotalPower"].sum()}"
        )
        start_time = time.time()

    # TODO: check if there are any duplicates in the DF_UID member
    merged = gpd.sjoin(
        data_roof,
        data_power,
        how="left",
    )
    if print_infos:
        print(f"time to join: {time.time() - start_time} seconds")
        start_time = time.time()
        print(
            f"Sum of total Power after merging: {merged["TotalPower"].sum()}.\nThe difference is the amount of PV on houses built after 2022"
        )

        # Group by DF_UID and aggregate
        print(merged)
        print(type(merged))

    # inspired by chatgpt. The problem of agg is that it stores the result as DataFrame and we have
    # add the geometry afterwards again
    # Preserve geometry column
    geometry_col = merged.geometry.name

    # Group by DF_UID and aggregate
    merged = merged.groupby("DF_UID", as_index=False).agg(
        {
            "InitialPower": "sum",
            "TotalPower": "sum",
            **{
                col: "first"
                for col in merged.columns
                if col
                not in ["DF_UID", "InitialPower", "TotalPower", geometry_col]
            },
        }
    )
    if print_infos:
        print(f"time to remove duplicates: {time.time() - start_time} seconds")
        start_time = time.time()

    # Restore geometry
    merged = gpd.GeoDataFrame(
        merged,
        geometry=data_roof.set_index("DF_UID")
        .geometry.loc[merged["DF_UID"]]
        .values,
        crs=data_roof.crs,
    )
    if print_infos:
        print(
            f"Sum of total Power after cleanup: {merged["TotalPower"].sum()}"
        )

        print(f"time to restore geometry: {time.time() - start_time} seconds")
        print(merged)
        print(type(merged))
    return merged


def drop_not_pv(
    data_power: gpd.GeoDataFrame, threshold_max_power: float = 100
) -> gpd.GeoDataFrame:
    data_pv = data_power[data_power["SubCategory"] == "subcat_2"]
    data_pv = data_power[data_power["TotalPower"] <= threshold_max_power]
    return data_pv


def add_municipality_data(
    data_merged: gpd.GeoDataFrame, data_muni: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # add another rsuffix because the first join already adds right

    data_merged["centroid"] = data_merged.geometry.centroid  # Compute centroid
    # print("centroid added")
    merged = gpd.sjoin(
        # data_merged,
        data_merged.set_geometry("centroid"),
        data_muni,
        how="left",
        predicate="within",
        rsuffix="right_muni",
    )
    # print("merged")
    merged = merged.drop(columns=["centroid"]).set_geometry("geometry")
    # print("dropped")
    return merged


# if PV is installed on one roof, show it on other roofs too
# Note that BeginningOfOperations and TotalPower are only stored in one Roof
def merge_households(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # TODO: merge households differently? GWR_EGID would be great but is not defined for a lot of homes...
    data["SubCategory"] = data.groupby("SB_UUID")["SubCategory"].transform(
        "first"
    )
    return data


def cleanup_data(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    list_of_keys = list(data.keys())
    # print(list_of_keys)
    list_to_keep = [
        "DF_UID",
        "SB_UUID",
        "KLASSE",
        "FLAECHE",
        "FLAECHE_KOLLEKTOREN",
        "GWR_EGID",
        "InitialPower",
        "TotalPower",
        "SubCategory",
        "GDE_NAME",
        "KT_KZ",
        "geometry",
        "BeginningOfOperation",
        "STROMERTRAG",
        "NEIGUNG",
    ]
    for keep in list_to_keep:
        if keep not in list_of_keys:
            print(f"{keep} not in keys!")
        else:
            list_of_keys.remove(keep)
    data.drop(
        columns=list_of_keys,
        inplace=True,
    )
    print("cols_removed")
    data.loc[data["SubCategory"] == "subcat_2", "SubCategory"] = "with_PV"
    data.loc[data["SubCategory"].isna(), "SubCategory"] = "without_PV"
    # TODO: rename key?
    return data


# adding a second class identifier, where the roofs with NEIGUNG <= THRSHOLD will be increased by 1 (or stay 5 if already 5)
# The reason for this is that it is easy to setup tilted pv plants on flat roofs
def add_second_class(
    data: gpd.GeoDataFrame, threshold: int = 5
) -> gpd.GeoDataFrame:
    data["KLASSE2"] = data["KLASSE"]
    data.loc[data["NEIGUNG"] <= threshold, "KLASSE2"] = (
        data.loc[data["NEIGUNG"] <= threshold, "KLASSE2"] + 1
    ).clip(upper=5)
    return data


# Preprocesses data in the sense of assign pv data and municipality top roofs and merge households
# result is stored in out/households.gpkg
# TODO: measure time to do these operations?
def preprocess_data():

    name_dataset_municipalities = os.path.join("data", "gemeindegrenzen.gpkg")
    name_dataset_roofs = os.path.join("data", "SOLKAT_DACH.gpkg")
    name_dataset_power = os.path.join(
        "data", "ch.bfe.elektrizitaetsproduktionsanlagen.gpkg"
    )

    start_time = time.time()
    data_municipalities = gpd.read_file(name_dataset_municipalities)
    data_roof = gpd.read_file(name_dataset_roofs)
    data_power = gpd.read_file(name_dataset_power)
    print(f"data read in {time.time() - start_time} seconds")

    start_time = time.time()
    data_pv = drop_not_pv(data_power)
    print(f"power cleaned in {time.time() - start_time} seconds")

    start_time = time.time()
    data_merged = join_roof_with_power(data_roof, data_pv, print_infos=False)
    print(f"pv and roofs merged in {time.time() - start_time} seconds")

    start_time = time.time()
    data_merged = add_municipality_data(data_merged, data_municipalities)
    print(f"municipalities added {time.time() - start_time} seconds")

    start_time = time.time()
    data_merged = merge_households(data_merged)

    # data_merged.to_file("out/merged.gpkg")
    # Does not work currently
    data_cleaned = cleanup_data(data_merged)
    # data_cleaned = data_merged
    data_cleaned_with_seconds_class = add_second_class(data_cleaned)

    print(
        f"PV data forwarded to whole house in {time.time() - start_time} seconds"
    )

    start_time = time.time()
    data_cleaned_with_seconds_class.to_file(
        os.path.join("out", "households.gpkg")
    )
    print(
        f"file written, preprocess data finished in {time.time() - start_time} seconds"
    )
