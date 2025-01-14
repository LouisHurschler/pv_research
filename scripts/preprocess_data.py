import geopandas as gpd
import os


def join_roof_with_power(
    data_roof: gpd.GeoDataFrame, data_power: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    merged = gpd.sjoin(
        data_roof,
        data_power,
        how="left",
    )
    return merged


def drop_not_pv(data_power: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data_pv = data_power[data_power["SubCategory"] == "subcat_2"]
    return data_pv


def add_municipality_data(
    data_merged: gpd.GeoDataFrame, data_muni: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # add another rsuffix because the first join already adds right
    merged = gpd.sjoin(
        data_merged, data_muni, how="left", rsuffix="right_muni"
    )
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
    print(list_of_keys)
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
    data["SubCategory"][data["SubCategory"] == "subcat_2"] = "with_PV"
    data["SubCategory"][data["SubCategory"].isna()] = "without_PV"
    # TODO: rename key?
    return data


# Preprocesses data in the sense of assign pv data and municipality top roofs and merge households
# result is stored in out/households.gpkg
def preprocess_data():
    dataset_municipalities = os.path.join("data", "gemeindegrenzen.gpkg")
    dataset_roofs = os.path.join("data", "SOLKAT_DACH.gpkg")
    dataset_power = os.path.join(
        "data", "ch.bfe.elektrizitaetsproduktionsanlagen.gpkg"
    )

    data_municipalities = gpd.read_file(dataset_municipalities)
    data_roof = gpd.read_file(dataset_roofs)
    data_power = gpd.read_file(dataset_power)
    print("data read")

    data_pv = drop_not_pv(data_power)
    print("power cleaned")

    # takes about
    data_merged = join_roof_with_power(data_roof, data_pv)
    print("pv and roofs merged")

    data_merged = add_municipality_data(data_merged, data_municipalities)
    print("municipalities added")

    data_merged = merge_households(data_merged)

    # data_merged.to_file("out/merged.gpkg")
    # Does not work currently
    data_cleaned = cleanup_data(data_merged)
    # data_cleaned = data_merged

    print("PV data forwarded to whole house")

    data_cleaned.to_file(os.path.join("out", "households.gpkg"))
    print("file written, preprocess data finished")
