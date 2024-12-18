import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd


def drop_not_pv(data_power: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data_pv = data_power[data_power["SubCategory"] == "subcat_2"]
    return data_pv


def join_roof_power(
    data_roof: gpd.GeoDataFrame, data_power: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    merged = gpd.sjoin(
        data_roof,
        data_power,
        how="left",
    )
    return merged


def cleanup_data(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    data.drop(
        columns=[
            "DF_UID",
            "DF_NUMMER",
            "DATUM_ERSTELLUNG",
            "DATUM_AENDERUNG",
            # "SB_UUID", used for classification of house
            "SB_OBJEKTART",
            "SB_DATUM_ERSTELLUNG",
            "DUSCHGAENGE",
            "DG_HEIZUNG",
            "DG_WAERMEBEDARF",
            "BEDARF_WARMWASSER",
            "xtf_id",
            "BeginningOfOperation",
            "MainCategory",
            # stuff from gemeindenamen
            "index_right_muni",
            "OBJECTID",
            "GMDHISTID",
            "GMDNR",
            "BZHISTID",
            "BZNR",
            "KTNR",
            "GRNR",
            "AREA_HA",
            "E_MIN",
            "E_MAX",
            "N_MIN",
            "N_MAX",
            "E_CNTR",
            "N_CNTR",
            "Z_MIN",
            "Z_MAX",
            "Z_CNTR",
            "Z_AVG",
            "Z_MED",
            "Shape_Length",
            "Shape_Area",
            "Kanton",
            "Address",
            "PostCode",
            "Municipality",
            "Canton",
            "PlantCategory",
        ],
        inplace=True,
    )
    print("cols_removed")
    data["SubCategory"][data["SubCategory"] == "subcat_2"] = "with_PV"
    data["SubCategory"][data["SubCategory"].isna()] = "without_PV"
    return data


def add_municipality(
    data_merged: gpd.GeoDataFrame, data_muni: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # add another rsuffix because the first join already adds right
    merged = gpd.sjoin(
        data_merged, data_muni, how="left", rsuffix="right_muni"
    )
    # TODO
    return merged


def merge_households(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    data["SubCategory"] = data.groupby("SB_UUID")["SubCategory"].transform(
        "first"
    )
    return data


# Preprocesses data in the sense of assign pv data and municipality top roofs and merge households
# result is stored in out/households.gpkg
def preprocess_data():
    dataset_municipalities = "data/gemeindegrenzen.gpkg"
    dataset_roofs = "data/SOLKAT_DACH.gpkg"
    dataset_power = "data/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg"

    data_municipalities = gpd.read_file(dataset_municipalities)
    data_roof = gpd.read_file(dataset_roofs)
    data_power = gpd.read_file(dataset_power)
    print("data read")

    data_pv = drop_not_pv(data_power)
    print("power cleaned")

    # takes about
    data_merged = join_roof_power(data_roof, data_pv)
    print("pv and roofs merged")

    data_merged = add_municipality(data_merged, data_municipalities)
    print("municipalities added")

    data_merged = merge_households(data_merged)

    data_cleaned = cleanup_data(data_merged)

    print("PV data forwarded to whole house")

    data_cleaned.to_file("out/households.gpkg")
    print("file written")


def fill_missing_indexes(data):
    print(type(data))
    full_index = pd.Series(0, index=range(1, 6))

    full_index.update(data)

    return full_index


def generate_single_plot(name: str, data: gpd.GeoDataFrame):
    # fix directory bug for example for Biel/Bienne
    if name == None:
        return
    name = name.replace("/", "_")
    dirname = "figures/" + name
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_without_PV = data[data["SubCategory"] != "with_PV"]

    roof_count_with = data_with_PV["KLASSE"].value_counts().sort_index()
    roof_count_without = data_without_PV["KLASSE"].value_counts().sort_index()

    # for now I am gonna skip all municipalities where not all classes are available for simplicity
    # TODO: fill mission values with 0?

    if len(roof_count_with) != 5:
        roof_count_with = fill_missing_indexes(roof_count_with)

    if len(roof_count_without) != 5:
        roof_count_without = fill_missing_indexes(roof_count_without)

    # clear old data
    plt.clf()
    plt.title(name)

    # add barplots
    plt.bar(
        roof_count_with.index,
        roof_count_with.values,
        label="Number of roofs with PV",
        color="green",
    )
    plt.bar(
        roof_count_without.index,
        roof_count_without.values,
        label="Number of roofs without PV",
        bottom=roof_count_with.values,
        color="red",
    )

    percentage_pv = roof_count_with / (roof_count_with + roof_count_without)
    max_percentage = max(percentage_pv)
    max_value = max(roof_count_with + roof_count_without)
    plt.ylabel("Number of Rooftops")
    plt.legend(loc="upper left")

    plt.twinx()
    plt.plot(
        roof_count_with.index,
        percentage_pv,
        label="Percentage with PV",
    )

    plt.ylabel("percentage of PV plants on roofs")

    plt.xlabel("Klasse")
    plt.legend(loc="upper right")
    plt.savefig(dirname + "/roof_quality.svg")


def generate_municipal_plots():
    data = gpd.read_file("out/households.gpkg")
    print("file read")

    municipalities = data["GMDNAME"].unique().tolist()
    for i, muni in enumerate(municipalities):
        if i % int(len(municipalities) / 20) == 0:
            print(f"{int(i * 100 / len(municipalities))}%")

        loc_data = data[data["GMDNAME"] == muni]
        generate_single_plot(muni, loc_data)
    generate_single_plot("whole_data", data)


def test_plots():
    data = gpd.read_file("out/regensdorf.gpkg")
    generate_single_plot("Regensdorf", data)


if __name__ == "__main__":
    # preprocess_data()
    # test_plots()
    generate_municipal_plots()
