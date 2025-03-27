import geopandas as gpd
import os


def generate_data(
    data: gpd.GeoDataFrame,
    name: str,
    folder: str = "municipalities",
    specifier: str = "_5.0_KLASSE",
):
    if name == None:
        return
    if folder == "municipalities":
        loc_data = data[data["GDE_NAME"] == name]
    else:
        loc_data = data[data["KT_KZ"] == name]
    # skip if empty: removed because we can write empty datasets
    # if loc_data.empty:
    #     print("data is empty")
    #     return

    # fixing bugs with Biel/Bienne for example
    name = name.replace("/", "_")
    dirname = os.path.join("out", folder, name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # There is a problem when writing data to a file with arrow where one column
    # consists only of none values. Therefore do not use arrow to write small datasets to files.
    loc_data.to_file(
        os.path.join(dirname, name + specifier + ".gpkg"),
        use_arrow=False,
    )
