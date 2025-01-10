import geopandas as gpd
import os


def generate_data(
    data: gpd.GeoDataFrame, name: str, folder: str = "municipalities"
):
    if name == None:
        return
    if folder == "municipalities":
        loc_data = data[data["GDE_NAME"] == name]
    else:
        loc_data = data[data["KT_KZ"] == name]

    # fixing bugs with Biel/Bienne for example
    name = name.replace("/", "_")
    dirname = os.path.join("out", folder, name)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    loc_data.to_file(os.path.join(dirname, name + ".gpkg"))
