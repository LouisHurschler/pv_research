import geopandas as gpd
from preprocess_data import preprocess_data
from plotting import *
from data_creation import generate_data
from redistribute_pv import redistribute_pv_power_to_best_available_roof
import datetime
import os
import time
import json
from tqdm import tqdm


def generate_distribution_plot_switzerland():
    data = gpd.read_file(os.path.join("out", "households.gpkg"))

    # data_trimmed = remove_large_plants(data, threshold=100)
    # generate_roof_quality_plot(data_trimmed, "out", "Switzerland")

    generate_roof_quality_plot(data, "out", "Switzerland")


# TODO: which ones are really relevant?
def generate_plot_of_single_file(dirname: str, filename: str):
    data = gpd.read_file(os.path.join(dirname, filename, filename + ".gpkg"))
    generate_roof_quality_plot(data, dirname, filename)
    # generate_building_times_plot(data, dirname, filename)
    # generate_time_plot_with_classes_with_pv_on_best_roof(
    #     data, dirname, filename
    # )
    # data_trimmed = remove_large_plants(data, threshold=100)

    # generate_time_plot_with_classes_with_pv_on_best_roof(
    #     data_trimmed, dirname, filename, app="_trimmed"
    # )
    # generate_diff_diff_plot(data_trimmed, dirname, filename)


def run_municipal(
    have_to_generate_data: bool = True, have_to_plot_data: bool = True
):

    if have_to_generate_data:
        data = gpd.read_file(os.path.join("out", "households_5.0_KLASSE.gpkg"))
        muni_names = data["GDE_NAME"].unique()
        # name_len = len(muni_names)
        for muni in tqdm(muni_names):
            # if i % int(name_len / 20) == 0:
            #     print(f"{int(i * 100 / name_len)}%")
            generate_data(data, muni, folder="municipalities")
        print("municipal data generated")

    if have_to_plot_data:
        path = os.path.join("out", "municipalities")
        muni_names = os.listdir(path)
        # name_len = len(muni_names)

        for muni in tqdm(muni_names):
            # if i % int(name_len / 20) == 0:
            #     print(f"{int(i * 100 / name_len)}%")
            generate_plot_of_single_file(path, muni)
        print("municipal data plotted")


def run_canton(
    have_to_generate_data: bool = True, have_to_plot_data: bool = True
):
    if have_to_generate_data:
        data = gpd.read_file(os.path.join("out", "households_5.0_KLASSE.gpkg"))
        canton_names = data["KT_KZ"].unique()
        for canton in tqdm(canton_names):
            print(canton)
            generate_data(data, canton, folder="cantons")
        print("cantonal data generated")

    if have_to_plot_data:
        path = os.path.join("out", "cantons")
        canton_names = os.listdir(path)
        for canton in tqdm(canton_names):
            print(canton)
            generate_plot_of_single_file(path, canton)
        print("cantonal data plotted")


if __name__ == "__main__":

    gpd.options.io_engine = "pyogrio"
    # gpd.options.use_pygeos = True
    os.environ["PYOGRIO_USE_ARROW"] = "1"
    # preprocess_data()
    # generate_distribution_plot_switzerland()
    # generate_plot_of_single_file("out/municipalities", "Aarau")
    # generate_plot_of_single_file("out/municipalities", "Thun")
    # generate_plot_of_single_file("out/cantons", "ZH")

    # data = gpd.read_file(
    #     # os.path.join("out/households_5.0_KLASSE2.gpkg"),
    #     os.path.join("out/households_5.0_KLASSE2.gpkg"),
    #     engine="pyogrio",
    #     use_arrow=True,
    # )
    # data = redistribute_pv_power_to_best_available_roof(
    #     data, used_area_m2_per_kwp=5.0, name_class="KLASSE2"
    # )

    # TODO: do something about a roof quality plot changed over time?
    # generate_roof_quality_plot(
    #     data, "out", "Switzerland", class_name="KLASSE2"
    # )
    # generate_roof_quality_plot(data, "out", "Thun", class_name="KLASSE2")

    # data = gpd.read_file("out/municipalities/Sempach/Sempach.gpkg")
    # data = project_pv_to_correct_roof(data)
    # print(data)

    # generate_plot_of_single_file("out/municipalities", "Sempach")
    # generate_plot_of_single_file("out/cantons", "ZH")
    run_municipal(have_to_generate_data=True, have_to_plot_data=True)
    run_canton(have_to_generate_data=True, have_to_plot_data=True)
    # plot_two_locations("Aarau", "Thun")
    # plot_two_locations("ZH", "LU", input_dir="out/cantons")
