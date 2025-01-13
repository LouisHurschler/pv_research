import geopandas as gpd
from preprocess_data import preprocess_data
from plotting import *
from data_creation import generate_data
import os
import time


def generate_plot_of_single_file(dirname: str, filename: str):
    data = gpd.read_file(os.path.join(dirname, filename, filename + ".gpkg"))
    generate_roof_quality_plot(data, dirname, filename)
    generate_building_times_plot(data, dirname, filename)
    generate_time_plot_with_classes_with_pv_on_best_roof(
        data, dirname, filename
    )
    data_trimmed = remove_large_plants(data, threshold=100)

    generate_time_plot_with_classes_with_pv_on_best_roof(
        data_trimmed, dirname, filename, app="_trimmed"
    )
    generate_diff_diff_plot(data_trimmed, dirname, filename)
    # generate_diff_diff_plot(data_trimmed, dirname, filename, resolution=10)


def run_municipal(
    have_to_generate_data: bool = True, have_to_plot_data: bool = True
):

    if have_to_generate_data:
        data = gpd.read_file(os.path.join("out", "households.gpkg"))
        muni_names = data["GDE_NAME"].unique()
        name_len = len(muni_names)
        for i, muni in enumerate(muni_names):
            if i % int(name_len / 20) == 0:
                print(f"{int(i * 100 / name_len)}%")
            generate_data(data, muni, folder="municipalities")
        print("municipal data generated")

    if have_to_plot_data:
        path = os.path.join("out", "municipalities")
        muni_names = os.listdir(path)
        name_len = len(muni_names)

        for i, muni in enumerate(muni_names):
            if i % int(name_len / 20) == 0:
                print(f"{int(i * 100 / name_len)}%")

            generate_plot_of_single_file(path, muni)
        print("municipal data plotted")


def run_canton(
    have_to_generate_data: bool = True, have_to_plot_data: bool = True
):
    if have_to_generate_data:
        data = gpd.read_file(os.path.join("out", "households.gpkg"))
        canton_names = data["KT_KZ"].unique()
        for canton in canton_names:
            print(canton)
            generate_data(data, canton, folder="cantons")
        print("cantonal data generated")

    if have_to_plot_data:
        path = os.path.join("out", "cantons")
        canton_names = os.listdir(path)
        for canton in canton_names:
            print(canton)
            generate_plot_of_single_file(path, canton)
        print("cantonal data plotted")


if __name__ == "__main__":
    # preprocess_data()

    # generate_plot_of_single_file("out/municipalities/", "Sempach")
    # generate_plot_of_single_file("out/municipalities", "Zürich")
    # run_municipal(have_to_generate_data=False)
    run_canton(have_to_generate_data=False)
