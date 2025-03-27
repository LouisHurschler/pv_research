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


def generate_plot_of_single_file(
    dirname: str, filename: str, specifier: str = ""
):
    data = gpd.read_file(
        os.path.join(dirname, filename, filename + specifier + ".gpkg")
    )
    if specifier.endswith("KLASSE"):
        generate_roof_quality_plot(
            data, dirname, filename, class_name="KLASSE"
        )
    elif specifier.endswith("KLASSE2"):
        generate_roof_quality_plot(
            data, dirname, filename, class_name="KLASSE2"
        )
    else:
        generate_roof_quality_plot(data, dirname, filename)


def run_municipal(
    have_to_generate_data: bool = True,
    have_to_plot_data: bool = True,
    specifier: str = "_5.0_KLASSE",
):

    if have_to_generate_data:
        data = gpd.read_file(
            os.path.join("out", "households" + specifier + ".gpkg")
        )

        muni_names = data["GDE_NAME"].unique()

        print(f"generating municipal data for specifier {specifier}")

        for muni in tqdm(muni_names):
            generate_data(
                data, muni, folder="municipalities", specifier=specifier
            )

    if have_to_plot_data:
        path = os.path.join("out", "municipalities")
        muni_names = os.listdir(path)

        print(f"generating municipal plots for specifier {specifier}")

        for muni in tqdm(muni_names):
            generate_plot_of_single_file(path, muni, specifier=specifier)


def run_canton(
    have_to_generate_data: bool = True,
    have_to_plot_data: bool = True,
    specifier: str = "_5.0_KLASSE",
):
    if have_to_generate_data:
        data = gpd.read_file(os.path.join("out", "households_5.0_KLASSE.gpkg"))
        canton_names = data["KT_KZ"].unique()
        for canton in tqdm(canton_names):
            print(canton)
            generate_data(data, canton, folder="cantons", specifier=specifier)
        print("cantonal data generated")

    if have_to_plot_data:
        path = os.path.join("out", "cantons")
        canton_names = os.listdir(path)
        for canton in tqdm(canton_names):
            print(canton)
            generate_plot_of_single_file(path, canton, specifier=specifier)
        print("cantonal data plotted")


# This function should be done once for every class_name and used_area_m2_per_kwp.
# It redistributes the pv on the specific roofs and saves the data for switzerland, as well as for every municipality and canton.
# NOTE: this takes a lot of time to run (1h+)
def data_to_roofs(
    name_class: str = "KLASSE",
    used_area: float = 5.0,
    do_redistribution: bool = True,
):
    specifier = "_" + str(used_area) + "_" + name_class
    if do_redistribution:
        redistribute_pv_power_to_best_available_roof(
            data=gpd.read_file(
                os.path.join("out", "households.gpkg"),
                engine="pyogrio",
                use_arrow=True,
            ),
            used_area_m2_per_kwp=used_area,
            name_class=name_class,
        )

    run_municipal(
        have_to_generate_data=False,
        have_to_plot_data=True,
        specifier=specifier,
    )
    run_canton(
        have_to_generate_data=True,
        have_to_plot_data=True,
        specifier=specifier,
    )


def create_specific_plots():
    # generate_plot_of_single_file(
    #     "out/municipalities", "Thun", specifier="_5.0_KLASSE2"
    # )
    # generate_plot_of_single_file("out/cantons", "ZH")

    # run_municipal(False, True, specifier="_5.0_KLASSE2")
    # run_canton(False, True, specifier="_5.0_KLASSE2")

    # data = gpd.read_file(
    #     # os.path.join("out/households.gpkg"),
    #     os.path.join("out/households_5.0_KLASSE2.gpkg"),
    #     engine="pyogrio",
    #     use_arrow=True,
    # )

    plot_two_locations("Aarau", "Thun")


if __name__ == "__main__":

    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    # should be done only once at the beginning
    # preprocess_data()

    # data_to_roofs("KLASSE", 5.0, do_redistribution=False)
    # data_to_roofs("KLASSE2", 5.0, do_redistribution=False)

    create_specific_plots()
