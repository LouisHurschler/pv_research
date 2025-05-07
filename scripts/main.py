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


def generate_distribution_plot_switzerland() -> None:
    data = gpd.read_file(os.path.join("out", "households.gpkg"))

    # data_trimmed = remove_large_plants(data, threshold=100)
    # generate_roof_quality_plot(data_trimmed, "out", "Switzerland")

    generate_roof_quality_plot(data, "out", "Switzerland")


def generate_plot_of_single_file(
    dirname: str, filename: str, specifier: str = "", built_until=None
) -> None:
    data = gpd.read_file(
        os.path.join(dirname, filename, filename + specifier + ".gpkg")
    )
    # used to analyze only rooftops where at least one part is covered with a PV system
    # data = data[
    #     data.groupby("SB_UUID")["SubCategory"].transform(
    #         lambda x: (x == "with_PV").any()
    #     )
    # ]
    if specifier.endswith("KLASSE"):
        generate_roof_quality_plot(
            data,
            dirname,
            filename,
            class_name="KLASSE",
            built_until=built_until,
        )
    elif specifier.endswith("KLASSE2"):
        generate_roof_quality_plot(
            data,
            dirname,
            filename,
            class_name="KLASSE2",
            built_until=built_until,
        )
    else:
        generate_roof_quality_plot(
            data, dirname, filename, built_until=built_until
        )


def generate_plots_aarau(
    have_to_generate_data: bool = True,
    have_to_plot_data: bool = True,
    specifier="_5.0_KLASSE2",
    built_until_list=[
        pd.to_datetime(str(year)) for year in range(2015, 2027, 1)
    ]
    + [None],
) -> None:
    name_aarau = "Aarau"
    if have_to_generate_data:
        data = gpd.read_file(
            os.path.join("out", "households" + specifier + ".gpkg")
        )
        generate_data(
            data, name_aarau, folder="municipalities", specifier=specifier
        )
        print(f"data for aarau generated with specifier: {specifier}")

    if have_to_plot_data:
        path = os.path.join("out", "municipalities")
        if built_until_list is not None:
            for built_until in built_until_list:
                generate_plot_of_single_file(
                    path,
                    name_aarau,
                    specifier=specifier,
                    built_until=built_until,
                )
                print(built_until)
        print(f"plot for aarau generated with specifier: {specifier}")


def run_municipal(
    have_to_generate_data: bool = True,
    have_to_plot_data: bool = True,
    specifier: str = "_5.0_KLASSE",
) -> None:

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
    generate_plot_of_single_file("out/cantons", "ZH", specifier="_5.0_KLASSE2")

    # run_municipal(False, True, specifier="_5.0_KLASSE2")
    # run_canton(False, True, specifier="_5.0_KLASSE2")

    # data = gpd.read_file(
    #     # os.path.join("out/households.gpkg"),
    #     os.path.join("out/households_5.0_KLASSE2.gpkg"),
    #     engine="pyogrio",
    #     use_arrow=True,
    # )

    # plot_two_locations("Aarau", "Thun")

    # plot roof distribution of switzerland
    # data = gpd.read_file("out/households_5.0_KLASSE2.gpkg")
    # data = data[
    #     data.groupby("SB_UUID")["SubCategory"].transform(
    #         lambda x: (x == "with_PV").any()
    #     )
    # ]

    # generate_roof_quality_plot(
    #     data=data,
    #     dirname="out/switzerland",
    #     filename="switzerland",
    #     class_name="KLASSE2",
    # )


if __name__ == "__main__":

    gpd.options.io_engine = "pyogrio"
    os.environ["PYOGRIO_USE_ARROW"] = "1"

    plot_two_locations("Cham", "Switzerland")
    plot_two_locations("Landquart", "Switzerland")
    # plot_two_locations("Aarau", "Thun")

    # do this once
    data_already_preprocessed = True
    if not data_already_preprocessed:
        preprocess_data()

    # data_to_roofs("KLASSE", 5.0, do_redistribution=False)
    # data_to_roofs("KLASSE2", 5.0, do_redistribution=False)

    # generate_plots_aarau(have_to_generate_data=False)
    # create_specific_plots()
