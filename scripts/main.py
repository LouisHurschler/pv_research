import geopandas as gpd
from preprocess_data import preprocess_data
from plotting import *
from data_creation import generate_data
import datetime
import os
import time
import json


# TODO: which ones are really relevant?
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
    # generate_diff_diff_plot(data_trimmed, dirname, filename)


# percentage of built pv plants are calculated for all files (municipalities or cantons)
# in files and dates in dates
def calculate_diff_n_diff(
    dirname: str,
    files: list,
    dates: list,
    output: str = "out/diff_n_diff.json",
):
    result = {"dates": dates}
    # assume 1000 kWh / year per kwp
    power_to_yearly_energy_factor = 1000
    for file in files:
        data = gpd.read_file(os.path.join(dirname, file, f"{file}.gpkg"))
        # cut off all big pv plants
        data = remove_large_plants(data, threshold=100)

        total_theoretical_energy = sum(data["STROMERTRAG"])
        percentage_installed = []
        for date in dates:
            percentage_installed.append(
                sum(data[data["BeginningOfOperation"] <= date]["TotalPower"])
                * power_to_yearly_energy_factor
                / total_theoretical_energy
            )
        result[file] = percentage_installed

    with open(output, "w") as outfile:
        json.dump(result, outfile, indent=10)


def plot_diff_n_diff(
    name1: str, name2: str, input_file: str = "out/diff_n_diff.json"
):
    plt.clf()
    data = pd.read_json(input_file)
    plt.xlabel("date")
    plt.ylabel("percentage of pv coverage")
    plt.title(f"pv coverage of {name1} and {name2}")
    plt.plot(data["dates"], data[name1], label=name1)
    plt.plot(data["dates"], data[name2], label=name2)
    plt.plot(
        data["dates"],
        data[name2] - data[name1],
        label=f"differene ({name2} - {name1})",
    )
    plt.legend()
    plt.grid()
    plt.savefig(f"out/diff_n_diff/{name1}_{name2}.pdf")


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

    # generate_plot_of_single_file("out/municipalities", "Sempach")
    # generate_plot_of_single_file("out/municipalities", "Zürich")
    # run_municipal(have_to_generate_data=True, have_to_plot_data=False)
    # run_canton(have_to_generate_data=True, have_to_plot_data=False)
    # calculate_diff_n_diff(
    #     "out/municipalities",
    #     os.listdir("out/municipalities"),
    #     [str(x) for x in range(2010, 2026)],
    # )
    # calculate_diff_n_diff(
    #     "out/cantons",
    #     os.listdir("out/cantons"),
    #     [str(x) for x in range(2010, 2026)],
    #     output="out/diff_n_diff_cantons.json",
    # )
