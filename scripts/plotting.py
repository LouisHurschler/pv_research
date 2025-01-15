import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import time
import datetime


def fill_missing_indexes(data: gpd.GeoDataFrame):
    # print(type(data))
    full_index = pd.Series(0, index=range(1, 6))
    full_index.update(data)

    return full_index


def cleanup_data(
    data: gpd.GeoDataFrame,
    threshold: int,
    start_date,
    rolling_mean_windows_size_days: int = 500,
) -> gpd.GeoDataFrame:
    data = remove_large_plants(data, threshold=threshold)
    data = data[data["BeginningOfOperation"] > start_date]
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_with_PV = data_with_PV.dropna()
    data_with_PV["BeginningOfOperation"] = np.asarray(
        data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
    )
    data_with_PV.set_index("BeginningOfOperation", inplace=True)
    data_with_PV = data_with_PV.sort_index()
    data_with_PV["cumulatedPower"] = data_with_PV["TotalPower"].cumsum()

    data_with_PV["rollingMean"] = (
        data_with_PV["TotalPower"]
        .rolling(datetime.timedelta(days=rolling_mean_windows_size_days))
        .sum()
        * 365
        / rolling_mean_windows_size_days
    )
    return data_with_PV


def generate_roof_quality_plot(
    data: gpd.GeoDataFrame, dirname: str, filename: str
):
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_without_PV = data[data["SubCategory"] != "with_PV"]

    roof_count_with = data_with_PV["KLASSE"].value_counts().sort_index()
    roof_count_without = data_without_PV["KLASSE"].value_counts().sort_index()

    if len(roof_count_with) != 5:
        roof_count_with = fill_missing_indexes(roof_count_with)

    if len(roof_count_without) != 5:
        roof_count_without = fill_missing_indexes(roof_count_without)

    # clear old data
    plt.clf()
    plt.title(filename)

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
    plt.savefig(
        os.path.join(
            dirname, filename, f"roof_quality_if_on_best_roof_filename.pdf"
        )
    )


def generate_building_times_plot(
    data: gpd.GeoDataFrame, dirname: str, filename: str
):
    plt.clf()
    plt.title(filename)
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_with_PV = data_with_PV.dropna()
    data_with_PV["BeginningOfOperation"] = np.asarray(
        data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
    )
    plt.ylabel("Gesamtleistung [kW]")

    data_with_PV.set_index("BeginningOfOperation", inplace=True)
    data_with_PV = data_with_PV.sort_index()
    data_with_PV["cumulatedPower"] = data_with_PV["TotalPower"].cumsum()
    plt.plot(data_with_PV["cumulatedPower"])
    plt.tight_layout()
    plt.savefig(
        os.path.join(dirname, filename, f"pv_building_times_{filename}.pdf")
    )


def data_to_class_data(data: gpd.GeoDataFrame) -> dict:
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    unique_households_with_pv = data_with_PV["SB_UUID"].unique()
    # TODO: create an efficient datastructure with 3 different entries: BeginningOfOperation(time format), class (1-5), TotalPower(float)
    # create 5 series to plot them efficiently

    datapoints = {
        "1": {"time": [], "value": []},
        "2": {"time": [], "value": []},
        "3": {"time": [], "value": []},
        "4": {"time": [], "value": []},
        "5": {"time": [], "value": []},
    }

    good_houses = data[data["SB_UUID"].isin(unique_households_with_pv)]

    for _, house_data in good_houses.groupby("SB_UUID"):
        # house_data = good_houses[good_houses["SB_UUID"] == unique_uuid]

        best_class = max(house_data["KLASSE"])
        try:
            total_power = house_data["TotalPower"].dropna().iloc[0]
        except:
            print(
                f"no total power in {unique_uuid} of file {filename}, skipping"
            )
            continue
        beginning_of_operation = pd.to_datetime(
            house_data["BeginningOfOperation"].dropna().iloc[0]
        )

        datapoints[str(best_class)]["time"].append(beginning_of_operation)
        datapoints[str(best_class)]["value"].append(total_power)

    for roof_quality in datapoints.keys():
        datapoints[roof_quality] = pd.Series(
            datapoints[roof_quality]["value"],
            index=datapoints[roof_quality]["time"],
        )

    for class_idx in datapoints.keys():
        loc_data = datapoints[class_idx]
        sum1 = loc_data.sum()
        unique_data = loc_data.groupby(loc_data.index).sum()
        sum2 = unique_data.sum()

        tol = 1e-6
        assert sum1 == 0.0 or abs(sum1 - sum2) / sum1 < tol, (
            sum2 - sum1
        ) / sum1

        datapoints[class_idx] = unique_data

    return datapoints


# working but slow, could be much more efficient
def generate_time_plot_with_classes_with_pv_on_best_roof(
    data: gpd.GeoDataFrame, dirname: str, filename: str, app: str = ""
):
    plt.clf()
    plt.title(filename)
    plt.ylabel("total Power installed [kW]")
    plt.xlabel("date")
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    unique_households_with_pv = data_with_PV["SB_UUID"].unique()
    # TODO: create an efficient datastructure with 3 different entries: BeginningOfOperation(time format), class (1-5), TotalPower(float)
    # create 5 series to plot them efficiently

    # skip empty data, could be done better
    is_empty = True
    for idx in datapoints.keys():
        if not datapoints[idx].empty:
            is_empty = False
    if is_empty:
        print(f"no datapoints in {filename}")
        return {}

    datapoints = data_to_class_data(data)

    # skip empty data, could be done better
    is_empty = True
    for idx in datapoints.keys():
        if not datapoints[idx].empty:
            is_empty = False
    if is_empty:
        print(f"no datapoints in {filename}")
        return

    whole_data = pd.concat(
        [
            datapoints[idx]
            for idx in datapoints.keys()
            if not datapoints[idx].empty
        ]
    ).sort_index()
    whole_data = whole_data.groupby(whole_data.index).sum()
    whole_data *= 0.0

    for class_idx in datapoints.keys():
        loc_data = datapoints[class_idx]
        new_whole_data = whole_data.copy()

        new_whole_data.loc[loc_data.index] += loc_data.values
        new_whole_data = new_whole_data.groupby(new_whole_data.index).sum()

        plt.fill_between(
            whole_data.index,
            y1=np.cumsum(new_whole_data).values,
            y2=np.cumsum(whole_data).values,
            label=class_idx,
            alpha=0.5,
        )
        whole_data = new_whole_data

    plt.legend()
    plt.savefig(
        os.path.join(dirname, filename, f"pv_evolution_{filename}{app}.pdf")
    )


def remove_large_plants(
    data: gpd.GeoDataFrame, threshold: float = 100
) -> gpd.GeoDataFrame:

    unique_uuids = data_with_large_plants = data[
        data["TotalPower"] >= threshold
    ]["SB_UUID"].unique()

    filtered_data = data[~data["SB_UUID"].isin(unique_uuids)]
    return filtered_data


def generate_diff_diff_plot(
    data: gpd.GeoDataFrame, dirname: str, filename: str, resolution: int = 10
):
    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_with_PV = data_with_PV.dropna()
    data_with_PV["BeginningOfOperation"] = np.asarray(
        data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
    )

    data_with_PV.set_index("BeginningOfOperation", inplace=True)
    data_with_PV = data_with_PV.sort_index()

    yearly_data = (
        data_with_PV["TotalPower"].groupby(data_with_PV.index.year).sum()
    )

    full_years = range(min(yearly_data.index), max(yearly_data.index) + 1)

    filled_yearly_data = yearly_data.reindex(full_years, fill_value=0)

    plt.plot(filled_yearly_data)
    plt.savefig(os.path.join(dirname, filename, f"new_pv_{filename}.pdf"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(np.cumsum(filled_yearly_data))
    plt.savefig(os.path.join(dirname, filename, f"sum_new_pv_{filename}.pdf"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(filled_yearly_data.diff())
    plt.savefig(os.path.join(dirname, filename, f"diff_{filename}.pdf"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(filled_yearly_data.diff().diff())
    plt.savefig(os.path.join(dirname, filename, f"diff_diff_{filename}.pdf"))


# note that diff n diff is a bad name for this, change this in the future or remove the whole function
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


def plot_two_locations(
    name1: str,
    name2: str,
    trimming_threshold: int = 100,
    input_dir: str = "out/municipalities",
):

    plt.clf()
    data1 = gpd.read_file(os.path.join(input_dir, name1, f"{name1}.gpkg"))
    data2 = gpd.read_file(os.path.join(input_dir, name2, f"{name2}.gpkg"))

    start_date = "2013"
    data1_with_PV = cleanup_data(data1, trimming_threshold, start_date)
    data2_with_PV = cleanup_data(data2, trimming_threshold, start_date)

    color1 = "b"
    color2 = "g"

    fig, ax1 = plt.subplots()
    plt.title(f"Vergleich installierte Gesamtleistung {name1} - {name2}")
    plt.xlabel("Datum")

    ax1.plot(data1_with_PV["cumulatedPower"], color=color1, label=name1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylabel(f"Gesamtleistung {name1} [kW]", color=color1)

    ax2 = ax1.twinx()
    ax2.plot(data2_with_PV["cumulatedPower"], color=color2, label=name2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylabel(f"Gesamtleistung {name2} [kW]", color=color2)

    fig.tight_layout()
    plt.grid(linestyle="--")

    # plt.show()
    plt.savefig(
        os.path.join("out", "plots", f"{name1}_{name2}_comparison.pdf")
    )

    plt.clf()

    fig, ax1 = plt.subplots()
    plt.title(f"Vergleich Installationsgeschwindigkeit {name1} - {name2}")
    plt.xlabel("Datum")

    ax1.plot(data1_with_PV["rollingMean"], color=color1, label=name1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylabel(
        f"Installationsgeschwindigkeit {name1} [installierte kWp pro Jahr]",
        color=color1,
        fontsize=7,
    )

    ax2 = ax1.twinx()
    ax2.plot(data2_with_PV["rollingMean"], color=color2, label=name2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylabel(
        f"Installationsgeschwindigkeit {name2} [installierte kWp pro Jahr]",
        color=color2,
        fontsize=7,
    )

    # ax2.annotate(
    #     "Einführung Förderrichtlinie Energie Aarau",
    #     xy=(
    #         pd.Timestamp("2018-03-14"),
    #         max(data2_with_PV["rollingMean"]) * 0.05,
    #     ),
    #     xytext=(
    #         pd.Timestamp("2015-07-01"),
    #         max(data2_with_PV["rollingMean"]) * 0.8,
    #     ),
    #     arrowprops=dict(facecolor="black", arrowstyle="->", lw=0.5),
    #     fontsize=7,
    # )

    fig.tight_layout()
    plt.grid(linestyle="--")

    # plt.show()
    plt.savefig(
        os.path.join("out", "plots", f"{name1}_{name2}_installation.pdf")
    )

    fig, axes1 = plt.subplots(5, 1, sharex=True)
    for i, ax in enumerate(axes1):
        ax.set_ylabel(f"Klasse {i+1}")
    class_dict_1 = data_to_class_data(data1)
    colors = ["b", "r", "g", "y", "o"]

    for i, ax in enumerate(axes1):
        loc_data = class_dict_1[str(i + 1)]
        if not loc_data.empty:
            loc_data = class_dict_1[str(i + 1)]
            rolling_length = min(len(loc_data), 500)
            ax.plot(
                loc_data.rolling(datetime.timedelta(days=rolling_length)).sum()
                * 365
                / rolling_length,
                color=color1,
            )
    axes2 = [ax.twinx() for ax in axes1]
    class_dict_2 = data_to_class_data(data2)

    for i, ax in enumerate(axes2):
        loc_data = class_dict_2[str(i + 1)]
        if not loc_data.empty:
            loc_data = class_dict_2[str(i + 1)]
            rolling_length = min(len(loc_data), 500)
            ax.plot(
                loc_data.rolling(datetime.timedelta(days=rolling_length)).sum()
                * 365
                / rolling_length,
                color=color2,
            )

    plt.savefig(
        os.path.join("out", "plots", f"{name1}_{name2}_overview_classes.pdf")
    )

    # TODO: do comparison for all 5 subclasses, plot them in subplots and save result
