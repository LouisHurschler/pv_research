import matplotlib
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import time
import datetime
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator
from get_tarif_data import *
from tqdm import tqdm


def gaussian_kernel(size, sigma):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normalize


def squared_kernel(size):
    kernel = np.linspace(0, 1, size) * (1.0 - np.linspace(0, 1, size))
    scaling_factor = 6.0 * (size - 1.0) / (size * (size - 2))
    return kernel * scaling_factor


# apply a linear weighted sum of window, preserves integral
def own_flattening_function(values):
    # rolling mean:
    # return values.sum() / len(values)
    N = len(values)
    if N <= 2:
        return 0.0
    # kernel = squared_kernel(N)
    kernel = gaussian_kernel(N, 200.0)
    return (values * kernel).sum()


def custom_running_mean(data: pd.Series, rolling_mean_windows_size_days: int):
    # add missing days
    data = data.resample("D").sum()
    first_date = data.index[0]
    last_date = data.index[-1]
    # append zeros for boundary effect
    data = pd.concat(
        [
            pd.Series(
                # [0.0] * rolling_mean_windows_size_days,
                data.iloc[: rolling_mean_windows_size_days // 2].values[::-1],
                index=pd.date_range(
                    start=first_date
                    - datetime.timedelta(
                        days=rolling_mean_windows_size_days // 2
                    ),
                    periods=rolling_mean_windows_size_days // 2,
                ),
            ),
            data,
            pd.Series(
                # [0.0] * rolling_mean_windows_size_days,
                data.iloc[-rolling_mean_windows_size_days // 2 :].values[::-1],
                index=pd.date_range(
                    start=last_date + datetime.timedelta(days=1),
                    periods=rolling_mean_windows_size_days // 2,
                ),
            ),
        ]
    )
    rolling_window = (
        data.rolling(
            window=datetime.timedelta(days=rolling_mean_windows_size_days),
            center=True,  # if only accounts for previous values
        ).apply(own_flattening_function)
        * 356
    )  # multiply by 356 to get yearly rate
    # remove padding
    rolling_window = rolling_window.iloc[
        rolling_mean_windows_size_days
        // 2 : -rolling_mean_windows_size_days
        // 2
    ]
    return rolling_window


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
    cumulated_sum = data_with_PV["TotalPower"].cumsum()

    smoothed1 = (
        data_with_PV["TotalPower"].cumsum().resample("W").last().ffill().diff()
    )
    numeric_timestamps = cumulated_sum.index.astype(np.int64)
    kde = gaussian_kde(numeric_timestamps, weights=cumulated_sum)
    smoothed_values = kde(numeric_timestamps)

    smoothed_cumulated = pd.Series(smoothed_values, index=cumulated_sum.index)
    smoothed = smoothed_cumulated
    smoothed = cumulated_sum

    # smoothed = smoothed_cumulated.resample("M").last().ffill().diff()

    data_with_PV["rollingMean"] = (
        data_with_PV["TotalPower"]
        .rolling(datetime.timedelta(days=rolling_mean_windows_size_days))
        .sum()
        * 365
        / rolling_mean_windows_size_days
    )
    smoothed = custom_running_mean(
        data_with_PV["TotalPower"], rolling_mean_windows_size_days
    )
    return data_with_PV, smoothed


def generate_roof_quality_plot(
    data: gpd.GeoDataFrame,
    dirname: str,
    filename: str,
    class_name: str = "KLASSE",
):
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    data_without_PV = data[data["SubCategory"] != "with_PV"]

    area_with = (
        data_with_PV.groupby(class_name)["FLAECHE"].sum().rename("area_with")
    ) / 1000000.0  # in km2
    # roof_count_with = data_with_PV[class_name].value_counts().sort_index()
    # print("roof_count_with: ", roof_count_with)

    area_without = (
        data_without_PV.groupby(class_name)["FLAECHE"]
        .sum()
        .rename("area_without")
    ) / 1000000.0  # in km2

    # roof_count_without = (
    #     data_without_PV[class_name].value_counts().sort_index()
    # )
    # print("roof_count_without: ", roof_count_without)

    # if len(roof_count_with) != len(roof_count_without) or len(roof_count_with) != 5:
    if len(area_with) != 5:
        area_with = fill_missing_indexes(area_with)

    if len(area_without) != 5:
        area_without = fill_missing_indexes(area_without)
    # if filename == "Aarau":
    #     roof_count_with = roof_count_with.drop(5)
    #     roof_count_without = roof_count_without.drop(5)

    # clear old data
    plt.clf()

    fontsize = 10
    # plt.rc("text", usetex=True)

    plt.rc("text", usetex=True)

    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # plt.title(
    #     f"Distribution of PV Installations: Municipality {filename}",
    #     fontsize=fontsize,
    # )
    # color_green = "#2CA02C"
    # color_red = "#D62728"
    color_green = "#27AE60"
    color_red = "#E74C3C"
    color_blue = "#1F77B4"

    # add barplots
    ax1.bar(
        area_with.index,
        area_with.values,
        label="Area with PV",
        color=color_green,
    )
    ax1.bar(
        area_without.index,
        area_without.values,
        label="Area without PV",
        bottom=area_with.values,
        color=color_red,
    )
    ax1.set_xlabel("Rooftop PV-Suitability", fontsize=fontsize)

    percentage_pv = 100 * area_with / (area_with + area_without)
    max_percentage = max(percentage_pv)
    max_value = max(area_with + area_without)
    ax1.set_ylabel(r"Area [$\mathrm{km}^2$]")
    # plt.legend(loc="upper left")
    handles1, labels1 = ax1.get_legend_handles_labels()
    # was used to create cleaner plots of aarau, can be ignored now
    # if filename == "Aarau":
    #     roof_count_with = roof_count_with.drop(5)
    #     percentage_pv = percentage_pv.drop(5)

    ax2 = ax1.twinx()
    ax2.plot(
        area_with.index,
        percentage_pv,
        label="Percentage with PV",
        color=color_blue,
    )
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        fontsize=fontsize,
    )

    ax2.set_ylabel(
        r"Percentage of Roof Area covered with PV Systems [\%]",
        fontsize=fontsize,
    )

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    plt.xlabel("Klasse")
    # plt.legend(loc="upper right")
    if not os.path.exists(os.path.join(dirname, filename)):
        os.makedirs(os.path.join(dirname, filename))

    plt.savefig(
        os.path.join(
            dirname, filename, f"roof_quality_{filename}_{class_name}.jpg"
        ),
        dpi=1200,
    )
    plt.close()


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
        os.path.join(dirname, filename, f"pv_building_times_{filename}.svg")
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
        os.path.join(dirname, filename, f"pv_evolution_{filename}{app}.svg")
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
    plt.savefig(os.path.join(dirname, filename, f"new_pv_{filename}.svg"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(np.cumsum(filled_yearly_data))
    plt.savefig(os.path.join(dirname, filename, f"sum_new_pv_{filename}.svg"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(filled_yearly_data.diff())
    plt.savefig(os.path.join(dirname, filename, f"diff_{filename}.svg"))

    plt.clf()
    plt.title(filename)
    plt.ylabel("Gesamtleistung [kW]")

    plt.plot(filled_yearly_data.diff().diff())
    plt.savefig(os.path.join(dirname, filename, f"diff_diff_{filename}.svg"))


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
    plt.savefig(f"out/diff_n_diff/{name1}_{name2}.svg")


# This funciton is used to plot the distribution of the installed plants over time for two municipalities.
# It plots the Power installed and smoothenes it using som e rolling window (currently using gaussian kde)
# Note that the beginning of operations entry has to be included in the data for each entry with subCategory with_PV
# The text and arrows are currently hardcoded to aarau and thun, change this if necessary.
def plot_two_locations(
    name1: str,
    name2: str,
    trimming_threshold: int = 100,
    input_dir: str = "out/municipalities",
):
    ending = "jpg"

    plt.clf()
    data1 = gpd.read_file(
        os.path.join(input_dir, name1, f"{name1}_5.0_KLASSE2.gpkg")
    )
    data2 = gpd.read_file(
        os.path.join(input_dir, name2, f"{name2}_5.0_KLASSE2.gpkg")
    )

    start_date = "2013"
    window_size = 2000
    data1_with_PV, smoothed_1 = cleanup_data(
        data1,
        trimming_threshold,
        start_date,
        rolling_mean_windows_size_days=window_size,
    )
    data2_with_PV, smoothed_2 = cleanup_data(
        data2,
        trimming_threshold,
        start_date,
        rolling_mean_windows_size_days=window_size,
    )
    smoothed_1 = smoothed_1[
        smoothed_1.index < pd.to_datetime("2023-06", format="%Y-%m")
    ]
    smoothed_2 = smoothed_2[
        smoothed_2.index < pd.to_datetime("2023-06", format="%Y-%m")
    ]
    # color1 = "#4C72B0"  # blue
    # color2 = "#DD8452"  # orange
    color1 = "#001C7F"  # dark blue
    color2 = "#017517"  # dark green
    # color1 = "#4878CF"  # muted blue
    # color2 = "#6ACC64"  # muted green

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
        os.path.join("out", "plots", f"{name1}_{name2}_comparison." + ending)
    )

    ######################## installation plot begin #######################################
    plt.clf()

    fontsize = 10
    # plt.rc("text", usetex=True)

    plt.rc("text", usetex=True)

    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
    matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
    matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
    fig, ax1 = plt.subplots(figsize=(6, 4))
    # ax1.set_title(
    #     f"PV Installation Rate Comparison: Municipalities {name1} and {name2}",
    #     fontsize=fontsize,
    # )
    ax1.set_xlabel("Year", fontsize=fontsize)

    test_data = pd.Series(
        data=np.linspace(0, 2000, num=len(data2_with_PV["rollingMean"])),
        index=data2_with_PV.index,
    )
    ax1.plot(
        smoothed_1,
        # data1_with_PV["rollingMean"],
        # test_data,
        color=color1,
        label=f"PV Installation Rate in {name1} [kWp/year]",
        linestyle="-",
        linewidth=0.8,
    )
    # ax1.plot(smoothed_1, color=color1, label=name1)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=fontsize)
    ax1.tick_params(axis="x", labelsize=fontsize)

    ax1.set_ylabel(
        r"\textbf{--------} \ \ \ PV Installation Rate in Aarau [kWp/year]",
        color=color1,
        fontsize=fontsize,
    )

    ax2 = ax1.twinx()
    ax2.plot(
        smoothed_2,
        # data2_with_PV["rollingMean"],
        # test_data.rolling(datetime.timedelta(days=50)).sum() / 50,
        color=color2,
        label=f"PV Installation Rate in {name2} [kWp/year]",
        linestyle="--",
        linewidth=0.8,
    )
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=fontsize)
    ax2.set_ylabel(
        r"\textbf{-- -- --} \ \ \ PV Installation Rate in Thun [kWp/year]",
        color=color2,
        fontsize=fontsize,
    )
    # ax2.set_yticks([0, 500, 1000, 1500, 2000, 2500])

    ax2.annotate(
        "Introduction of Aarau's new energy support directive",
        xy=(
            pd.Timestamp("2018-03-14"),
            max(data2_with_PV["rollingMean"]) * 0.28,
        ),
        xytext=(
            pd.Timestamp("2013-11-01"),
            max(data2_with_PV["rollingMean"]) * 1.45,
            # max(data2_with_PV["rollingMean"]) * 1.4,
            # max(data2_with_PV["rollingMean"]) * 0.8,
        ),
        arrowprops=dict(facecolor="black", arrowstyle="->", lw=0.5),
        fontsize=fontsize,
    )

    fig.tight_layout()
    ax1.grid(linestyle="--", axis="y")

    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # legend = ax1.legend(
    #     handles1 + handles2,
    #     labels1 + labels2,
    #     loc="upper left",
    #     fontsize=fontsize,
    # )
    # Align zero points by having the same percentage of negative values

    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    percentage_negative = 0.01
    if y1_min < 0:
        negative_1 = -y1_min / (y1_max - y1_min)
        percentage_negative = max(percentage_negative, negative_1)
    else:
        negative_1 = 0.0
    if y2_min < 0:
        negative_2 = -y2_min / (y2_max - y2_min)
        percentage_negative = max(percentage_negative, negative_2)
    else:
        negative_2 = 0.0

    if percentage_negative != negative_1:
        y1_min = -(percentage_negative * y1_max) / (1 - percentage_negative)
        ax1.set_ylim(y1_min, y1_max)
    if percentage_negative != negative_2:
        y2_min = -(percentage_negative * y2_max) / (1 - percentage_negative)
        ax2.set_ylim(y2_min, y2_max)

    # text1.set_color(color1)
    # text2.set_color(color2)
    # plt.show()
    plt.savefig(
        os.path.join(
            "out", "plots", f"{name1}_{name2}_installation." + ending
        ),
        dpi=1200,
    )
    ######################## installation plot end #########################################

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
        os.path.join(
            "out", "plots", f"{name1}_{name2}_overview_classes." + ending
        )
    )

    # TODO: do comparison for all 5 subclasses, plot them in subplots and save result


def plot_price_vs_installation_rate(
    years: list, municipality: str, data_electricity_prices: pd.DataFrame
):
    data_elec_muni = data_electricity_prices[
        data_electricity_prices["municipality"] == municipality
    ]
    # data_elec_muni = data_electricity_prices

    # print(data_elec_muni)
    electricity_prices = [
        data_elec_muni[data_elec_muni["year"] == year]["energy"].mean()
        for year in years
    ]

    data = gpd.read_file(
        f"out/municipalities/{municipality}/{municipality}_5.0_KLASSE2.gpkg"
    )

    # data = gpd.read_file(f"out/households_5.0_KLASSE2.gpkg")
    data = data[data["KLASSE2"] <= 3]

    data_with_PV = data[data["SubCategory"] == "with_PV"]

    data_with_PV["BeginningOfOperation"] = np.asarray(
        data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
    )
    data_with_PV = data_with_PV[
        data_with_PV["BeginningOfOperation"] >= pd.to_datetime(str(min(years)))
    ]

    data_with_PV.set_index("BeginningOfOperation", inplace=True)
    data_with_PV = data_with_PV.sort_index()
    yearly_installations = (
        data_with_PV["TotalPower"]
        .cumsum()
        .resample("YE")
        .last()
        .ffill()
        .diff()
    )
    plt.plot(yearly_installations, label="yearly installations (Klasse2 <= 3)")

    # print(electricity_prices)
    # print(
    #     np.nanmax(
    #         yearly_installations.values,
    #     )
    # )
    # print(np.nanmax(electricity_prices))
    scaling = np.nanmax(yearly_installations.values) / np.nanmax(
        electricity_prices
    )
    normed_elec_prices = np.array(
        [elec * scaling for elec in electricity_prices]
    )
    plt.plot(
        [pd.to_datetime(str(year)) for year in years[1:]],
        normed_elec_prices[1:] - normed_elec_prices[:-1],
        label="electricity prices (normed)",
    )
    plt.legend()

    plt.show()


# TODO: calculate correlation between installed pv and difference (percentage) of electricity prices
def plot_correlation(
    years: list,
    municipalities: list,
    data_electricity_prices: pd.DataFrame,
    max_class=5,
):
    list_change_electricity_prices_percentage = []
    list_installation_rate_normalized = []
    # those are the years the electricity differences are calculated.
    years_diff = years[1:-1]
    for muni in tqdm(municipalities):
        if muni is None:
            continue
        muni = muni.replace("/", "_")

        try:
            data = gpd.read_file(
                f"out/municipalities/{muni}/{muni}_5.0_KLASSE2.gpkg"
            )
        except Exception as e:
            print(e)
            print(muni)
            continue

        data_loc = data_electricity_prices[
            data_electricity_prices["municipality"] == muni
        ]
        if data.empty or data_loc.empty:
            continue
        electricity_prices = np.array(
            [
                data_loc[data_loc["year"] == year]["energy"].mean()
                for year in years
            ]
        )
        # calculate difference of electricity prices (in percentages) and remove the last year to account for boundary effects
        if (
            np.min(electricity_prices) <= 0.0
            or np.isnan(electricity_prices).any()
        ):
            # print(electricity_prices, muni)
            electricity_prices[electricity_prices == 0.0] = np.nanmean(
                electricity_prices
            )
            electricity_prices[np.isnan(electricity_prices)] = np.nanmean(
                electricity_prices
            )
            # print(electricity_prices, muni)
        electricity_prices = (
            electricity_prices[1:-1] - electricity_prices[:-2]
        ) / electricity_prices[:-2]
        assert len(electricity_prices) == len(years_diff)
        # print(electricity_prices)
        # print(years_diff)

        # data = gpd.read_file(f"out/households_5.0_KLASSE2.gpkg")
        data = data[data["KLASSE2"] <= max_class]

        data_with_PV = data[data["SubCategory"] == "with_PV"]

        data_with_PV.loc[:, "BeginningOfOperation"] = pd.to_datetime(
            data_with_PV["BeginningOfOperation"], errors="coerce"
        )
        # data_with_PV.loc[:, "BeginningOfOperation"] = np.asarray(
        #     data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
        # )
        data_with_PV = data_with_PV[
            data_with_PV["BeginningOfOperation"]
            >= pd.to_datetime(str(min(years)))
        ]
        if data_with_PV.empty:
            continue

        data_with_PV = data_with_PV.infer_objects()

        data_with_PV.set_index("BeginningOfOperation", inplace=True)
        data_with_PV = data_with_PV.sort_index()
        yearly_installations = (
            data_with_PV["TotalPower"]
            .cumsum()
            .resample("YE")
            .last()
            .ffill()
            .diff()
        )
        yearly_installations = yearly_installations[
            yearly_installations.index.year.isin(years_diff)
        ]
        yearly_installations.index = yearly_installations.index.year
        # TODO: what to do if the dimensions do not match up? add 0 as installed amount?
        if len(yearly_installations) != len(years_diff):
            yearly_installations = yearly_installations.reindex(
                years_diff, fill_value=0
            )

        if sum(yearly_installations) == 0.0:
            continue
        list_change_electricity_prices_percentage.extend(electricity_prices)
        # print(yearly_installations)
        yearly_installations = yearly_installations / sum(yearly_installations)
        # print(yearly_installations)
        list_installation_rate_normalized.append(yearly_installations.values)

    list_installation_rate_normalized = np.concatenate(
        list_installation_rate_normalized
    )
    list_change_electricity_prices_percentage = np.asarray(
        list_change_electricity_prices_percentage
    )
    mask = ~(
        np.isnan(list_installation_rate_normalized)
        | np.isnan(list_change_electricity_prices_percentage)
    )
    list_installation_rate_normalized = list_installation_rate_normalized[mask]
    list_change_electricity_prices_percentage = (
        list_change_electricity_prices_percentage[mask]
    )

    # print(list_installation_rate_normalized)
    # print(list_change_electricity_prices_percentage)
    plt.plot(
        list_change_electricity_prices_percentage,
        list_installation_rate_normalized,
        linestyle="none",
        marker=".",
    )

    # Step 1: Fit a line y = m*x + b
    m, b = np.polyfit(
        np.array(list_change_electricity_prices_percentage),
        np.array(list_installation_rate_normalized),
        deg=1,
    )

    # Step 2: Generate x values for the line (spanning the domain of l1)
    x_fit = np.linspace(
        min(list_change_electricity_prices_percentage),
        max(list_change_electricity_prices_percentage),
        100,
    )
    y_fit = m * x_fit + b
    plt.plot(x_fit, y_fit)

    installation_rate_predication = (
        m * list_change_electricity_prices_percentage + b
    )

    ss_res = np.sum(
        (installation_rate_predication - list_installation_rate_normalized)
        ** 2
    )
    ss_tot = np.sum(
        (
            np.mean(list_installation_rate_normalized)
            - list_installation_rate_normalized
        )
        ** 2
    )
    r_squared = 1 - (ss_res / ss_tot)
    print(f"r2 for max_class: {max_class} is {r_squared}")

    plt.savefig(f"out/correlations_max_{max_class}.pdf")
    # print(yearly_installations)
    # print(yearly_installations.index.year)

    pass


def generate_installation_rate_vs_electricity_price_plot(municipalities: list):
    years = [year for year in range(2009, 2025)]
    data_electricity_prices = get_merged_data(years)
    # for municipality in municipalities:
    #     plot_price_vs_installation_rate(
    #         years, municipality, data_electricity_prices
    #     )

    for i in range(1, 6):
        plot_correlation(
            years, municipalities, data_electricity_prices, max_class=i
        )


# working but slow, could be much more efficient
def generate_evolution_classes_switzerland(
    # data: gpd.GeoDataFrame, dirname: str, filename: str, app: str = ""
):
    plt.figure(figsize=(6, 4))
    data = gpd.read_file("out/households_5.0_KLASSE2.gpkg")

    plt.title(
        "Overview of installation of PV plants based on class of rooftop"
    )
    plt.ylabel("total Power installed [kW]")
    plt.xlabel("date")
    data_with_PV = data[data["SubCategory"] == "with_PV"]
    # unique_households_with_pv = data_with_PV["SB_UUID"].unique()
    # TODO: create an efficient datastructure with 3 different entries: BeginningOfOperation(time format), class (1-5), TotalPower(float)
    # create 5 series to plot them efficiently

    # skip empty data, could be done better
    # is_empty = True
    # for idx in datapoints.keys():
    #     if not datapoints[idx].empty:
    data_with_PV.loc[:, "BeginningOfOperation"] = pd.to_datetime(
        data_with_PV["BeginningOfOperation"], errors="coerce"
    )
    # data_with_PV.loc[:, "BeginningOfOperation"] = np.asarray(
    #     data_with_PV["BeginningOfOperation"], dtype="datetime64[s]"
    # )
    data_with_PV = data_with_PV[
        data_with_PV["BeginningOfOperation"] >= pd.to_datetime("2014")
    ]

    data_with_PV = data_with_PV.infer_objects()

    data_with_PV.set_index("BeginningOfOperation", inplace=True)
    data_with_PV = data_with_PV.sort_index()

    cumulated_values = pd.DataFrame(
        data=0.0,
        index=pd.date_range(
            start=data_with_PV.index.min(),
            end=data_with_PV.index.max(),
            freq="D",
        ),
        columns=["TotalPower"],
    )
    plt.plot(cumulated_values)
    print(cumulated_values)
    for class_idx in range(1, 6):
        class_data = data_with_PV[data_with_PV["KLASSE2"] == class_idx]
        power_unique = class_data["TotalPower"].groupby(class_data.index).sum()
        tmp = cumulated_values["TotalPower"].add(power_unique, fill_value=0)
        plt.fill_between(
            cumulated_values.index,
            cumulated_values["TotalPower"].cumsum(),
            tmp.cumsum(),
            alpha=0.7,
            label=f"rooftop class {class_idx}",
        )
        cumulated_values["TotalPower"] = tmp
    plt.legend()

    plt.savefig("out/Switzerland/evolution_switzerland.jpg", dpi=1200)
