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

    roof_count_with = data_with_PV[class_name].value_counts().sort_index()
    roof_count_without = (
        data_without_PV[class_name].value_counts().sort_index()
    )

    if len(roof_count_with) != len(roof_count_without):
        if len(roof_count_with) != 5:
            roof_count_with = fill_missing_indexes(roof_count_with)

        if len(roof_count_without) != 5:
            roof_count_without = fill_missing_indexes(roof_count_without)
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

    fig, ax1 = plt.subplots(figsize=(5, 4))

    plt.title(
        f"Distribution of PV Installations: Municipality {filename}",
        fontsize=fontsize,
    )
    # color_green = "#2CA02C"
    # color_red = "#D62728"
    color_green = "#27AE60"
    color_red = "#E74C3C"
    color_blue = "#1F77B4"

    # add barplots
    ax1.bar(
        roof_count_with.index,
        roof_count_with.values,
        label="Roofs with PV",
        color=color_green,
    )
    ax1.bar(
        roof_count_without.index,
        roof_count_without.values,
        label="Roofs without PV",
        bottom=roof_count_with.values,
        color=color_red,
    )
    ax1.set_xlabel("Rooftop PV-Suitability", fontsize=fontsize)

    percentage_pv = (
        100 * roof_count_with / (roof_count_with + roof_count_without)
    )
    max_percentage = max(percentage_pv)
    max_value = max(roof_count_with + roof_count_without)
    ax1.set_ylabel("Number of Rooftops")
    # plt.legend(loc="upper left")
    handles1, labels1 = ax1.get_legend_handles_labels()
    if filename == "Aarau":
        roof_count_with = roof_count_with.drop(5)
        percentage_pv = percentage_pv.drop(5)

    ax2 = ax1.twinx()
    ax2.plot(
        roof_count_with.index,
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
        r"Percentage of Roofs with PV Systems [\%]", fontsize=fontsize
    )

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    plt.xlabel("Klasse")
    # plt.legend(loc="upper right")
    if not os.path.exists(os.path.join(dirname, filename)):
        os.makedirs(os.path.join(dirname, filename))

    plt.savefig(
        os.path.join(dirname, filename, f"roof_quality_{filename}.pdf")
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
        os.path.join("out", "plots", f"{name1}_{name2}_comparison.svg")
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
    ax1.set_title(
        f"PV Installation Rate Comparison: Municipalities {name1} and {name2}",
        fontsize=fontsize,
    )
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
        os.path.join("out", "plots", f"{name1}_{name2}_installation.svg")
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
        os.path.join("out", "plots", f"{name1}_{name2}_overview_classes.svg")
    )

    # TODO: do comparison for all 5 subclasses, plot them in subplots and save result
