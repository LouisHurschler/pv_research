import matplotlib.pyplot as plt
import os
import geopandas as gpd
import numpy as np
import pandas as pd


def fill_missing_indexes(data: gpd.GeoDataFrame):
    # print(type(data))
    full_index = pd.Series(0, index=range(1, 6))
    full_index.update(data)

    return full_index


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
    datapoints = {
        "1": pd.Series(dtype=float),
        "2": pd.Series(dtype=float),
        "3": pd.Series(dtype=float),
        "4": pd.Series(dtype=float),
        "5": pd.Series(dtype=float),
    }

    for unique_uuid in unique_households_with_pv:
        house_data = data[data["SB_UUID"] == unique_uuid]
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

        datapoints[str(best_class)] = pd.concat(
            [
                datapoints[str(best_class)],
                pd.Series([total_power], index=[beginning_of_operation]),
            ]
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

    whole_data = pd.concat(
        [datapoints[idx] for idx in datapoints.keys()]
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
    data_with_large_plants = data[data["TotalPower"] >= threshold]
    for uuid in data_with_large_plants["SB_UUID"].unique():
        data = data[data["SB_UUID"] != uuid]
    return data


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
