import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


from plot_percentage_built_switzerland_overview import (
    get_percentage_built_from_data,
    change_names,
)

gpd.options.io_engine = "pyogrio"
os.environ["PYOGRIO_USE_ARROW"] = "1"


data_switzerland = pd.read_excel("data/je-d-21.03.01.xlsx", header=5)

# cleanup data from excel
# remove unnececary information
data_switzerland = data_switzerland[3:-16]

# replace nonnumeric values with mean
# Convert to numeric, forcing errors to NaN for non-numeric entries
muni_id = data_switzerland["Gemeindename"]
data_switzerland = data_switzerland.drop(columns=["Gemeindename"])
data_switzerland = data_switzerland.apply(pd.to_numeric, errors="coerce")

# Replace NaNs (i.e., non-numeric values) with column means
data_switzerland = data_switzerland.apply(lambda col: col.fillna(col.mean()))
data_switzerland["municipality"] = muni_id
data_switzerland = change_names(data_switzerland)

data_installation_rate = gpd.read_file("out/households_5.0_KLASSE2.gpkg")
data_installation_rate = get_percentage_built_from_data(data_installation_rate)

merged = data_installation_rate.merge(
    data_switzerland,
    how="left",
    left_on="GDE_NAME",
    right_on="municipality",
)
data_boundaries = change_names(gpd.read_file("data/muni_boundaries.gpkg"))
merged = data_boundaries.merge(
    merged,
    how="left",
    on="municipality",
)
# print(merged)


def calculate_correlation(data: pd.DataFrame, left_col: str, right_col: str):

    try:
        correlation = data[left_col].corr(data[right_col])

        plt.plot(data[left_col], data[right_col], linestyle="none", marker=".")
        plt.title(left_col)
        plt.show()
        plt.close()
        if abs(correlation) > 10.3:
            print(
                f"correlation between {left_col} and {right_col} is {correlation}"
            )
            merged[left_col] = merged[left_col].astype(float)
            merged.plot(column=left_col, cmap="viridis", legend=True)
            plt.title(left_col)
            # plt.show()
            plt.savefig(
                f"out/plots/correlation_{left_col.replace("/", "_")}_pv_coverage.jpg",
                dpi=1200,
            )
            plt.close()

    except Exception as e:
        print(e)
        print(left_col)

        # plt.plot(data[left_col], data[right_col], linestyle="none", marker=".")
        # plt.show()


not_relevant = [
    "municipality_id",
    "municipality",
    "count_with",
    "total",
    "percentage",
    "easting",
    "northing",
    "geometry",
]
for key in merged.keys():
    if key in not_relevant:
        continue

    calculate_correlation(merged, key, "percentage")


# tested some linear regression models, r2 ~ 0.4
merged["percentage"] = pd.to_numeric(merged["percentage"], errors="coerce")
merged = merged.dropna(subset=["percentage"])

percentages = merged["percentage"]


merged.drop(columns=not_relevant, inplace=True)


# model = LinearRegression()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(merged)
X_scaled = pd.DataFrame(X_scaled, columns=merged.columns)
print(X_scaled)
# plt.plot(X_scaled["population"], percentages, linestyle="none", marker=".")
# plt.plot(merged["population"], percentages, linestyle="none", marker=".")
# plt.show()
# plt.close()

percentages = percentages.reset_index(drop=True)
print(X_scaled.corrwith(percentages))
print(merged.corrwith(percentages))


results = pd.DataFrame()
# plot lasso weights. alpha=0.35 reduces it to about 5 good indicators
# for i in range(100):
#     model = Lasso(alpha=i / 100.0)
#     model.fit(X_scaled, percentages)
#     results[i] = model.coef_

# for _, row in results.iterrows():
#     plt.plot(row.values)

# plt.show()
# plt.close()

model = Lasso(alpha=0.35)
# model = KernelRidge(alpha=0.1, kernel="laplacian")
model.fit(X_scaled, percentages)

y_pred = model.predict(X_scaled)
r_squared = model.score(X_scaled, percentages)

coef_df = pd.DataFrame(
    {
        "feature": X_scaled.columns,
        "coefficient": model.coef_,
        "correlation": X_scaled.corrwith(percentages),
    }
)


print(coef_df)
print(r_squared)

print(percentages.corr(pd.Series(y_pred)))
plt.plot(y_pred, percentages, linestyle="none", marker=".")
plt.show()
