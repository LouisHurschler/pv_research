import json
import re
import string

import folium
import mapclassify
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import requests

from graphly.api_client import SparqlClient
import numpy as np


# Truncate colormap: keep only the darker 80%–100% of viridis
def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def get_electricity_data_of_specific_year(
    year: int, sparql: SparqlClient
) -> pd.DataFrame:
    query = f"""
    SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts
    FROM <https://lindas.admin.ch/elcom/electricityprice>
    WHERE {{
        <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
    
        ?observation
          elcom:category/schema:name ?category;
          elcom:municipality ?municipality_id;
          elcom:period "{year}"^^<http://www.w3.org/2001/XMLSchema#gYear>;
          elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
          elcom:fixcosts ?fixcosts;
          elcom:total ?variablecosts;
          elcom:gridusage ?grid;
          elcom:energy ?energy;
          elcom:charge ?community_fees;
          elcom:aidfee ?aidfee.
      
    }}
    ORDER BY ?muncipality ?category ?variablecosts
    """
    tariffs = sparql.send_query(query)
    tariffs["year"] = year
    return tariffs


def get_electricity_data(
    years: list = [2020], categories: list = ["H1"]
) -> pd.DataFrame:

    sparql = SparqlClient("https://lindas.admin.ch/query")

    sparql.add_prefixes(
        {
            "schema": "<http://schema.org/>",
            "cube": "<https://cube.link/>",
            "elcom": "<https://energy.ld.admin.ch/elcom/electricityprice/dimension/>",
            "admin": "<https://schema.ld.admin.ch/>",
        }
    )
    results = pd.concat(
        [
            get_electricity_data_of_specific_year(year, sparql)
            for year in years
        ],
        ignore_index=True,
    )
    results = results[results["category"].isin(categories)]

    return results


# get elevation data from external api, generated with chatgpt
def get_elevation(easting, northing):
    # print(easting, northing)
    url = f"https://api3.geo.admin.ch/rest/services/height?easting={easting}&northing={northing}&sr=2056"
    try:
        response = requests.get(url)
    except Exception as e:
        print("some exception has happened. returning none")
        print(e)
        return None

    if response.status_code == 200:
        return response.json().get("height")
    print("response code was ", response.status_code)
    print(response.json())
    return None


def get_municipal_data() -> gpd.GeoDataFrame:
    geosparql = SparqlClient("https://geo.ld.admin.ch/query")

    geosparql.add_prefixes(
        {
            "dct": "<http://purl.org/dc/terms/>",
            "geonames": "<http://www.geonames.org/ontology#>",
            "schema": "<http://schema.org/>",
            "geosparql": "<http://www.opengis.net/ont/geosparql#>",
        }
    )

    query = """
        
    SELECT ?municipality_id ?municipality ?population ?boundary

    WHERE {
      ?muni_iri dct:hasVersion ?version ;
                geonames:featureCode geonames:A.ADM3 .
  
      ?version schema:validUntil "2020-12-31"^^<http://www.w3.org/2001/XMLSchema#date>;
               geonames:population ?population;
               schema:name ?municipality;
               geosparql:hasGeometry/geosparql:asWKT ?boundary.

      BIND(IRI(REPLACE(STR(?muni_iri), "https://geo.ld.admin.ch/boundaries/", "https://ld.admin.ch/")) AS ?municipality_id)
    }    
    """

    muni = geosparql.send_query(query)

    muni = muni.set_crs(epsg=4326)
    muni_centroids = gpd.GeoDataFrame(
        geometry=muni.geometry.centroid, crs="EPSG:4326"
    ).to_crs(epsg=2056)
    muni["easting"] = muni_centroids.geometry.x
    muni["northing"] = muni_centroids.geometry.y

    muni["elevation"] = muni.apply(
        lambda row: get_elevation(row["easting"], row["northing"]), axis=1
    )
    muni.plot("elevation")
    plt.show()
    return muni

    # H3: 4-zimmerwohnung mit Elektroherd und Elektroboiler


def get_merged_data(years: list, categories: list = ["H3"]) -> pd.DataFrame:

    data_available = True

    if data_available:
        merged = gpd.read_file("data/municipal_and_energy_data.gpkg")
    else:

        tariffs = get_electricity_data(years, ["H3"])
        muni_data = get_municipal_data()
        muni_data = muni_data.set_crs(epsg=4326)
        muni_data.to_file("data/muni_boundaries.gpkg")
        print(muni_data)

        merged = muni_data.merge(
            tariffs,
            how="inner",
            on="municipality_id",
        )
        merged = merged.set_crs(epsg=4326)
        merged.to_file("data/municipal_and_energy_data.gpkg")

    return merged


if __name__ == "__main__":
    years = [year for year in range(2015, 2025)]
    merged = get_merged_data(years)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    merged["elevation"] = merged["elevation"].astype(float)
    print(merged.keys())

    custom_cmap = truncate_colormap(cm.viridis, -0.6, 0.8)
    # merged.plot(
    #     column="elevation",
    #     cmap=custom_cmap,
    #     legend=True,
    #     ax=ax,
    #     # linewidth=0.5,
    #     # edgecolor="0.8",
    #     # vmin=-2000.0,
    #     legend_kwds={
    #         "label": "elevation of municipalitites",
    #         "shrink": 0.6,
    #         "orientation": "vertical",
    #     },
    # )
    # merged = merged[merged["year"] > 2017]

    # plt.title("heatmap of elevation in switzerland")
    # plt.savefig("out/elevation.jpg", dpi=1200)
    # print(merged)
    # merged.plot()
    # plt.show()
    mean_energy_prices = (
        merged.groupby("municipality")["energy"].mean().rename("mean_energy")
    )
    merged_with_energy_prices = merged.drop_duplicates("municipality").merge(
        mean_energy_prices, on="municipality", how="left"
    )
    merged_with_energy_prices.plot(
        column="mean_energy",
        cmap=custom_cmap,
        legend=True,
        ax=ax,
        legend_kwds={
            "label": "mean energy price (2014 - 2024)",
            "shrink": 0.6,
            "orientation": "vertical",
        },
        # linewidth=0.0,
    )
    plt.title("Mean Energy Prices Switzerland")
    ax.axis("off")
    plt.savefig("out/energy_prices.jpg", dpi=1200)
    # for year in years:
    #     merged[merged["year"] == year].plot("energy")
    #     plt.show()

    # print(merged[merged["municipality"] == "Sempach"])
    # prices_years = [
    #     merged[merged["year"] == year]["energy"].mean() for year in years
    # ]
    # plt.plot(years, prices_years)
    # plt.show()
