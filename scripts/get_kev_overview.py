import pandas as pd


# used for testing some things, not relevant
data = pd.read_csv("data/ogd6_kev-bezueger.csv")
data = data[data["Anlage_energietraeger"] == "Photovoltaik"]
data = data[data["Leistung_kw"] <= 100]
data = data.groupby("Anlage_ort").agg(
    {"Leistung_kw": "sum", "Produktion_kwh": "sum"}
)
print(data)
print(data["Leistung_kw"].sum(), data["Produktion_kwh"].sum())
data.to_csv("data/kev_overview_under_100kw.csv")
