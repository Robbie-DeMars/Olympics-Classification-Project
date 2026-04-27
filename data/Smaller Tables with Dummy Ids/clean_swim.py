import pandas as pd
import numpy as np


swim = pd.read_csv(
    "swimming_data/swimming_raw.csv", encoding="utf-8")

def create_event_id(event_name, year, location):
    stripped_name = str(event_name).strip().replace(" ", "_").upper()
    return f"{stripped_name}_{int(year)}_{str(location).strip()}"

swim["event_name"] = (swim["Distance (in meters)"].astype(str) + " "
    + swim["Stroke"] + " "
    + swim["Gender"]
)

for i, row in swim.iterrows():
    if str(row["Relay?"]).strip() == "1":
        swim.at[i, "event_name"] = swim.at[i, "event_name"] + " Relay"


swim["event_type"] = "Swimming"
swim["athlete_str"]  = swim["Athlete"].str.strip().str.replace(" ", "_").str.upper()

swim_olympics = swim[["Year", "Location"]].drop_duplicates()

swim_olympics = swim_olympics.rename(columns={
    "Year": "year",
    "Location": "host_city_name"
})

# Swim is a summer sport only
swim_olympics["summer"] = 1

swim_olympics["olympics_id"] = range(1, len(swim_olympics) + 1)

swim_olympics = swim_olympics[["olympics_id", "year", "summer", "host_city_name"]]

swim = swim.merge(
    swim_olympics,
    left_on=["Year", "Location"],
    right_on=["year", "host_city_name"],
    how="left"
)
swim = swim.drop(columns=["year", "host_city_name", "summer"])

swim_event_ids = []
for _, row in swim.iterrows():
    swim_event_ids.append(create_event_id(row["event_name"], row["Year"], row["Location"]))

swim["event_str"] = swim_event_ids

swim["athlete_id"], athlete_index = pd.factorize(swim["athlete_str"])
swim["event_id"], event_index = pd.factorize(swim["event_str"])

swim_events = swim[["event_id", "olympics_id", "event_name", "event_type"]].drop_duplicates()

gold_map = (
    swim[swim["Rank"] == 1]
    .groupby("event_id")["athlete_id"]
    .first()
)

silver_map = (
    swim[swim["Rank"] == 2]
    .groupby("event_id")["athlete_id"]
    .first()
)

bronze_map = (
    swim[swim["Rank"] == 3]
    .groupby("event_id")["athlete_id"]
    .first()
)

# Map them onto the events table
swim_events["gold_id"] = swim_events["event_id"].map(gold_map)
swim_events["silver_id"] = swim_events["event_id"].map(silver_map)
swim_events["bronze_id"] = swim_events["event_id"].map(bronze_map)

swim_events = swim_events[["event_id", "gold_id", "silver_id", "bronze_id", "olympics_id", "event_name", "event_type"]]

swim_candidates = swim[["athlete_id", "Team", "Gender", "Athlete"]].drop_duplicates()

names = swim_candidates["Athlete"].str.split()

swim_candidates["first_name"] = names.str[0]
swim_candidates["last_name"]  = names.str[-1]

if len(names) > 2:
    swim_candidates["middle_name"] = names.str[1:-1].str.join(" ")
else:
    swim_candidates["middle_name"] = np.nan

swim_candidates = swim_candidates.rename(columns={"Team": "nationality_id", "Gender": "gender"})

swim_candidates["height(cm)"] = np.nan
swim_candidates["weight(kg)"] = np.nan
swim_candidates["year_of_birth"] = np.nan
swim_candidates["nationality_at_birth"] = np.nan

swim_candidates = swim_candidates[["athlete_id", "first_name", "middle_name", "last_name",
                          "height(cm)", "weight(kg)", "nationality_id",
                          "year_of_birth", "nationality_at_birth", "gender"]]

swim_combined = swim[["athlete_id", "event_id", "Results", "Rank"]].drop_duplicates()
swim_combined = swim_combined.rename(columns={"Results": "time", "Rank": "result"})

# Weird issue with the dataset where any placement that isn't 1-3 is given a 4 as a default
# 0 = DNF
# 5 = people without a time
# Opted to just give them a NaN since the numbers don't have a real placement meaning
swim_combined.loc[swim_combined["result"].isin([4, 5, 0]), "result"] = np.nan

swim_events.to_csv("swim_events.csv", index=False)
swim_candidates.to_csv("swim_candidates.csv", index=False)
swim_combined.to_csv("swim_combined.csv", index=False)
swim_olympics.to_csv("swim_olympics.csv", index=False)