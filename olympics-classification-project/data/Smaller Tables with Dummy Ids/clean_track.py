import pandas as pd
import numpy as np

track = pd.read_csv("track_field/track_field_raw.csv", usecols=range(8), encoding="utf-8")

def create_event_id(event_name, year, location):
    stripped_name = str(event_name).strip().replace(" ", "_").upper()
    return f"{stripped_name}_{int(year)}_{str(location).strip()}"


track["event_name"] = track["Event"]
track["event_type"] = "Athletics"

track["athlete_str"] = track["Name"].str.strip().str.replace(" ", "_").str.upper()
mask = track["athlete_str"].isna()
track.loc[mask, "athlete_str"] = track.loc[mask, "Nationality"] + track.loc[mask, "Year"].astype(str)


track_olympics = track[["Year", "Location"]].drop_duplicates()

track_olympics = track_olympics.rename(columns={
    "Year": "year",
    "Location": "host_city_name"
})

# Track only takes places in the summer
track_olympics["summer"] = 1

track_olympics["olympics_id"] = range(1, len(track_olympics) + 1)

track_olympics = track_olympics[["olympics_id", "year", "summer", "host_city_name"]]

track = track.merge(
    track_olympics,
    left_on=["Year", "Location"],
    right_on=["year", "host_city_name"],
    how="left"
)

track = track.drop(columns=["year", "host_city_name", "summer"])
track_event_ids = []
for _, row in track.iterrows():
    track_event_ids.append(create_event_id(row["event_name"], row["Year"], row["Location"]))

track["event_str"] = track_event_ids

track["athlete_id"], athlete_index = pd.factorize(track["athlete_str"])
track["event_id"], event_index = pd.factorize(track["event_str"])

track_events = track[["event_id", "olympics_id", "event_name", "event_type"]].drop_duplicates()

gold_map = (
    track[track["Medal"] == "G"]
    .groupby("event_id")["athlete_id"]
    .first()
)

silver_map = (
    track[track["Medal"] == "S"]
    .groupby("event_id")["athlete_id"]
    .first()
)

bronze_map = (
    track[track["Medal"] == "B"]
    .groupby("event_id")["athlete_id"]
    .first()
)

track_events["gold_id"]   = track_events["event_id"].map(gold_map)
track_events["silver_id"] = track_events["event_id"].map(silver_map)
track_events["bronze_id"] = track_events["event_id"].map(bronze_map)

track_events = track_events[["event_id", "gold_id", "silver_id", "bronze_id", "olympics_id", "event_name", "event_type"]]

track_candidates = track[["athlete_id", "Nationality", "Gender", "Name"]].drop_duplicates()

names = track_candidates["Name"].str.split()

track_candidates["first_name"] = names.str[0]
track_candidates["last_name"]  = names.str[-1]

if len(names) > 2:
    track_candidates["middle_name"] = names.str[1:-1].str.join(" ")
else:
    track_candidates["middle_name"] = np.nan
track_candidates = track_candidates.rename(columns={"Nationality": "nationality_id", "Gender": "gender"})

track_candidates["height(cm)"] = np.nan
track_candidates["weight(kg)"] = np.nan
track_candidates["year_of_birth"] = np.nan
track_candidates["nationality_at_birth"] = np.nan
track_candidates["gender"] = track_candidates["gender"].replace({"M": "Men", "W": "Women"})

track_candidates = track_candidates[["athlete_id", "first_name", "middle_name", "last_name",
                                     "height(cm)", "weight(kg)", "nationality_id",
                                     "year_of_birth", "nationality_at_birth", "gender"]]

track_combined = track[["athlete_id", "event_id", "Result", "Medal"]].drop_duplicates()
track_combined = track_combined.rename(columns={"Result": "time"})
track_combined["result"] = track_combined["Medal"].replace({"G": 1, "S": 2, "B": 3})
track_combined = track_combined[["athlete_id", "event_id", "time", "result"]]

track_events.to_csv("track_events.csv", index=False)
track_candidates.to_csv("track_candidates.csv", index=False)
track_combined.to_csv("track_combined.csv", index=False)
track_olympics.to_csv("track_olympics.csv", index=False)

