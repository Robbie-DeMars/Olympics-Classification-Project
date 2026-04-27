import pandas as pd
import numpy as np

df = pd.read_csv("weight/athlete_height_weight.csv", encoding="utf-8")


def create_event_id(event_name, year, location):
    stripped_name = str(event_name).strip().replace(" ", "_").upper()
    return f"{stripped_name}_{int(year)}_{str(location).strip()}"

olympics =df[["Year", "City", "Season"]].drop_duplicates()
olympics = olympics.rename(columns={
    "Year": "year",
    "City": "host_city_name",
    "Season": "season"})

olympics["summer"] = (olympics["season"] == "Summer").astype(int)
olympics["olympics_id"] = range(1, len(olympics) + 1)
olympics = olympics[["olympics_id", "year", "summer", "host_city_name"]]

df = df.merge(
    olympics,
    left_on=["Year", "City"],
    right_on=["year", "host_city_name"],
    how="left"
)

df = df.drop(columns=["year", "host_city_name", "summer"])

df["athlete_str"] = df["Name"].str.strip().str.replace(" ", "_").str.upper()

event_ids = []
for _, row in df.iterrows():
    event_ids.append(create_event_id(row["Event"], row["Year"], row["City"]))

df["event_str"] = event_ids

df["athlete_id"], athlete_index = pd.factorize(df["athlete_str"])
df["event_id"], event_index = pd.factorize(df["event_str"])

event = df[["event_id", "olympics_id", "Event", "Sport"]].drop_duplicates()
event = event.rename(columns={"Event": "event_name", "Sport": "event_type"})

gold_map = (
    df[df["Medal"] == "Gold"]
    .groupby("event_id")["athlete_id"]
    .first()
)

silver_map = (
    df[df["Medal"] == "Silver"]
    .groupby("event_id")["athlete_id"]
    .first()
)

bronze_map = (
    df[df["Medal"] == "Bronze"]
    .groupby("event_id")["athlete_id"]
    .first()
)

event["gold_id"]   = event["event_id"].map(gold_map)
event["silver_id"] = event["event_id"].map(silver_map)
event["bronze_id"] = event["event_id"].map(bronze_map)
event = event[[
    "event_id", "gold_id", "silver_id", "bronze_id",
    "olympics_id", "event_name", "event_type"
]]

df["year_of_birth"] = df["Year"] - df["Age"]

athlete = df[["athlete_id", "Name", "NOC", "Sex", "Height", "Weight", "year_of_birth"]].drop_duplicates()

name_parts = athlete["Name"].str.split()
athlete["first_name"]  = name_parts.str[0]
athlete["last_name"]   = name_parts.str[-1]
if len(name_parts) > 2:
    athlete["middle_name"] = name_parts.str[1:-1].str.join(" ")
else:
    athlete["middle_name"] = np.nan

athlete = athlete.rename(columns={
    "NOC":    "nationality_id",
    "Sex":    "gender",
    "Height": "height(cm)",
    "Weight": "weight(kg)",
})

athlete["nationality_at_birth"] = np.nan
athlete["gender"] = athlete["gender"].replace({"M": "Men", "F": "Women"})

athlete = athlete[[
    "athlete_id", "first_name", "middle_name", "last_name",
    "height(cm)", "weight(kg)", "nationality_id",
    "year_of_birth", "nationality_at_birth", "gender"
]]


combined = df[df["Medal"].notna()][["athlete_id", "event_id", "Medal"]].drop_duplicates()

combined["time"]   = np.nan
combined["result"] = combined["Medal"].replace({"G": 1, "S": 2, "B": 3})
combined = combined[["athlete_id", "event_id", "time", "result"]]

event.to_csv("weight_height_events.csv", index=False)
athlete.to_csv("weight_height_candidates.csv", index=False)
combined.to_csv("weight_height_combined.csv", index=False)
olympics.to_csv("weight_height_olympics.csv", index=False)

