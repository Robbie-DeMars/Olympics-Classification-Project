from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from itertools import combinations
from graphframes import GraphFrame
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import when, col, lit, concat
import re
import os
import sys

#############################

# SETUP

#############################

# Python 3.11
# PySpark 3.5.3
# GraphFrames 0.8.3-spark3.5-s_2.12
# Java: OpenJDK 17

# Installation
#     python3 -m venv venv
#     source venv/bin/activate
#     pip install pyspark==3.5.3 graphframes==0.8.3

spark = SparkSession.builder \
    .appName("olympic_joins") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .config("spark.local.dir", "/home/logan/tmp") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("file:///mnt/c/temp/spark-checkpoint")

# EVENTS DF'S
events = spark.read.csv("./datasets/Scraped Data/events.csv", header=True, inferSchema=True)
swim_events = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_events.csv", header=True, inferSchema=True)
track_events = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_events.csv", header=True, inferSchema=True)


# DATASET LABELS FOR EVENTS
swim_events = swim_events.withColumn("source", F.lit("swim_events"))
track_events = track_events.withColumn("source", F.lit("track_events"))
events = events.withColumn("source", F.lit("base_events"))

# BIG UNIFIED TABLE
events_all = events.unionByName(swim_events.unionByName(track_events))
events_all = events_all.withColumn("row_id", F.monotonically_increasing_id())

# event_clusters = events_all.select("event_id", "source", "cluster_id")

# swim_events = swim_events.withColumn("source", F.lit("swim_events"))
# track_events = track_events.withColumn("source", F.lit("track_events"))
# events = events.withColumn("source", F.lit("base_events"))

# swim_events = swim_events.join(event_clusters, on=["event_id", "source"], how="left")
# track_events = track_events.join(event_clusters, on=["event_id", "source"], how="left")
# events = events.join(event_clusters, on=["event_id", "source"], how="left")


# ============================================
# ============================================
###### PARSE EVENT ##########
# ============================================
# ============================================

schema = StructType([
    StructField("distance", IntegerType(), True),
    StructField("gender", StringType(), True),
    StructField("discipline", StringType(), True),
    StructField("round", StringType(), True),
])

def parse_event(name, event_type):
    try:
        if name is None or not isinstance(name, str):
            name_str = str(name) if name is not None else ""
        else:
            name_str = name
        text = name_str.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        
        e_type = str(event_type).lower() if event_type is not None else ""

        # ===== distance =====
        dist_match = re.search(r"(\d+)\s*(m|meter|metre|km)?", text)
        distance = None
        if dist_match:
            distance = int(dist_match.group(1))
            unit = dist_match.group(2)
            if unit and "k" in unit:
                distance *= 1000

        # ===== gender =====
        gender = None
        if re.search(r"\bmen\b|\bmens\b", text):
            gender = "men"
        elif re.search(r"\bwomen\b|\bwomens\b", text):
            gender = "women"

        # ===== round =====
        round_type = None
        if "heat" in text:
            round_type = "heat"
        elif "semi" in text:
            round_type = "semifinal"
        elif "final" in text:
            round_type = "final"
        elif "summary" in text or "result" in text:
            round_type = "summary"
        elif "repechage" in text:
            round_type = "repechage"

        # ===== discipline =====
        discipline = None
        e_type = event_type.lower() if event_type else ""

        if e_type == "swimming":
            if "freestyle" in text:
                discipline = "freestyle"
            elif "backstroke" in text:
                discipline = "backstroke"
            elif "breaststroke" in text:
                discipline = "breaststroke"
            elif "butterfly" in text:
                discipline = "butterfly"

        elif e_type == "athletics":
            if "hurdle" in text:
                discipline = "hurdles"
            elif "steeplechase" in text:
                discipline = "steeplechase"
            elif "relay" in text:
                discipline = "relay"
            elif "marathon" in text:
                discipline = "marathon"
            elif "race walk" in text:
                discipline = "race_walk"
            elif "jump" in text:
                discipline = "jump"
            elif "throw" in text or "shot put" in text or "discus" in text:
                discipline = "throw"
            elif distance:
                discipline = "sprint"  # fallback

        return {
            "distance": distance,
            "gender": gender,
            "discipline": discipline,
            "round": round_type
    }
    except Exception as e:
        sys.stderr.write(f"ERROR in parse_event: name={repr(name)}, event_type={repr(event_type)}\n")
        sys.stderr.write(f"Exception: {e}\n")
        return None

parse_udf = F.udf(lambda e_name, e_type: parse_event(e_name, e_type), schema)

events_all = events_all.withColumn(
    "parsed",
    parse_udf("event_name", "event_type")
)

events_all = events_all.select(
    "*",
    "parsed.distance",
    "parsed.gender",
    "parsed.discipline",
    "parsed.round"
).drop("parsed")

events_all.select("event_name", "distance", "gender", "discipline", "round").show(5, truncate=False)

# Catch any crappy disciplines with tokens
events_all = events_all.withColumn(
    "tokens",
    F.array_except(
        F.split(F.lower(F.regexp_replace("event_name", r"[^\w\s]", "")), " "),
        F.array(F.lit("mens"), F.lit("womens"), F.lit("heat"), F.lit("final"))
    )
)

events_all = events_all.filter(
    ~F.col("event_name").rlike("summary|results summary")
)

pairs = events_all.alias("a").join(
    events_all.alias("b"),
    on=["olympics_id", "event_type"],
).filter(F.col("a.row_id") < F.col("b.row_id")) \

pairs = pairs.withColumn(
    "distance_match",
    F.when(F.col("a.distance") == F.col("b.distance"), 1).otherwise(0)
)

pairs = pairs.withColumn(
    "discipline_match",
    F.when(F.col("a.discipline") == F.col("b.discipline"), 1).otherwise(0)
)

pairs = pairs.withColumn(
    "gender_match",
    F.when(F.col("a.gender") == F.col("b.gender"), 1).otherwise(0)
)

def jaccard(colA, colB):
    try:
        return F.when(
            colA.isNull() | colB.isNull(), 0.0
        ).when(
            F.size(F.array_union(colA, colB)) == 0, 0.0
        ).otherwise(
            F.size(F.array_intersect(colA, colB)) / F.size(F.array_union(colA, colB))
        )
    except Exception:
        return F.lit(0.0)

pairs = pairs.withColumn(
    "token_sim",
    jaccard(F.col("a.tokens"), F.col("b.tokens"))
)

pairs = pairs.withColumn(
    "score",
    0.35 * F.col("distance_match") +
    0.2 * F.col("gender_match") +
    0.25 * F.col("token_sim") +
    0.2 * F.col("discipline_match")
)

matches = pairs.filter(F.col("score") >= 0.8)

vertices = events_all.select(F.col("row_id").alias("id"))

edges = matches.select(
    F.col("a.row_id").alias("src"),
    F.col("b.row_id").alias("dst")
)

g = GraphFrame(
    events_all.select(F.col("row_id").alias("id")),
    edges
)
components = g.connectedComponents()

event_clusters = components.select(
    F.col("id").alias("row_id"),
    F.col("component").alias("cluster_id")
)

events_all = events_all.join(event_clusters, on="row_id", how="left")

swim_events = swim_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
track_events = track_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
events = events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")

events_canon = events_all.select(
    "row_id",
    "event_id",
    "olympics_id",
    "event_name",
    "event_type",
    "distance",
    "gender",
    "discipline",
    "round",
    "cluster_id"
)

event_lookup = events_canon.groupBy("event_id") \
    .agg(F.first("cluster_id") \
         .alias("cluster_id")
)

################################
################################

# ATHLETE EVENT

################################
################################


athlete_events = spark.read.csv("./datasets/Scraped Data/athlete_event.csv", header=True, inferSchema=True)
swim_combined = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_combined.csv", header=True, inferSchema=True)
track_combined = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_combined.csv", header=True, inferSchema=True)


# DATASET LABELS FOR ATHLETE EVENTS
swim_combined = swim_combined.withColumn("source", F.lit("swim_combined"))
track_combined = track_combined.withColumn("source", F.lit("track_combined"))
athlete_events = athlete_events.withColumn("source", F.lit("base_athlete_events"))

# BIG UNIFIED TABLE
athlete_events_all = athlete_events.unionByName(swim_combined.unionByName(track_combined))

def parse_time(t):
    try:
        if t is None or t == "":
            return None
        t_str = str(t)
        if ":" in t_str:
            parts = t_str.split(":")
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        else:
            return float(t_str)
    except Exception:
        return None

time_udf = F.udf(parse_time, DoubleType())

athlete_events_all = athlete_events_all.withColumn("time_seconds", time_udf("time"))

# athlete_events_all = athlete_events_all.withColumn(
#     "result",
#     F.coalesce(F.col("result"), F.lit(-1))
# )

athlete_events_all = athlete_events_all.join(
    event_lookup,
    on="event_id",
    how="left"
)

athlete_events_canon = athlete_events_all.groupBy(
    "cluster_id", "time_seconds", "result"
).agg(
    F.collect_list("athlete_id").alias("athletes")
)




################################
################################

# CANDIDATES

################################
################################

candidates = spark.read.csv("./datasets/Scraped Data/athletes.csv", header=True, inferSchema=True)
swim_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_candidates.csv", header=True, inferSchema=True)
track_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_candidates.csv", header=True, inferSchema=True)


# DATASET LABELS FOR CANDIDATES
swim_candidates = swim_candidates.withColumn("source", F.lit("swim_candidates"))
track_candidates = track_candidates.withColumn("source", F.lit("track_candidates"))
candidates = candidates.withColumn("source", F.lit("candidates"))

# BIG UNIFIED TABLE
candidates_all = candidates.unionByName(swim_candidates.unionByName(track_candidates))


# ==========================
# CLEAN RELAYS FOR SWIMMING
# ==========================

relay_rows = candidates_all.filter(
    F.col("middle_name").contains(",")
)

split_names = relay_rows.withColumn(
    "name_list",
    F.split(F.col("middle_name"), ",")
)

exploded = split_names.withColumn(
    "middle_name",
    F.explode("name_list")
)

exploded = exploded.withColumn(
    "middle_name",
    F.trim("middle_name")
)

exploded = exploded.withColumn(
    "name_sections",
    F.split("middle_name", " ")
)

exploded = exploded.withColumn(
    "first_name",
    F.col("name_sections").getItem(0)
).withColumn(
    "last_name",
    F.element_at("name_sections", - 1) # Parses for middle names in list
)

exploded = exploded.drop("name_list", "name_sections")

clean_candidates = candidates_all.filter(
    ~F.col("middle_name").contains(",")
)

candidates_all = clean_candidates.unionByName(exploded)
# Row id
#candidates_all = candidates_all.withColumn("row_id", F.monotonically_increasing_id())


candidates_all = candidates_all.withColumn(
    "first_name", F.lower(F.trim("first_name"))
).withColumn(
    "last_name", F.lower(F.trim("last_name"))
).withColumn(
    "middle_name", F.lower(F.trim("middle_name"))
)

# Do a quick substring of athlete fnames for blocking later
candidates_all = candidates_all.withColumn(
    "first_initial",
    F.substring("first_name", 1, 1)
)

candidates_all = candidates_all.withColumn(
    "last_four",
    F.substring("last_name", 1, 4)
)


# Tokenize names
candidates_all = candidates_all.withColumn(
    "name_tokens",
    F.array_distinct(
        F.split(
            F.regexp_replace(
                F.concat_ws(" ", "first_name", "middle_name", "last_name"),
                r"[^\w\s]",
                ""
            ),
            " "
        )
    )
)

# Create blocking key
candidates_all = candidates_all.withColumn(
    "block_key",
    F.concat_ws(
        "_",
        F.col("first_initial"),
        F.col("last_name"),
        F.col("last_four"),
        F.coalesce(F.col("nationality_id"), F.lit("UNK"))
    )
)

# Extract distinct athlete_ids from athlete_events that are not in candidates_all
missing_athlete_ids = athlete_events.select("athlete_id").distinct() \
    .join(candidates_all.select("athlete_id").distinct(), "athlete_id", "left_anti")

# Create minimal candidate rows for them (you'll need to fill required columns)
missing_candidates = missing_athlete_ids.select(
    F.col("athlete_id"),
    F.lit("").alias("first_name"),
    F.lit("").alias("last_name"),
    F.lit("").alias("middle_name"),
    F.lit(None).cast(IntegerType()).alias("year_of_birth"),
    F.lit("").alias("nationality_id"),
    F.lit("missing_from_candidates").alias("source")
)

candidates_all = candidates_all.unionByName(missing_candidates, allowMissingColumns=True)
# Row ID after
candidates_all = candidates_all.withColumn("row_id", F.monotonically_increasing_id())

# Self join on block key
pairs = candidates_all.alias("a").join(
    candidates_all.alias("b"),
    on="block_key"
).filter(F.col("a.row_id") < F.col("b.row_id"))

# Sim features with easy matching
pairs = pairs.withColumn(
    "first_match",
    F.when(F.col("a.first_name") == F.col("b.first_name"), 1).otherwise(0)
)

pairs = pairs.withColumn(
    "last_match",
    F.when(F.col("a.last_name") == F.col("b.last_name"), 1).otherwise(0)
)

pairs = pairs.withColumn(
    "yob_match",
    F.when(F.col("a.year_of_birth") == F.col("b.year_of_birth"), 1).otherwise(0)
)

# Jaccard sim with name tokens
pairs = pairs.withColumn(
    "token_sim",
    jaccard(F.col("a.name_tokens"), F.col("b.name_tokens"))
)


# TODO : optional hard filter on last names
# pairs = pairs.filter(
#     (F.col("a.last_name") == F.col("b.last_name"))
# )

pairs = pairs.filter(
    (F.col("token_sim") > 0.3) | 
    (F.col("a.last_name") == F.col("b.last_name"))
)

pairs = pairs.withColumn(
    "score",
    0.4 * F.col("last_match") +
    0.3 * F.col("first_match") +
    0.2 * F.col("yob_match") +
    0.1 * F.col("token_sim")
)


# Cluster matches again
matches = pairs.filter(F.col("score") >= 0.8)

edges = matches.select(
    F.col("a.row_id").alias("src"),
    F.col("b.row_id").alias("dst")
)

g = GraphFrame(
    candidates_all.select(F.col("row_id").alias("id")),
    edges
)

components = g.connectedComponents()

# Build canon for candidates
candidates_canon = candidates_all.join(
    components.select(
        F.col("id").alias("row_id"),
        F.col("component").alias("cluster_id")
    ),
    on="row_id",
    how="left"
)

# Connect a_events and candidates
athlete_lookup = candidates_canon.select(
    "athlete_id",
    "cluster_id"
).dropDuplicates(["athlete_id"]) \
.withColumnRenamed("cluster_id", "athlete_cluster_id")


print(f"Distinct athlete_id in athlete_events_all: {athlete_events_all.select('athlete_id').distinct().count()}")
print(f"Distinct athlete_id in athlete_lookup: {athlete_lookup.select('athlete_id').distinct().count()}")

# Show a few athlete_id from each that are missing in the other
athlete_ids_events = athlete_events_all.select('athlete_id').distinct()
athlete_ids_lookup = athlete_lookup.select('athlete_id').distinct()

missing_in_lookup = athlete_ids_events.join(athlete_ids_lookup, 'athlete_id', 'left_anti')
missing_in_events = athlete_ids_lookup.join(athlete_ids_events, 'athlete_id', 'left_anti')

print(f"Athlete IDs in events but NOT in lookup: {missing_in_lookup.count()}")
missing_in_lookup.show(10, truncate=False)

print(f"Athlete IDs in lookup but NOT in events: {missing_in_events.count()}")
missing_in_events.show(10, truncate=False)


athlete_events_all = athlete_events_all.join(
    athlete_lookup,
    on="athlete_id",
    how="left"
)

# athlete_events_all = athlete_events_all.withColumn(
#     "athlete_cluster_id",
#     F.coalesce(
#         F.col("athlete_cluster_id"),
#         F.col("athlete_id") + 10000000   # dont collide with real id
#     )
# )


################################
################################

# FINAL ATHLETE

################################
################################

athletes_final = candidates_canon.groupBy("cluster_id").agg(
    F.first("first_name").alias("first_name"),
    F.first("last_name").alias("last_name"),
    F.first("nationality_id").alias("nationality_id"),
    F.first("year_of_birth").alias("year_of_birth")
)

################################
################################

# FINAL EVENTS

################################
################################

events_final = events_canon.groupBy("cluster_id").agg(
    F.first("event_name", ignorenulls=True).alias("event_name"),
    F.first("event_type", ignorenulls=True).alias("event_type"),
    F.first("distance", ignorenulls=True).alias("distance"),
    F.first("gender", ignorenulls=True).alias("gender"),
    F.first("discipline", ignorenulls=True).alias("discipline")
)

################################
################################

# FINAL RESULTS

################################
################################

results_final = athlete_events_all.select(
    "athlete_cluster_id",
    "cluster_id",
    "time_seconds",
    "result"
)

# Drop extra id columns.
# TODO: maybe come back and fix this / comment out
# results_final = results_final.dropDuplicates([
#     "athlete_cluster_id",
#     "cluster_id",
#     ##"time_seconds"
# ])


print("20 of results_final:")
results_final.show(20, truncate=False)

# Show 20 rows of events_final
print("20 of events_final:")
events_final.show(20, truncate=False)

# Show 20 rows of athletes_final
print("20 of athletes_final:")
athletes_final.show(20, truncate=False)


print(f"athlete events count: {athlete_events_all.count()}")
print(f"athletes final count: {athletes_final.count()}")
print(f"events final count: {events_final.count()}")
print(f"final results count: {results_final.count()}")
print(f"athlete cluster id distinct count: {athlete_events_all.select('athlete_cluster_id').distinct().count()}")

exit(0)