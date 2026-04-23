from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.functions import levenshtein, greatest, length
from pyspark.sql.window import Window
from graphframes import GraphFrame
import re
import sys
import os
import shutil
import tempfile

#############################

# SETUP

#############################

# LINUX ONLY.
# Using wsl

# Must use these specific versions of python, pyspark, and graphframes
# Requirements
# Python 3.11
# PySpark 3.5.3
# GraphFrames 0.8.3-spark3.5-s_2.12

# Installation with linux.
#     python3 -m venv venv
#     source venv/bin/activate
#     pip install pyspark==3.5.3 graphframes==0.8.3

# sudo apt update && sudo apt install python3-pip openjdk-11-jdk
# pip3 install pyspark==3.5.3 graphframes
# sudo apt install python3-venv python3-full -y

# NAVIGATE TO PROJECT FOLDER
# cd /mnt/c/...
# 
#############################

# SETUP END

#############################


spark = SparkSession.builder \
    .appName("olympic_joins") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .config("spark.local.dir", "/home/logan/tmp") \
    .config("spark.driver.memory", "10g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setCheckpointDir("file:///mnt/c/temp/spark-checkpoint")

################################
################################

# NORMALIZE COUNTRIES

################################
################################

countries = spark.read.csv("./datasets/Scraped Data/countries.csv", header=True, inferSchema=True)
city_country = spark.read.csv("./datasets/Scraped Data/city_country.csv", header=True, inferSchema=True)

country_code_map = countries.groupBy("id").agg(
    F.first("country_id").alias("country_code")
)


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


# Regex clean for garbage values

events_all = events_all.withColumn(
    "event_name_clean",
    F.lower(F.regexp_replace("event_name", r"&nbsp;|&ndash;|&mdash;|–|—", " "))
)
events_all = events_all.withColumn(
    "event_name_clean",
    F.regexp_replace("event_name_clean", r"\s+", " ")
)
events_all = events_all.withColumn(
    "event_name_clean",
    F.regexp_replace("event_name_clean", r"(\d+)\s*(metre|meter)s?\b", "$1m")
)

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


################################
################################

# EVENTS

################################
################################

events_all = events_all.withColumn(
    "parsed",
    parse_udf("event_name_clean", "event_type")
)

events_all = events_all.select(
    "*",
    "parsed.distance",
    "parsed.gender",
    "parsed.discipline",
    "parsed.round"
).drop("parsed")

#events_all.select("event_name", "distance", "gender", "discipline", "round").show(5, truncate=False)

# Catch any crappy disciplines with tokens
events_all = events_all.withColumn(
    "tokens",
    F.array_except(
        F.split(F.lower(F.regexp_replace("event_name_clean", r"[^\w\s]", "")), " "),
        F.array(F.lit("mens"), F.lit("womens"), F.lit("heat"), F.lit("final"))
    )
)

# events_all = events_all.filter(
#     ~F.col("event_name_clean").rlike("summary|results summary")
# )

pairs = events_all.alias("a").join(
    events_all.alias("b"),
    on=["olympics_id", "event_type"],
).filter(F.col("a.row_id") < F.col("b.row_id")) \
.filter(F.col("a.distance") == F.col("b.distance"))

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

matches = pairs.filter(F.col("score") >= 0.825)

vertices = events_all.select(F.col("row_id").alias("id"))

edges = matches.select(
    F.col("a.row_id").alias("src"),
    F.col("b.row_id").alias("dst")
)

g = GraphFrame(
    events_all.select(F.col("row_id").alias("id")),
    edges
)
components = g.connectedComponents(checkpointInterval=2)

event_clusters = components.select(
    F.col("id").alias("row_id"),
    F.col("component").alias("cluster_id")
)


events_all = events_all.join(event_clusters, on="row_id", how="left")


event_lookup_raw = events_all.groupBy(
    "event_id") \
    .agg(
        F.first("cluster_id").alias("event_cluster_id"))

swim_events = swim_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
track_events = track_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
events = events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")



################################
################################

# OLYMPICS

################################
################################

olympics = spark.read.csv(
    "./datasets/Scraped Data/olympics.csv", header=True, inferSchema=True)
swim_olympics = spark.read.csv(
    "./datasets/Smaller Tables with Dummy Ids/swim_olympics.csv", header=True, inferSchema=True)
track_olympics = spark.read.csv(
    "./datasets/Smaller Tables with Dummy Ids/track_olympics.csv", header=True, inferSchema=True)

olympics = olympics.withColumn("source", F.lit("base_olympics"))
swim_olympics = swim_olympics.withColumn("source", F.lit("swim_olympics"))
track_olympics = track_olympics.withColumn("source", F.lit("track_olympics"))

olympics_all = olympics.unionByName(swim_olympics).unionByName(track_olympics)
olympics_all = olympics_all.withColumn("row_id", F.monotonically_increasing_id())

olympics_all = olympics_all.withColumn("host_city_norm", F.lower(F.trim("host_city_name")))



olympics_all = olympics_all.withColumn(
    "block_key",
    F.concat_ws("_", F.col("year"), F.col("summer"), F.substring("host_city_norm", 1, 5))
)

pairs = olympics_all.alias("a").join(
    olympics_all.alias("b"),
    on="block_key"
).filter(F.col("a.row_id") < F.col("b.row_id"))

pairs = pairs.withColumn(
    "lev_dist",
    levenshtein(F.col("a.host_city_norm"), F.col("b.host_city_norm"))
).withColumn(
    "max_len",
    greatest(length("a.host_city_norm"), length("b.host_city_norm"))
).withColumn(
    "city_sim",
    F.when(F.col("max_len") == 0, 1.0) \
    .otherwise(1.0 - (F.col("lev_dist") / F.col("max_len")))
)

pairs = pairs.withColumn("score", F.col("city_sim"))


edges = pairs.filter(F.col("score") >= 0.75).select(
    F.col("a.row_id").alias("src"),
    F.col("b.row_id").alias("dst")
)


g = GraphFrame(olympics_all.select(F.col("row_id").alias("id")), edges)
components = g.connectedComponents(checkpointInterval=2)

olympics_clusters = olympics_all.join(
    components.select(F.col("id").alias("row_id"), F.col("component").alias("olympics_cluster_id")),
    on="row_id",
    how="left"
)

window_oly = Window.orderBy("olympics_cluster_id")
olympics_clusters = olympics_clusters.withColumn(
    "canonical_olympics_id",
    F.row_number().over(window_oly)
)

olympics_lookup = olympics_clusters.select("olympics_id", "canonical_olympics_id").distinct()

events_all = events_all.join(olympics_lookup, on="olympics_id", how="left")



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

athlete_events_all = athlete_events_all.join(
    event_lookup_raw,
    on="event_id",
    how="left"
)

################################
################################

# CANDIDATES

################################
################################

candidates = spark.read.csv("./datasets/Scraped Data/athletes.csv", header=True, inferSchema=True)
swim_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_candidates.csv", header=True, inferSchema=True)
track_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_candidates.csv", header=True, inferSchema=True)

candidates = candidates.join(
    country_code_map,
    candidates["nationality_id"] == country_code_map["id"],
    "left"
).withColumn("country_code", F.col("country_code")).drop("id")

# Remap nation_id to country code
swim_candidates = swim_candidates.withColumn("country_code", F.col("nationality_id"))
track_candidates = track_candidates.withColumn("country_code", F.col("nationality_id"))


# DATASET LABELS FOR CANDIDATES
swim_candidates = swim_candidates.withColumn("source", F.lit("swim_candidates"))
track_candidates = track_candidates.withColumn("source", F.lit("track_candidates"))
candidates = candidates.withColumn("source", F.lit("candidates"))

# BIG UNIFIED TABLE
candidates_all = candidates.unionByName(swim_candidates.unionByName(track_candidates))

# 1. Only process rows where middle_name contains commas and looks like a relay list
def is_valid_relay_list(mid):
    if mid is None: return False
    parts = [p.strip() for p in mid.split(",")]
    if any(p == "" for p in parts):      # dont use trailing/leading commas
        return False
    if len(parts) < 2: return False
    middle_parts = parts[1:-1] if len(parts) > 2 else []
    return all(len(p.split()) >= 2 for p in middle_parts if p)

is_valid_udf = F.udf(is_valid_relay_list, BooleanType())

candidates_all = candidates_all.withColumn("relay_list", is_valid_udf("middle_name"))
relay_df = candidates_all.filter(F.col("relay_list"))
relay_df = relay_df.filter(F.col("middle_name").isNotNull() & (F.col("middle_name") != ""))
non_relay_df = candidates_all.filter(~F.col("relay_list"))

"""
Assumes first athlete = (first, first word of middle)
            last athlete  = (last word of middle, last)
            middle athletes = each comma-separated segment between first and last
"""
def extract_athletes(first, middle, last):
    if not middle:
        return []
    parts = [p.strip() for p in middle.split(",")]
    athletes = []
    # First athlete
    first_last = parts[0].split()[0] if parts[0] else ""
    athletes.append((first, first_last))
    # Middle athletes
    for part in parts[1:-1]:
        tokens = part.split()
        if len(tokens) >= 2:
            athletes.append((tokens[0], tokens[-1]))   # first and last word
        else:
            athletes.append((tokens[0], ""))
    # Last athlete
    last_first = parts[-1].split()[-1] if parts[-1] else ""
    athletes.append((last_first, last))
    return athletes

extract_athletes_udf = F.udf(extract_athletes, ArrayType(StructType([
    StructField("first", StringType(), True),
    StructField("last", StringType(), True)
])))

relay_processed = relay_df.withColumn(
    "athlete_array",
    extract_athletes_udf("first_name", "middle_name", "last_name")
)

relay_exploded = relay_processed.withColumn("athlete", F.explode("athlete_array"))

relay_exploded = relay_exploded.withColumn("first_name", F.col("athlete.first")) \
                               .withColumn("last_name", F.col("athlete.last")) \
                               .drop("athlete", "athlete_array", "relay_list")


non_relay_df = non_relay_df.drop("relay_list")

# TODO: show exploded relays
#relay_exploded.select("first_name","last_name","middle_name","country_code").show(20, truncate=False)

candidates_all = non_relay_df.unionByName(relay_exploded)

# Get rid of any commas or whitespace
candidates_all = candidates_all.withColumn(
    "middle_name",
    F.regexp_replace(F.col("middle_name"), r",\s*$", "")
)

candidates_all = candidates_all.withColumn(
    "first_name", F.lower(F.trim("first_name"))
).withColumn(
    "last_name", F.lower(F.trim("last_name"))
).withColumn(
    "middle_name", F.lower(F.trim("middle_name"))
)

candidates_all = candidates_all.withColumn("first_name", F.regexp_replace("first_name", '["“”]', ''))
candidates_all = candidates_all.withColumn("last_name", F.regexp_replace("last_name", '["“”]', ''))

candidates_all = candidates_all.withColumn(
    "middle_name",
    F.when(
        F.col("middle_name").contains(F.col("first_name")) |
        F.col("middle_name").contains(F.col("last_name")),
        F.lit(None)
    ).otherwise(F.col("middle_name"))
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

# Coalesce bad decade blocks of yob's 
candidates_all = candidates_all.withColumn(
    "birth_decade", 
    F.when(F.col("year_of_birth").isNotNull(), 
           F.floor(F.col("year_of_birth") / 10) * 10)
     .otherwise(F.lit(0))
)

candidates_all = candidates_all.filter(
    F.col("first_name").rlike("^[a-zA-ZÀ-ÖØ-öø-ÿ'.-]+$") &   # accented / cool letters allowed
    (F.length("first_name") >= 2) &
    ~F.lower(F.col("first_name")).rlike("record|progression|kilometre|metre|world|summary|qualif|heat|final")
)

# Suffix swap
suffix_list = ["jr", "sr", "ii", "iii", "iv", "v", "jr.", "sr."]

# If last name is a suffix and middle name is not empty, swap them
candidates_all = candidates_all.withColumn(
    "temp_last",
    F.when(
        F.lower(F.col("last_name")).isin(suffix_list) & (F.length(F.col("middle_name")) > 0),
        F.col("middle_name")
    ).otherwise(F.col("last_name"))
).withColumn(
    "temp_middle",
    F.when(
        F.lower(F.col("last_name")).isin(suffix_list) & (F.length(F.col("middle_name")) > 0),
        F.col("last_name")                      # the suffix becomes the new middle name
    ).otherwise(F.col("middle_name"))
).withColumn("last_name", F.col("temp_last")) \
 .withColumn("middle_name", F.col("temp_middle")) \
 .drop("temp_last", "temp_middle")

# Remove trailing suffix from last_name
candidates_all = candidates_all.withColumn(
    "last_name",
    F.regexp_replace(F.col("last_name"), r" (jr\.?|sr\.?|ii|iii|iv|v)$", "")
)

# Drop rows where first or last name is purely a suffix
candidates_all = candidates_all.filter(
    ~F.lower(F.col("first_name")).isin(suffix_list) &
    ~F.lower(F.col("last_name")).isin(suffix_list)
)


candidates_all = candidates_all.withColumn(
    "gender",
    F.when(F.lower(F.col("gender")).startswith("m"), "M")
     .when(F.lower(F.col("gender")).startswith("w"), "F")
     .when(F.lower(F.col("gender")).startswith("f"), "F")
     .otherwise(F.col("gender"))
)

# Create blocking key
candidates_all = candidates_all.withColumn(
    "block_key",
    F.concat_ws(
        "_",
        F.col("first_initial"),
        F.col("last_four"),
        F.col("birth_decade"),
        F.col("gender"),
        F.coalesce(F.col("country_code"), F.lit("UNK")) # Changed to country code instead of nationality id
    )
)

candidates_all = candidates_all.withColumn(
    "block_key_no_decade",
    F.concat_ws("_",
        F.col("first_initial"),
        F.col("last_four"),
        F.col("gender"),
        F.coalesce(F.col("country_code"), F.lit("UNK"))
    )
)

# Extract distinct athlete_ids from athlete_events that are not in candidates_all
missing_athlete_ids = athlete_events_all.select("athlete_id").distinct() \
    .join(candidates_all.select("athlete_id").distinct(), "athlete_id", "left_anti")

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

candidates_all = candidates_all.withColumn("row_id", F.monotonically_increasing_id())

# Constantly going OOM, setting checkpoint
candidates_all = candidates_all.checkpoint()

# Self join on block key
pairs = candidates_all.alias("a").join(
    candidates_all.alias("b"),
    on="block_key"
).filter(F.col("a.row_id") < F.col("b.row_id")) \
 .select(
    F.col("a.row_id").alias("row_id_a"),
    F.col("b.row_id").alias("row_id_b"),
    F.col("a.first_name").alias("first_a"),
    F.col("b.first_name").alias("first_b"),
    F.col("a.last_name").alias("last_a"),
    F.col("b.last_name").alias("last_b"),
    F.col("a.year_of_birth").alias("yob_a"),
    F.col("b.year_of_birth").alias("yob_b"),
    F.col("a.name_tokens").alias("tokens_a"),
    F.col("b.name_tokens").alias("tokens_b")
)

# Add similarity features
pairs = pairs.withColumn("first_match", F.when(F.col("first_a") == F.col("first_b"), 1).otherwise(0))
pairs = pairs.withColumn("last_match", F.when(F.col("last_a") == F.col("last_b"), 1).otherwise(0))
pairs = pairs.withColumn("yob_match", F.when(F.col("yob_a") == F.col("yob_b"), 1).otherwise(0))
pairs = pairs.withColumn("token_sim", jaccard(F.col("tokens_a"), F.col("tokens_b")))

pairs = pairs.withColumn("fullname_a", F.concat_ws(" ", F.col("first_a"), F.col("last_a")))
pairs = pairs.withColumn("fullname_b", F.concat_ws(" ", F.col("first_b"), F.col("last_b")))
pairs = pairs.withColumn("lev_dist", levenshtein(F.col("fullname_a"), F.col("fullname_b")))
pairs = pairs.withColumn("max_len", greatest(F.length("fullname_a"), F.length("fullname_b")))
pairs = pairs.withColumn("name_sim",
    F.when(F.col("max_len") == 0, 1.0)
     .otherwise(1.0 - (F.col("lev_dist") / F.col("max_len")))
)

pairs = pairs.withColumn(
    "score",
    0.25 * F.col("last_match") +
    0.15 * F.col("first_match") +
    0.15 * F.col("yob_match") +
    0.15 * F.col("token_sim") +
    0.30 * F.col("name_sim")
)

# New block key for athletes that dont have yob listed. Michael phelps had this issue for some reason.
pairs2 = candidates_all.alias("a").join(
    candidates_all.alias("b"),
    on="block_key_no_decade"
).filter(F.col("a.row_id") < F.col("b.row_id")) \
 .select(
    F.col("a.row_id").alias("row_id_a"),
    F.col("b.row_id").alias("row_id_b"),
    F.col("a.first_name").alias("first_a"),
    F.col("b.first_name").alias("first_b"),
    F.col("a.last_name").alias("last_a"),
    F.col("b.last_name").alias("last_b"),
    F.col("a.year_of_birth").alias("yob_a"),
    F.col("b.year_of_birth").alias("yob_b"),
    F.col("a.name_tokens").alias("tokens_a"),
    F.col("b.name_tokens").alias("tokens_b")
)

# Same similarity columns
pairs2 = pairs2.withColumn("first_match", F.when(F.col("first_a") == F.col("first_b"), 1).otherwise(0))
pairs2 = pairs2.withColumn("last_match", F.when(F.col("last_a") == F.col("last_b"), 1).otherwise(0))
pairs2 = pairs2.withColumn("yob_match", F.when(F.col("yob_a") == F.col("yob_b"), 1).otherwise(0))
pairs2 = pairs2.withColumn("token_sim", jaccard(F.col("tokens_a"), F.col("tokens_b")))

# Levy again
pairs2 = pairs2.withColumn("fullname_a", F.concat_ws(" ", F.col("first_a"), F.col("last_a")))
pairs2 = pairs2.withColumn("fullname_b", F.concat_ws(" ", F.col("first_b"), F.col("last_b")))
pairs2 = pairs2.withColumn("lev_dist", levenshtein(F.col("fullname_a"), F.col("fullname_b")))
pairs2 = pairs2.withColumn("max_len", greatest(F.length("fullname_a"), F.length("fullname_b")))
pairs2 = pairs2.withColumn("name_sim",
    F.when(F.col("max_len") == 0, 1.0)
     .otherwise(1.0 - (F.col("lev_dist") / F.col("max_len")))
)

pairs2 = pairs2.withColumn(
    "score",
    0.25 * F.col("last_match") +
    0.15 * F.col("first_match") +
    0.15 * F.col("yob_match") +
    0.15 * F.col("token_sim") +
    0.30 * F.col("name_sim")
)

# Union the two pairs
pairs = pairs.unionByName(pairs2)


# Cluster matches again
matches = pairs.filter(F.col("score") >= 0.825)

edges = matches.select(
    F.col("row_id_a").alias("src"),
    F.col("row_id_b").alias("dst")
)

g = GraphFrame(
    candidates_all.select(F.col("row_id").alias("id")),
    edges
)

components = g.connectedComponents(checkpointInterval=2)

# Build canon for candidates
candidates_canon = candidates_all.join(
    components.select(
        F.col("id").alias("row_id"),
        F.col("component").alias("cluster_id")
    ),
    on="row_id",
    how="left"
)

# Drop empty athletes
candidates_canon = candidates_canon.filter(
    (F.col("first_name") != "") | (F.col("last_name") != "")
)

# Help normalize athlete ID's to keep them at low numbers
window_spec = Window.orderBy("cluster_id")
candidates_canon = candidates_canon.withColumn(
    "canonical_athlete_id",
    F.row_number().over(window_spec)
)

# Create medal map for candidates
medal_map = candidates_canon.select("athlete_id", F.col("canonical_athlete_id").alias("athlete_cluster_id")).distinct()

events_all = events_all.join(
    medal_map.withColumnRenamed("athlete_id", "gold_id").withColumnRenamed("athlete_cluster_id", "gold_cluster_id"),
    on="gold_id",
    how="left"
).join(
    medal_map.withColumnRenamed("athlete_id", "silver_id").withColumnRenamed("athlete_cluster_id", "silver_cluster_id"),
    on="silver_id",
    how="left"
).join(
    medal_map.withColumnRenamed("athlete_id", "bronze_id").withColumnRenamed("athlete_cluster_id", "bronze_cluster_id"),
    on="bronze_id",
    how="left"
)

events_canon = events_all.select(
    "row_id",
    "event_id",
    "olympics_id",
    "canonical_olympics_id",
    "event_name",
    "event_type",
    "distance",
    "gender",
    "discipline",
    "round",
    "cluster_id",
    "gold_cluster_id",
    "silver_cluster_id",
    "bronze_cluster_id"
)

# Window and partition to help keep id's readable
window_event = Window.orderBy("cluster_id")
events_canon = events_canon.withColumn(
    "canonical_event_id",
    F.row_number().over(window_event)
)

event_id_mapping = events_canon.select("cluster_id", "canonical_event_id").distinct()

athlete_events_all = athlete_events_all.join(event_id_mapping,
    athlete_events_all["event_cluster_id"] == event_id_mapping["cluster_id"], "left") \
    .drop("cluster_id", "event_cluster_id") \
    .withColumnRenamed("canonical_event_id", "event_cluster_id")

athlete_lookup = candidates_canon.select(
    "athlete_id",
    "canonical_athlete_id") \
    .dropDuplicates(["athlete_id"]) \
    .withColumnRenamed("canonical_athlete_id",
                       "athlete_cluster_id"
    )


athlete_events_all = athlete_events_all.join(
    athlete_lookup,
    on="athlete_id",
    how="left")

################################
################################

# FINAL ATHLETE

################################
################################

athletes_final = candidates_canon.groupBy("canonical_athlete_id").agg(
    F.first("first_name").alias("first_name"),
    F.first("last_name").alias("last_name"),
    F.first("country_code").alias("country_code"), # Change to country code from nationality id
    F.first("year_of_birth").alias("year_of_birth")
)

athletes_final = athletes_final.groupBy(
    "first_name", "last_name", "country_code", "gender"
).agg(
    F.min("canonical_athlete_id").alias("athlete_id"),
    F.first("year_of_birth", ignorenulls=True).alias("year_of_birth")
)

################################
################################

# FINAL EVENTS

################################
################################

events_final = events_canon.groupBy("canonical_event_id").agg(
    F.first("event_name", ignorenulls=True).alias("event_name"),
    F.first("event_type", ignorenulls=True).alias("event_type"),
    F.first("distance", ignorenulls=True).alias("distance"),
    F.first("gender", ignorenulls=True).alias("gender"),
    F.first("discipline", ignorenulls=True).alias("discipline"),
    F.first("canonical_olympics_id", ignorenulls=True).alias("olympics_id"),
    F.first("gold_cluster_id", ignorenulls=True).alias("gold_id"),
    F.first("silver_cluster_id", ignorenulls=True).alias("silver_id"),
    F.first("bronze_cluster_id", ignorenulls=True).alias("bronze_id")
).withColumnRenamed("canonical_event_id", "event_id")

################################
################################

# FINAL RESULTS

################################
################################

results_final = athlete_events_all.select(
    "athlete_cluster_id",
    "event_cluster_id",
    "time_seconds",
    "result"
)


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






##############################################################################
##############################################################################
##############################################################################

######################## SCHEMA ALIGNED CREATION #############################

##############################################################################
##############################################################################
##############################################################################

events_clean = events_canon.select(
    F.col("canonical_event_id").alias("event_id"),
    F.col("gold_cluster_id").alias("gold_id"),
    F.col("silver_cluster_id").alias("silver_id"),
    F.col("bronze_cluster_id").alias("bronze_id"),
    F.col("canonical_olympics_id").alias("olympics_id"),
    "event_name",
    "event_type"
).distinct()

events_clean = events_clean.groupBy(
    "olympics_id", "event_name", "event_type"
).agg(
    F.min("event_id").alias("event_id_old"),   # keep track for remapping
    F.first("gold_id", ignorenulls=True).alias("gold_id"),
    F.first("silver_id", ignorenulls=True).alias("silver_id"),
    F.first("bronze_id", ignorenulls=True).alias("bronze_id")
)

# # Create a mapping from old canonical event IDs to the new dedup ID
# event_remap = events_clean.select(
#     F.col("event_id_old").alias("old_event_id"),
#     F.monotonically_increasing_id().alias("new_event_id")
# )

# athlete_event_clean = athlete_event_clean.alias("aec") \
#     .join(event_remap, F.col("aec.event_id") == F.col("old_event_id"), "left") \
#     .select(
#         F.col("athlete_id"),
#         F.col("new_event_id").alias("event_id"),
#         "time", "result"
#     )

# # Remap results_final event IDs
# results_final = results_final.alias("rf") \
#     .join(event_remap, F.col("rf.event_cluster_id") == F.col("old_event_id"), "left") \
#     .select(
#         "athlete_cluster_id",
#         F.col("new_event_id").alias("event_cluster_id"),
#         "time_seconds", "result"
#     )

events_clean = events_clean.select(
    F.col("event_id_old").alias("event_id"),
    "gold_id", "silver_id", "bronze_id",
    "olympics_id", "event_name", "event_type"
)


athlete_event_clean = athlete_events_all.select(
    F.col("athlete_cluster_id").alias("athlete_id"),
    F.col("event_cluster_id").alias("event_id"),
    "time",
    "result"
)



# athletes_clean = candidates_canon.select(
#     F.col("canonical_athlete_id").alias("athlete_id"),
#     "first_name",
#     "middle_name",
#     "last_name",
#     "height(cm)",
#     "weight(kg)",
#     F.col("country_code").alias("nationality_id"),
#     "year_of_birth",
#     "nationality_id_at_birth",
#     "gender"
# ).dropDuplicates([
#     "athlete_id", 
#     "nationality_id", 
#     "year_of_birth", 
#     "gender"]) \
# .distinct()

# athletes_clean = candidates_canon.groupBy("canonical_athlete_id").agg(
#     F.first("first_name", ignorenulls=True).alias("first_name"),
#     F.first("middle_name", ignorenulls=True).alias("middle_name"),
#     F.first("last_name", ignorenulls=True).alias("last_name"),
#     F.first("height(cm)", ignorenulls=True).alias("height(cm)"),
#     F.first("weight(kg)", ignorenulls=True).alias("weight(kg)"),
#     F.first("country_code", ignorenulls=True).alias("nationality_id"),
#     F.first("year_of_birth", ignorenulls=True).alias("year_of_birth"),
#     F.first("nationality_id_at_birth", ignorenulls=True).alias("nationality_id_at_birth"),
#     F.first("gender", ignorenulls=True).alias("gender")
# ).withColumnRenamed("canonical_athlete_id", "athlete_id")


athletes_clean = candidates_canon.groupBy("canonical_athlete_id").agg(
    F.first("first_name", ignorenulls=True).alias("first_name"),
    F.first("middle_name", ignorenulls=True).alias("middle_name"),
    F.first("last_name", ignorenulls=True).alias("last_name"),
    F.first("height(cm)", ignorenulls=True).alias("height(cm)"),
    F.first("weight(kg)", ignorenulls=True).alias("weight(kg)"),
    F.first("country_code", ignorenulls=True).alias("nationality_id"),
    F.first("year_of_birth", ignorenulls=True).alias("year_of_birth"),
    F.first("nationality_id_at_birth", ignorenulls=True).alias("nationality_id_at_birth"),
    F.first("gender", ignorenulls=True).alias("gender")
)

athletes_clean = athletes_clean.groupBy(
    "first_name", "last_name", "nationality_id", "gender"
).agg(
    F.min("canonical_athlete_id").alias("athlete_id"),
    F.first("middle_name", ignorenulls=True).alias("middle_name"),
    F.first("height(cm)", ignorenulls=True).alias("height(cm)"),
    F.first("weight(kg)", ignorenulls=True).alias("weight(kg)"),
    F.first("year_of_birth", ignorenulls=True).alias("year_of_birth"),
    F.first("nationality_id_at_birth", ignorenulls=True).alias("nationality_id_at_birth")
)

olympics_clean = olympics_clusters.select(
    F.col("canonical_olympics_id").alias("olympics_id"),
    "year",
    "summer",
    "host_city_name"
).distinct()

olympics_clean = olympics_clean.groupBy("year", "summer", "host_city_name") \
    .agg(F.min("olympics_id").alias("olympics_id"))


# events_clean = events_clean.alias("ec").join(
#     olympics_clean.select(
#         F.col("olympics_id").alias("new_oly_id"),
#         "year", "summer", "host_city_name"
#     ),
#     on="host_city_name",   # or on year+summer+city
#     how="left"
# ).select(
#     "event_id", "gold_id", "silver_id", "bronze_id",
#     F.col("new_oly_id").alias("olympics_id"),
#     "event_name", "event_type"
# )

# # Remap events_final similarly, if used later
# events_final = events_final.alias("ef").join(
#     olympics_clean.select(F.col("olympics_id").alias("new_oly_id"), "year", "summer", "host_city_name"),
#     on="host_city_name",
#     how="left"
# ).select(
#     "event_id", "gold_id", "silver_id", "bronze_id",
#     F.col("new_oly_id").alias("olympics_id"),
#     "event_name", "event_type", "distance", "gender", "discipline"
# )


def write_single_csv(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Remove any existing file or directory at the target path
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    tmp_dir = tempfile.mkdtemp()
    df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")
    part_file = None
    for f in os.listdir(tmp_dir):
        if f.startswith("part-") and f.endswith(".csv"):
            part_file = os.path.join(tmp_dir, f)
            break
    if part_file:
        shutil.move(part_file, path)
        shutil.rmtree(tmp_dir)
    else:
        raise RuntimeError("No part file found in " + tmp_dir)


# Clean datasets
write_single_csv(events_clean, "./clean_datasets/events_clean.csv")
write_single_csv(athlete_event_clean, "./clean_datasets/athlete_event_clean.csv")
write_single_csv(athletes_clean, "./clean_datasets/athletes_clean.csv")
write_single_csv(olympics_clean, "./clean_datasets/olympics_clean.csv")

# Canon datasets
write_single_csv(athletes_final, "./canonical/athletes.csv")
write_single_csv(events_final, "./canonical/events.csv")
write_single_csv(results_final, "./canonical/results.csv")

print("All done!")
exit(0)
