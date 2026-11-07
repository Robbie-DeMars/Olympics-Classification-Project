"""
Olympic Data Integration – Entity Resolution Pipeline
-----------------------------------------------------
Authors:
    Robbie DeMars, u1411323
    Logan Hunter, u1342460
    Quang Nguyen u1470229

Reads multiple scraped Olympic CSVs, resolves duplicate entities using
GraphFrames connected components, and produces canonical clean datasets.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import levenshtein, greatest, length
from pyspark.sql.window import Window
from graphframes import GraphFrame
import re
import sys
import os
import shutil
import tempfile
import unicodedata

# =============================================================================
# SETUP – Works for fresh install on WSL / Linux.
# =============================================================================
#
# Requirement: Must run the program on Linux.
#
# 1. Install system packages (Java 11, Python 3.11, venv)
#    sudo add-apt-repository ppa:deadsnakes/ppa -y
#    sudo apt update
#    sudo apt install python3.11 python3.11-venv python3.11-dev openjdk-11-jdk -y
#
# 2. Create a virtual environment
#    python3.11 -m venv venv
#    source venv/bin/activate
#
# 2.5 Venv errors
#    If there are any errors while creating and running the venv,
#    you have to exit the venv and remove the environment completely
#    deactivate
#    rm -rf venv
#    Then you may attempt to create another virtual environment with step 2.
#
# 3. Install Python dependencies
#    pip install pyspark==3.5.3 graphframes-latest
#    pip install --upgrade pip
#
# 4.  Verify the dataset folders contain the following:
#    ./datasets/Scraped Data/   (original CSVs)
#    ./datasets/Smaller Tables with Dummy Ids/   (swim, track CSVs)
#    ./datasets/   (weight_height CSVs)
#
# 5. Run the script
#    python ER_pipeline.py
#
# 6. Output will appear in:
#    ./clean_datasets/   (schema‑aligned CSVs)
#    ./canonical/        (canonical tables)
# =============================================================================

# =============================================================================
# =============================================================================
# SPARK SETUP
# =============================================================================
# =============================================================================

spark = SparkSession.builder \
    .appName("olympic_joins") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .config("spark.local.dir", "/home/logan/tmp") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "400") # For OOM
spark.sparkContext.setLogLevel("ERROR") # Getting rid of the partition warnings to debug

spark.sparkContext.setCheckpointDir("file:///mnt/c/temp/spark-checkpoint")

# =============================================================================
# =============================================================================
# LOAD COUNTRY MAPPINGS
# =============================================================================
# =============================================================================

countries = spark.read.csv("./datasets/Scraped Data/countries.csv", header=True, inferSchema=True)
city_country = spark.read.csv("./datasets/Scraped Data/city_country.csv", header=True, inferSchema=True)

country_code_map = countries.groupBy("id").agg(
    F.first("country_id").alias("country_code")
)
# Create manual mappings from host cities to their respective nations.
full_to_code = spark.createDataFrame([
    ("Greece", "GRE"), ("France", "FRA"), ("United States", "USA"),
    ("United Kingdom", "GBR"), ("Sweden", "SWE"), ("Germany", "GER"),
    ("Belgium", "BEL"), ("Netherlands", "NED"), ("Switzerland", "SUI"),
    ("Italy", "ITA"), ("Norway", "NOR"), ("Japan", "JPN"),
    ("Finland", "FIN"), ("Australia", "AUS"), ("Canada", "CAN"),
    ("South Korea", "KOR"), ("China", "CHN"), ("Spain", "ESP"),
    ("Russia", "RUS"), ("Brazil", "BRA"), ("Mexico", "MEX"),
    ("Austria", "AUT"), ("Hungary", "HUN"), ("Argentina", "ARG"),
    ("Yugoslavia", "YUG"), ("Soviet Union", "URS"), ("Czech Republic", "CZE"),
    ("Poland", "POL"), ("Romania", "ROU"), ("Bulgaria", "BUL"),
    ("Turkey", "TUR"), ("Portugal", "POR"), ("Denmark", "DEN"),
    ("New Zealand", "NZL"), ("Croatia", "CRO"), ("Slovenia", "SLO"),
    ("Ukraine", "UKR"), ("Kazakhstan", "KAZ"), ("Estonia", "EST"),
    ("Latvia", "LAT"), ("Lithuania", "LTU"), ("Georgia", "GEO"),
    ("Azerbaijan", "AZE"), ("Belarus", "BLR"), ("Uzbekistan", "UZB"),
    ("Thailand", "THA"), ("Malaysia", "MAS"), ("Singapore", "SGP"),
    ("Philippines", "PHI"), ("India", "IND"), ("Pakistan", "PAK"),
    ("Iran", "IRI"), ("Iraq", "IRQ"), ("Kuwait", "KUW"),
    ("Qatar", "QAT"), ("Saudi Arabia", "KSA"), ("United Arab Emirates", "UAE"),
    ("Egypt", "EGY"), ("South Africa", "RSA"), ("Nigeria", "NGR"),
    ("Kenya", "KEN"), ("Ethiopia", "ETH"), ("Tanzania", "TAN"),
    ("Algeria", "ALG"), ("Morocco", "MAR"), ("Tunisia", "TUN"),
    ("Cote d'Ivoire", "CIV"), ("Cameroon", "CMR"), ("Ghana", "GHA"),
    ("Senegal", "SEN"), ("Zambia", "ZAM"), ("Zimbabwe", "ZIM"),
    ("Jamaica", "JAM"), ("Cuba", "CUB"), ("Dominican Republic", "DOM"),
    ("Puerto Rico", "PUR"), ("Trinidad and Tobago", "TTO"),
    ("Bahamas", "BAH"), ("Barbados", "BAR"), ("Grenada", "GRN"),
    ("Venezuela", "VEN"), ("Colombia", "COL"), ("Ecuador", "ECU"),
    ("Peru", "PER"), ("Chile", "CHI"), ("Uruguay", "URU"),
    ("Paraguay", "PAR"), ("Bolivia", "BOL"), ("Costa Rica", "CRC"),
    ("Guatemala", "GUA"), ("El Salvador", "ESA"), ("Honduras", "HON"),
    ("Nicaragua", "NCA"), ("Panama", "PAN"), ("Suriname", "SUR"),
    ("Guyana", "GUY"), ("Fiji", "FIJ"), ("Papua New Guinea", "PNG"),
    ("Samoa", "SAM"), ("Tonga", "TGA"), ("Vanuatu", "VAN"),
    ("Solomon Islands", "SOL"), ("Kiribati", "KIR"), ("Tuvalu", "TUV")
], ["full_name", "code"])

# =============================================================================
# =============================================================================
# 3. LOAD & UNION EVENT CSVs
# =============================================================================
# =============================================================================
# EVENTS DF'S
events = spark.read.csv("./datasets/Scraped Data/events.csv", header=True, inferSchema=True)
swim_events = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_events.csv", header=True, inferSchema=True)
track_events = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_events.csv", header=True, inferSchema=True)
wh_events = spark.read.csv("./datasets/weight_height_events.csv", header=True, inferSchema=True)

# DATASET LABELS FOR EVENTS
swim_events = swim_events.withColumn("source", F.lit("swim_events"))
track_events = track_events.withColumn("source", F.lit("track_events"))
events = events.withColumn("source", F.lit("base_events"))
wh_events = wh_events.withColumn("source", F.lit("wh_events"))

# Filter for athletics and swimming
wh_events = wh_events.filter(F.lower(F.col("event_type")).isin("athletics", "swimming"))

# BIG UNIFIED TABLE
events_all = events.unionByName(swim_events.unionByName(track_events))
events_all = events_all.unionByName(wh_events, allowMissingColumns=True)
events_all = events_all.filter(F.lower(F.col("event_type")).isin("athletics", "swimming"))

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

# ==================================================================
# ==================================================================
# PARSE EVENT ATTRIBUTES (distance, gender, discipline, round)
# ==================================================================
# ==================================================================

event_schema = T.StructType([
    T.StructField("distance", T.IntegerType(), True),
    T.StructField("gender", T.StringType(), True),
    T.StructField("discipline", T.StringType(), True),
    T.StructField("round", T.StringType(), True),
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

parse_udf = F.udf(lambda e_name, e_type: parse_event(e_name, e_type), event_schema)


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

# -----------------------------------------------------------------------------
# Tokenize event names for fuzzy matching
# -----------------------------------------------------------------------------

# Catch any crappy disciplines with tokens
events_all = events_all.withColumn(
    "tokens",
    F.array_except(
        F.split(F.lower(F.regexp_replace("event_name_clean", r"[^\w\s]", "")), " "),
        F.array(F.lit("mens"), F.lit("womens"), F.lit("heat"), F.lit("final"))
    )
)

# -----------------------------------------------------------------------------
# Pairwise similarity and graph clustering for events
# -----------------------------------------------------------------------------

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

event_pairs = events_all.alias("a").join(
    events_all.alias("b"),
    on=["olympics_id", "event_type", "gender"]
).filter(F.col("a.row_id") < F.col("b.row_id")) \
 .filter(F.col("a.distance") == F.col("b.distance"))

event_pairs = event_pairs.withColumn("distance_match", F.when(F.col("a.distance") == F.col("b.distance"), 1).otherwise(0))
event_pairs = event_pairs.withColumn("discipline_match", F.when(F.col("a.discipline") == F.col("b.discipline"), 1).otherwise(0))
event_pairs = event_pairs.withColumn("gender_match", F.when(F.col("a.gender") == F.col("b.gender"), 1).otherwise(0))
event_pairs = event_pairs.withColumn("token_sim", jaccard(F.col("a.tokens"), F.col("b.tokens")))

event_pairs = event_pairs.withColumn(
    "score",
    0.35 * F.col("distance_match") +
    0.20 * F.col("gender_match") +
    0.25 * F.col("token_sim") +
    0.20 * F.col("discipline_match")
)

event_matches = event_pairs.filter(F.col("score") >= 0.825)
event_edges = event_matches.select(F.col("a.row_id").alias("src"), F.col("b.row_id").alias("dst"))

g_events = GraphFrame(
    events_all.select(F.col("row_id").alias("id")),
    event_edges
)
components = g_events.connectedComponents(checkpointInterval=2)

event_raw_clusters = components.select(
    F.col("id").alias("row_id"),
    F.col("component").alias("cluster_id")
)

events_all = events_all.join(event_raw_clusters, on="row_id", how="left")

event_cluster_map = events_all.groupBy("event_id").agg(F.first("cluster_id").alias("event_cluster_id"))

swim_events = swim_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
track_events = track_events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")
events = events.join(events_all.select("event_id", "cluster_id"), "event_id", "left")

# =============================================================================
# =============================================================================
# LOAD AND UNION OLYMPIC CSVs
# =============================================================================
# =============================================================================

olympics = spark.read.csv("./datasets/Scraped Data/olympics.csv", header=True, inferSchema=True)
swim_olympics = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_olympics.csv", header=True, inferSchema=True)
track_olympics = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_olympics.csv", header=True, inferSchema=True)
wh_olympics = spark.read.csv("./datasets/weight_height_olympics.csv", header=True, inferSchema=True)

olympics = olympics.withColumn("source", F.lit("base_olympics"))
swim_olympics = swim_olympics.withColumn("source", F.lit("swim_olympics"))
track_olympics = track_olympics.withColumn("source", F.lit("track_olympics"))
wh_olympics = wh_olympics.withColumn("source", F.lit("wh_olympics"))

olympics_all = olympics.unionByName(swim_olympics).unionByName(track_olympics)
olympics_all = olympics_all.unionByName(wh_olympics, allowMissingColumns=True)
olympics_all = olympics_all.withColumn("row_id", F.monotonically_increasing_id())
olympics_all = olympics_all.withColumn("host_city_norm", F.lower(F.trim("host_city_name")))

# Block on year + summer + first 5 characters of host city
olympics_all = olympics_all.withColumn(
    "block_key",
    F.concat_ws("_", F.col("year"), F.col("summer"), F.substring("host_city_norm", 1, 5))
)

olympic_pairs = olympics_all.alias("a").join(
    olympics_all.alias("b"),
    on="block_key"
).filter(F.col("a.row_id") < F.col("b.row_id"))

olympic_pairs = olympic_pairs.withColumn(
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

olympic_pairs = olympic_pairs.withColumn("score", F.col("city_sim"))

olympic_edges = olympic_pairs.filter(F.col("score") >= 0.75).select(F.col("a.row_id").alias("src"), F.col("b.row_id").alias("dst"))

g_olympics = GraphFrame(olympics_all.select(F.col("row_id").alias("id")), olympic_edges)
olympic_components = g_olympics.connectedComponents(checkpointInterval=2)

olympics_clusters = olympics_all.join(
    olympic_components.select(F.col("id").alias("row_id"), F.col("component").alias("olympics_cluster_id")),
    on="row_id",
    how="left"
)

# -----------------------------------------------------------------------
# Olympics canonical ID: sequential / readable but memory heavy
# Alternative raw 64-bit cluster to execute program faster is below.
# -----------------------------------------------------------------------
window_oly = Window.orderBy("olympics_cluster_id")
olympics_clusters = olympics_clusters.withColumn("canonical_olympics_id", F.row_number().over(window_oly))
olympics_lookup = olympics_clusters.select("olympics_id", "canonical_olympics_id").distinct()

# ----------------------------------------------------------------
# --- Alternative raw cluster ID (non-sequential 64-bit number)---
# ----------------------------------------------------------------
#
# olympics_clusters = olympics_clusters.withColumn(
#     "canonical_olympics_id", F.col("olympics_cluster_id")
# )
# olympics_lookup = olympics_clusters.select(
#     "olympics_id", "canonical_olympics_id"
# ).distinct()

# Give canon olympic ID to events
events_all = events_all.join(olympics_lookup, on="olympics_id", how="left")

# =============================================================================
# =============================================================================
# LOAD & UNION ATHLETE-EVENT CSVs
# =============================================================================
# =============================================================================

athlete_events = spark.read.csv("./datasets/Scraped Data/athlete_event.csv", header=True, inferSchema=True)
swim_combined = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_combined.csv", header=True, inferSchema=True)
track_combined = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_combined.csv", header=True, inferSchema=True)
wh_combined = spark.read.csv("./datasets/weight_height_combined.csv", header=True, inferSchema=True)

# DATASET LABELS FOR ATHLETE EVENTS
swim_combined = swim_combined.withColumn("source", F.lit("swim_combined"))
track_combined = track_combined.withColumn("source", F.lit("track_combined"))
athlete_events = athlete_events.withColumn("source", F.lit("base_athlete_events"))
wh_combined = wh_combined.withColumn("source", F.lit("wh_combined"))

# BIG UNIFIED TABLE
athlete_events_all = athlete_events.unionByName(swim_combined.unionByName(track_combined))
athlete_events_all = athlete_events_all.unionByName(wh_combined, allowMissingColumns=True)

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

time_udf = F.udf(parse_time, T.DoubleType())
athlete_events_all = athlete_events_all.withColumn("time_seconds", time_udf("time"))

# Join on raw event cluster ID
athlete_events_all = athlete_events_all.join(event_cluster_map, on="event_id", how="left"
)

# =============================================================================
# =============================================================================
# LOAD & CLEAN CANDIDATE ATHLETES
# =============================================================================
# =============================================================================

candidates = spark.read.csv("./datasets/Scraped Data/athletes.csv", header=True, inferSchema=True)
swim_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/swim_candidates.csv", header=True, inferSchema=True)
track_candidates = spark.read.csv("./datasets/Smaller Tables with Dummy Ids/track_candidates.csv", header=True, inferSchema=True)
wh_candidates = spark.read.csv("./datasets/weight_height_candidates.csv", header=True, inferSchema=True)


# Convert country to 3-letter code
candidates = candidates.join(
    country_code_map,
    candidates["nationality_id"] == country_code_map["id"],
    "left"
).withColumn("country_code", F.col("country_code")).drop("id")

# Remap nation_id to country code
swim_candidates = swim_candidates.withColumn("country_code", F.col("nationality_id"))
track_candidates = track_candidates.withColumn("country_code", F.col("nationality_id"))
wh_candidates = wh_candidates.withColumn("country_code", F.col("nationality_id"))

# DATASET LABELS FOR CANDIDATES
swim_candidates = swim_candidates.withColumn("source", F.lit("swim_candidates"))
track_candidates = track_candidates.withColumn("source", F.lit("track_candidates"))
candidates = candidates.withColumn("source", F.lit("candidates"))
wh_candidates = wh_candidates.withColumn("source", F.lit("wh_candidates"))

# BIG UNIFIED TABLE
candidates_all = candidates.unionByName(swim_candidates.unionByName(track_candidates))
candidates_all = candidates_all.unionByName(wh_candidates, allowMissingColumns=True)

# ------------------------------------------------------------------
# Robust name cleaning
# ------------------------------------------------------------------
@F.udf("string")
def normalize_nfc(s):
    return unicodedata.normalize('NFC', s) if s else s

# Unicode normalization on names
candidates_all = candidates_all.withColumn("first_name", normalize_nfc(F.col("first_name"))) \
                               .withColumn("last_name", normalize_nfc(F.col("last_name"))) \
                               .withColumn("middle_name", normalize_nfc(F.col("middle_name")))

# Lowercase & trim
candidates_all = candidates_all.withColumn("first_name", F.lower(F.trim("first_name"))) \
                               .withColumn("last_name", F.lower(F.trim("last_name"))) \
                               .withColumn("middle_name", F.lower(F.trim("middle_name")))

# Get rid of smart quotes
candidates_all = candidates_all.withColumn("first_name", F.regexp_replace("first_name", '["“”]', ''))
candidates_all = candidates_all.withColumn("middle_name", F.regexp_replace("middle_name", '["“”]', ''))
candidates_all = candidates_all.withColumn("last_name", F.regexp_replace("last_name", '["“”]', ''))

# Remove anything inside parentheses and the parentheses themselves
candidates_all = candidates_all.withColumn("first_name", F.regexp_replace("first_name", r"\(.*?\)", "")) \
                               .withColumn("last_name", F.regexp_replace("last_name", r"\(.*?\)", "")) \
                               .withColumn("middle_name", F.regexp_replace("middle_name", r"\(.*?\)", ""))

# Trim again for parentheses step
candidates_all = candidates_all.withColumn("first_name", F.trim("first_name")) \
                               .withColumn("last_name", F.trim("last_name")) \
                               .withColumn("middle_name", F.trim("middle_name"))

# Remove rows where first_name is empty
candidates_all = candidates_all.filter(F.col("first_name") != "")

# Allow only Unicode letters, apostrophe, hyphen, dot
name_regex = "^[\\p{L}'\\.\\-]+$"
candidates_all = candidates_all.filter(
    F.col("first_name").rlike(name_regex) &
    (F.length("first_name") >= 2) &
    ~F.lower(F.col("first_name")).rlike(
        "record|progression|kilometre|metre|world|summary|qualif|heat|final"
    )
)

candidates_all = candidates_all.filter(
    F.col("last_name").rlike(name_regex) |
    (F.col("last_name") == "")
)


# -----------------------------------------------------------------------------
# Relay explosion for concatenated athlete lists in middle_name
# -----------------------------------------------------------------------------
# Only process rows where middle_name contains commas and looks like a relay list
def is_valid_relay_list(mid):
    if mid is None: return False
    parts = [p.strip() for p in mid.split(",")]
    if any(p == "" for p in parts):      # dont use trailing/leading commas
        return False
    if len(parts) < 2: return False
    middle_parts = parts[1:-1] if len(parts) > 2 else []
    return all(len(p.split()) >= 2 for p in middle_parts if p)

is_valid_udf = F.udf(is_valid_relay_list, T.BooleanType())
candidates_all = candidates_all.withColumn("relay_list", is_valid_udf("middle_name"))


relay_df = candidates_all.filter(F.col("relay_list") & F.col("middle_name").isNotNull() & (F.col("middle_name") != ""))
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

extract_athletes_udf = F.udf(extract_athletes, T.ArrayType(T.StructType([
    T.StructField("first", T.StringType(), True),
    T.StructField("last", T.StringType(), True)
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

candidates_all = non_relay_df.unionByName(relay_exploded)

# -----------------------------------------------------------------------------
# Normalize names, handle suffixes, and create blocking keys
# -----------------------------------------------------------------------------
# Get rid of any commas or whitespace
candidates_all = candidates_all.withColumn("middle_name", F.regexp_replace(F.col("middle_name"), r",\s*$", ""))


# If middle name has first or last name, nullify.
candidates_all = candidates_all.withColumn(
    "middle_name",
    F.when(
        F.col("middle_name").contains(F.col("first_name")) |
        F.col("middle_name").contains(F.col("last_name")),
        F.lit(None)
    ).otherwise(F.col("middle_name"))
)

# Substring of athlete first / last names for blocking later
candidates_all = candidates_all.withColumn("first_initial", F.substring("first_name", 1, 1))
candidates_all = candidates_all.withColumn("last_four", F.substring("last_name", 1, 4))

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

# Coalesce decade blocks of yob's to 0 if missing
candidates_all = candidates_all.withColumn(
    "birth_decade", 
    F.when(F.col("year_of_birth").isNotNull(), 
           F.floor(F.col("year_of_birth") / 10) * 10)
     .otherwise(F.lit(0))
)

# Clean up garbage candidates
candidates_all = candidates_all.filter(
    ~((F.col("first_name").rlike(r"^[a-z]\.$")) &
      (F.col("year_of_birth").isNull()) &
      (F.col("height(cm)").isNull()) &
      (F.col("weight(kg)").isNull()) &
      (F.col("middle_name").isNull()))
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
        F.col("last_name")
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

# Gender to single letter
candidates_all = candidates_all.withColumn(
    "gender",
    F.when(F.lower(F.col("gender")).startswith("m"), "M")
     .when(F.lower(F.col("gender")).startswith("w"), "F")
     .when(F.lower(F.col("gender")).startswith("f"), "F")
     .otherwise(F.col("gender"))
)

# ------------------------------------------------------------------
# Global correction of short first names using similar longer ones
# ------------------------------------------------------------------
# Get frequency of each first_name
# Using a window for specific global comparisons on frequency of small names
name_freq_all = candidates_all.groupBy("first_name").agg(F.count("*").alias("freq"))

# Self‑join to find the best correction for short first names that are
# likely corrupted source truncations / data, but only if the short name appears
# often enough to be suspicious (freq > 30).  Extremely rare short names
# are left untouched because they're probably real.
#
# We compare short names (length ≤ 4) with longer names (length ≥ 5),
# compute Levenshtein distance, and target names where:
#   - distance ≤ 2, &&
#   - distance equals the absolute length difference, (only targets
#     missing letters, not substitutions).

correct_pairs = name_freq_all.alias("a").join(
    name_freq_all.alias("b"),
    (F.length("a.first_name") <= 4) & (F.length("b.first_name") >= 5) &
    (F.col("a.freq") > 30)  # Protects very low frequency names
).withColumn("lev", levenshtein("a.first_name", "b.first_name")) \
 .filter(F.col("lev") <= 2) \
 .filter(F.col("lev") == F.abs(F.length("a.first_name") - F.length("b.first_name"))) \
 .withColumn("row_number", F.row_number().over(
     Window.partitionBy("a.first_name").orderBy(F.desc("b.freq"))
 )) \
 .filter(F.col("row_number") == 1) \
 .select(
     F.col("a.first_name").alias("old_name"),
     F.col("b.first_name").alias("new_name")
 )

# Apply the correction
candidates_all = candidates_all.join(correct_pairs,
    candidates_all["first_name"] == correct_pairs["old_name"], "left") \
 .withColumn("first_name", F.coalesce(F.col("new_name"), F.col("first_name"))) \
 .drop("old_name", "new_name")

# ------------------------------------------------------------------
# Ultra‑strict last‑name correction with block of same nation
# ------------------------------------------------------------------

last_freq_nation = candidates_all.groupBy("country_code", "last_name") \
    .agg(F.count("*").alias("freq"))

# Group by country. This way we can compare last name
# frequency compared to other countries to see if it
# is likely to be a legitimate name.
countries_per_name = candidates_all.groupBy("last_name") \
    .agg(F.countDistinct("country_code").alias("num_countries"))

# Only allow correction if the short surname has length ≤ 3, 
# rare within its country (freq < 30), and
# appears in exactly ONE country .
# This ensures cross‑cultural short names are safe.
safe_short = last_freq_nation.alias("s") \
    .join(countries_per_name, on="last_name") \
    .filter(F.col("num_countries") == 1) \
    .filter(F.length("last_name") <= 3) \
    .filter(F.col("freq") < 30) \
    .select("country_code", "last_name", "freq")

# Self‑join to find the best longer candidate:
#    - same country
#    - longer name (≥ 4)
#    - frequent enough to be a plausible correction (≥ 100)
#    - same first two letters
correct_last = safe_short.alias("a").join(
    last_freq_nation.alias("b"),
    (F.col("a.country_code") == F.col("b.country_code")) &
    (F.length("b.last_name") >= 4) &
    (F.col("b.freq") >= 100) &
    (F.substring("a.last_name", 1, 2) == F.substring("b.last_name", 1, 2))
).withColumn("lev", levenshtein("a.last_name", "b.last_name")) \
 .filter(F.col("lev") <= 2) \
 .filter(F.col("lev") == F.abs(F.length("a.last_name") - F.length("b.last_name"))) \
 .withColumn("row_number", F.row_number().over(
     Window.partitionBy("a.country_code", "a.last_name").orderBy(F.desc("b.freq"))
 )) \
 .filter(F.col("row_number") == 1) \
 .select(
     F.col("a.country_code").alias("country"),
     F.col("a.last_name").alias("old_last"),
     F.col("b.last_name").alias("new_last")
 )


# Apply the correction
candidates_all = candidates_all.join(correct_last,
    (candidates_all["country_code"] == correct_last["country"]) &
    (candidates_all["last_name"] == correct_last["old_last"]),
    "left") \
 .withColumn("last_name", F.coalesce(F.col("new_last"), F.col("last_name"))) \
 .drop("country", "old_last", "new_last")

# Create blocking keys
candidates_all = candidates_all.withColumn(
    "block_key",
    F.concat_ws(
        "_",
        F.col("first_initial"),
        F.col("last_four"),
        F.col("birth_decade"),
        F.col("gender"),
        F.coalesce(F.col("country_code"), F.lit("UNK"))
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

# -----------------------------------------------------------------------------
# Missing athletes from athlete_events (dummy id rows)
# -----------------------------------------------------------------------------
# Extract distinct athlete_ids from athlete_events that are not in candidates_all
missing_athlete_ids = athlete_events_all.select("athlete_id").distinct() \
    .join(candidates_all.select("athlete_id").distinct(), "athlete_id", "left_anti")

missing_candidates = missing_athlete_ids.select(
    F.col("athlete_id"),
    F.lit("").alias("first_name"), 
    F.lit("").alias("last_name"),
    F.lit("").alias("middle_name"), 
    F.lit(None).cast(T.IntegerType()).alias("year_of_birth"),
    F.lit("").alias("nationality_id"), 
    F.lit("missing_from_candidates").alias("source")
    )

candidates_all = candidates_all.unionByName(missing_candidates, allowMissingColumns=True)
candidates_all = candidates_all.withColumn("row_id", F.monotonically_increasing_id())
candidates_all = candidates_all.checkpoint() # Constantly going OOM, setting checkpoint

# =============================================================================
# PAIRWISE SIMILARITY & GRAPH CLUSTERING FOR ATHLETES
# =============================================================================
def add_similarity_features(pairs_df):
    pairs_df = pairs_df.withColumn("first_match", F.when(F.col("first_a") == F.col("first_b"), 1).otherwise(0))
    pairs_df = pairs_df.withColumn("last_match", F.when(F.col("last_a") == F.col("last_b"), 1).otherwise(0))
    pairs_df = pairs_df.withColumn("yob_match", F.when(F.col("yob_a") == F.col("yob_b"), 1).otherwise(0))
    pairs_df = pairs_df.withColumn("token_sim", jaccard(F.col("tokens_a"), F.col("tokens_b")))
    pairs_df = pairs_df.withColumn("fullname_a", F.concat_ws(" ", F.col("first_a"), F.col("last_a")))
    pairs_df = pairs_df.withColumn("fullname_b", F.concat_ws(" ", F.col("first_b"), F.col("last_b")))
    pairs_df = pairs_df.withColumn("lev_dist", levenshtein(F.col("fullname_a"), F.col("fullname_b")))
    pairs_df = pairs_df.withColumn("max_len", greatest(F.length("fullname_a"), F.length("fullname_b")))
    pairs_df = pairs_df.withColumn("name_sim",
        F.when(F.col("max_len") == 0, 1.0)
         .otherwise(1.0 - (F.col("lev_dist") / F.col("max_len"))))
    pairs_df = pairs_df.withColumn("score",
        0.25 * F.col("last_match") + 0.15 * F.col("first_match") +
        0.15 * F.col("yob_match") + 0.15 * F.col("token_sim") + 0.30 * F.col("name_sim"))
    return pairs_df

# Strict block key
athlete_pairs_strict = candidates_all.alias("a").join(candidates_all.alias("b"), on="block_key") \
    .filter(F.col("a.row_id") < F.col("b.row_id")) \
    .select(F.col("a.row_id").alias("row_id_a"), F.col("b.row_id").alias("row_id_b"),
            F.col("a.first_name").alias("first_a"), F.col("b.first_name").alias("first_b"),
            F.col("a.last_name").alias("last_a"), F.col("b.last_name").alias("last_b"),
            F.col("a.year_of_birth").alias("yob_a"), F.col("b.year_of_birth").alias("yob_b"),
            F.col("a.name_tokens").alias("tokens_a"), F.col("b.name_tokens").alias("tokens_b"))
athlete_pairs_strict = add_similarity_features(athlete_pairs_strict)

# Relaxed block key (no birth decade) for athletes missing year_of_birth (Michael Phelps somehow)
athlete_pairs_relaxed = candidates_all.alias("a").join(candidates_all.alias("b"), on="block_key_no_decade") \
    .filter(F.col("a.row_id") < F.col("b.row_id")) \
    .select(F.col("a.row_id").alias("row_id_a"), F.col("b.row_id").alias("row_id_b"),
            F.col("a.first_name").alias("first_a"), F.col("b.first_name").alias("first_b"),
            F.col("a.last_name").alias("last_a"), F.col("b.last_name").alias("last_b"),
            F.col("a.year_of_birth").alias("yob_a"), F.col("b.year_of_birth").alias("yob_b"),
            F.col("a.name_tokens").alias("tokens_a"), F.col("b.name_tokens").alias("tokens_b"))
athlete_pairs_relaxed = add_similarity_features(athlete_pairs_relaxed)

# --- Direct Levenshtein block for first‑name corruptions, lev_distance ≤ 2 ---
athlete_pairs_lev = candidates_all.alias("a").join(
    candidates_all.alias("b"),
    on=["last_name", "gender", "country_code"]
).filter(F.col("a.row_id") < F.col("b.row_id")) \
 .filter(
    levenshtein(F.col("a.first_name"), F.col("b.first_name")) <= 2
 ) \
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
athlete_pairs_lev = add_similarity_features(athlete_pairs_lev)


# We filter all blocks individually specifically for 
# athlete_pairs_lev different threshold score
athlete_pairs_strict = athlete_pairs_strict.filter(F.col("score") >= 0.825)
athlete_pairs_relaxed = athlete_pairs_relaxed.filter(F.col("score") >= 0.825)
athlete_pairs_lev = athlete_pairs_lev.filter(F.col("score") >= 0.55)   

# Union the filtered pairs
athlete_pairs_all = athlete_pairs_strict \
    .unionByName(athlete_pairs_relaxed) \
    .unionByName(athlete_pairs_lev)

athlete_matches = athlete_pairs_all

athlete_edges = athlete_matches.select(F.col("row_id_a").alias("src"), F.col("row_id_b").alias("dst"))

g_athletes = GraphFrame(candidates_all.select(F.col("row_id").alias("id")), athlete_edges)
athlete_components = g_athletes.connectedComponents(checkpointInterval=2)

# Build canon for candidates
candidates_canon = candidates_all.join(
    athlete_components.select(
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

# -----------------------------------------------------------------------
# Athlete canonical ID: sequential / readable but memory heavy
# Alternative raw 64-bit cluster to execute program faster is below.
# -----------------------------------------------------------------------
window_spec = Window.orderBy("cluster_id")
candidates_canon = candidates_canon.withColumn(
    "canonical_athlete_id",
    F.row_number().over(window_spec)
)
# Faster non‑sequential alternative:
# candidates_canon = candidates_canon.withColumn(
#     "canonical_athlete_id", F.col("cluster_id")
# )

# Create medal map for candidates
# Maps raw athlete_id -> canonical_athlete_id
medal_map = candidates_canon.select("athlete_id", F.col("canonical_athlete_id").alias("athlete_cluster_id")).distinct()

# =============================================================================
# ATTACH CANONICAL MEDAL WINNERS TO EVENTS
# =============================================================================
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

# Create canon event IDs (one per cluster)
event_id_mapping = events_canon.select("cluster_id") \
    .distinct() \
    .withColumn("canonical_event_id", F.monotonically_increasing_id())

# Map athlete events to canon event IDs
athlete_events_all = athlete_events_all.join(
    event_id_mapping,
    athlete_events_all["event_cluster_id"] == event_id_mapping["cluster_id"],
    "left") \
    .drop("cluster_id", "event_cluster_id") \
    .withColumnRenamed("canonical_event_id", "event_cluster_id")

# Athlete lookup: raw athlete_id -> canonical_athlete_id
athlete_id_map = candidates_canon.select(
    "athlete_id",
    "canonical_athlete_id") \
    .dropDuplicates(["athlete_id"]) \
    .withColumnRenamed("canonical_athlete_id",
                       "athlete_cluster_id"
    )

athlete_events_all = athlete_events_all.join(athlete_id_map, on="athlete_id", how="left")

# =============================================================================
# CANONICAL EVENT TABLE
# =============================================================================
events_clean_raw = events_canon.join(event_id_mapping, "cluster_id") \
    .groupBy("canonical_event_id") \
    .agg(
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

events_final = events_clean_raw   # already one-per-event

# =============================================================================
# RESULTS TABLE
# =============================================================================

results_final = athlete_events_all.select("athlete_cluster_id", "event_cluster_id", "time_seconds", "result")

# =============================================================================
# BUILD FINAL SCHEMA-ALIGNED CLEAN DATASETS
# =============================================================================

# --- Olympics Clean. Dedup by year+season then keep longest city name ---
# Select year + sunmmer + olympic_id, get host_city_name, the aggregate on canon olympics.
olympics_clean = olympics_clusters.groupBy("year", "summer") \
    .agg(
        F.min("canonical_olympics_id").alias("olympics_id"),
        F.max(F.struct(F.length("host_city_name"), "host_city_name")).alias("city_struct")
    ) \
    .select("year", "summer", "olympics_id",
            F.col("city_struct.host_city_name").alias("host_city_name"))

# ----------------------------------------------------------------------------------------
# Supplement olympics with host country name and GDP per capita at the time of olympics.
# ----------------------------------------------------------------------------------------
# Basic lower trims on cities
city_country_norm = city_country.withColumn("city_lower", F.lower(F.trim("city_id")))
olympics_clean = olympics_clean.withColumn("city_lower", F.lower(F.trim("host_city_name")))

# Join city_country_norm with olympics on the city.
olympics_clean = olympics_clean.join(
    city_country_norm.select("city_lower", F.col("country_id").alias("host_country_full")),
    on="city_lower", how="left"
).drop("city_lower")

# Map full country name to 3‑letter code
olympics_clean = olympics_clean.join(full_to_code, 
                                     olympics_clean["host_country_full"] == full_to_code["full_name"], 
                                     "left") \
                               .drop("full_name") \
                               .withColumnRenamed("code", "host_country_code")

# Join with countries GDP on (host_country_code, year)
olympics_clean = olympics_clean.join(
    countries.select("country_id", "year", "gdp_per_capita"),
    (olympics_clean["host_country_code"] == countries["country_id"]) &
    (olympics_clean["year"] == countries["year"]),
    "left"
).drop(countries["year"]) \
.drop("country_id")

olympics_clean = olympics_clean.withColumn("year", F.col("year").cast("int"))
# --- Events Clean ---
events_clean = events_clean_raw.select(
    "event_id", "gold_id", "silver_id", "bronze_id",
    "olympics_id", "event_name", "event_type", "gender"
).distinct()

# --- Olympics Remap. Maps old canon olympic ids to new olympic ids ---
# Get canon olympic id + year + summer and join on olympics_clean year + summer
olympics_remap = olympics_clusters.select(
    "canonical_olympics_id", "year", "summer"
).distinct().join(
    olympics_clean, on=["year", "summer"]
).select("canonical_olympics_id", "olympics_id")

olympics_clean = olympics_clean.select("olympics_id", "year", "summer", "host_city_name")

events_clean = events_clean.join(
    olympics_remap,
    events_clean["olympics_id"] == olympics_remap["canonical_olympics_id"],
    "left"
    # Drop old id and the join key
).drop(events_clean["olympics_id"]) \
 .drop("canonical_olympics_id") 

# Reformat to custom schema
events_clean = events_clean.select(
    "event_id", "gold_id", "silver_id",
    "bronze_id", "olympics_id", "event_name",
    "event_type", "gender"
)

# Normalise event gender to M / F
events_clean = events_clean.withColumn(
    "gender",
    F.when(F.col("gender") == "men", "M")
     .when(F.col("gender") == "women", "F")
     .otherwise(None)
)

# --- Athletes Clean (from candidates_canon. one row per cluster) ---
# Canonical athletes – pick best values per cluster, frequency based.

name_freq = candidates_canon.groupBy("canonical_athlete_id", "first_name", "last_name") \
    .agg(F.count("*").alias("name_count"))

# Keep the cluster (first_name, last_name) combination that appears most often
win_first = Window.partitionBy("canonical_athlete_id").orderBy(F.desc("name_count"), F.desc(F.length("last_name")))
best_names = name_freq.withColumn("row_number", F.row_number().over(win_first)) \
    .filter(F.col("row_number") == 1) \
    .select("canonical_athlete_id", "first_name", "last_name")

# Use best names and aggregate everything else
athletes_clean = candidates_canon.join(best_names, on=["canonical_athlete_id", "first_name", "last_name"], how="inner") \
    .groupBy("canonical_athlete_id", "first_name", "last_name") \
    .agg(
        F.max("height(cm)").alias("height(cm)"),
        F.max("weight(kg)").alias("weight(kg)"),
        F.first("country_code", ignorenulls=True).alias("nationality_id"),
        F.first("year_of_birth", ignorenulls=True).alias("year_of_birth"),
        F.first("nationality_id_at_birth", ignorenulls=True).alias("nationality_id_at_birth"),
        F.first("gender", ignorenulls=True).alias("gender")
    )


# Pick the longest middle_name from the winning rows
best_middle = candidates_canon.join(best_names, on=["canonical_athlete_id", "first_name", "last_name"], how="inner") \
    .groupBy("canonical_athlete_id") \
    .agg(F.max(F.struct(F.length("middle_name"), "middle_name")).alias("mid_struct")) \
    .select("canonical_athlete_id", F.col("mid_struct.middle_name").alias("middle_name"))

athletes_clean = athletes_clean.join(best_middle, on="canonical_athlete_id", how="left")

# Dedup by identity features
athletes_clean = athletes_clean.groupBy("first_name", "last_name", "nationality_id", "gender") \
    .agg(
        F.min("canonical_athlete_id").alias("athlete_id"),
        F.first("middle_name", ignorenulls=True).alias("middle_name"),
        F.first("height(cm)", ignorenulls=True).alias("height(cm)"),
        F.first("weight(kg)", ignorenulls=True).alias("weight(kg)"),
        F.first("year_of_birth", ignorenulls=True).alias("year_of_birth"),
        F.first("nationality_id_at_birth", ignorenulls=True).alias("nationality_id_at_birth")
    )


# Reformat to custom schema
athletes_clean = athletes_clean.select(
    "athlete_id", "first_name", "last_name", "middle_name",
    "height(cm)", "weight(kg)", "nationality_id",
    "year_of_birth", "nationality_id_at_birth", "gender"
)

athletes_clean = athletes_clean.withColumn("year_of_birth", F.col("year_of_birth").cast("int"))

athletes_final = athletes_clean.select(
    "athlete_id", "first_name", "last_name", "nationality_id", "gender", "year_of_birth"
)

# --- Athlete-Event Clean ---
athlete_event_clean = athlete_events_all.select(
    F.col("athlete_cluster_id").alias("athlete_id"),
    F.col("event_cluster_id").alias("event_id"),
    "time", "result"
)

# --- Add medal information ---
events_medal_map = events_clean.select(
    "event_id", "gold_id", "silver_id", "bronze_id"
).distinct()

athlete_event_clean_full = athlete_event_clean.join(
    events_medal_map, on="event_id", how="left"
).withColumn(
    "medal",
    F.when(F.col("athlete_id") == F.col("gold_id"), "Gold")
     .when(F.col("athlete_id") == F.col("silver_id"), "Silver")
     .when(F.col("athlete_id") == F.col("bronze_id"), "Bronze")
     .otherwise(None)
).drop("gold_id", "silver_id", "bronze_id")

# ------------------------------------------------------------------
# Filter athlete-event rows. Only keep Athletics/Swimming events and
# only rows where the athlete gender matches the event gender.
# ------------------------------------------------------------------

# Attach event gender and event type from events_clean
athlete_event_clean_full = (
    athlete_event_clean_full.join(
        events_clean.select("event_id", "event_type", F.col("gender").alias("evt_gender")),
        on="event_id",
        how="inner"
    )
)
# Merges the IDs automatically
athlete_event_clean_full = (
    athlete_event_clean_full.join(
        athletes_clean.select("athlete_id", F.col("gender").alias("ath_gender")),
        on="athlete_id", 
        how="inner"
    )
)
# Keep only rows where athlete gender matches event gender
athlete_event_clean_full = athlete_event_clean_full.filter(
    F.col("ath_gender") == F.col("evt_gender")
)

# Drop the temporary columns
athlete_event_clean_full = athlete_event_clean_full.drop("ath_gender", "evt_gender")

# Drop duplicate athlete‑event pairs (keep first row)
athlete_event_clean_full = athlete_event_clean_full.dropDuplicates(["athlete_id", "event_id"])

# Filter athletes_clean and athletes_final to only athletes that appear in the filtered results
athletes_with_results = athlete_event_clean_full.select("athlete_id").distinct()
athletes_clean = athletes_clean.join(athletes_with_results, on="athlete_id")
athletes_final = athletes_final.join(athletes_with_results, on="athlete_id")

# --- Country and City cleanup ---
city_country_code = (
    city_country
    .join(full_to_code, city_country["country_id"] == full_to_code["full_name"], "left")
    .drop("full_name", "country_id")
    .withColumnRenamed("code", "country_id")
)

# =============================================================================
# READABLE JOINED TABLE FOR FUN
# =============================================================================

readable_results = (
    athlete_event_clean_full.alias("aec")
    .join(
        athletes_clean.select("athlete_id", "first_name", "last_name", "nationality_id", "gender"),
        on="athlete_id", 
        how="inner"
    )
    .join(
        events_clean.select("event_id", "event_name", "olympics_id"),
        on="event_id", 
        how="inner"
    )
    .join(
        olympics_clean.select("olympics_id", "year", "host_city_name"),
        on="olympics_id", 
        how="inner"
    )
    .select(
        "athlete_id",
        "first_name",
        "last_name",
        "nationality_id",
        "gender",
        "event_name",
        "event_type",
        "time",
        "result",
        "year",
        "host_city_name"
    )
)

# Remove impossible participations where athlete is born after
# olympic year
readable_results = readable_results.join(
    athletes_clean.select("athlete_id", "year_of_birth"), on="athlete_id", how="left"
).filter(
    (F.col("year_of_birth") <= F.col("year")) | F.col("year_of_birth").isNull()
).drop("year_of_birth")

# Remove any person whose age at time of competition is greater than 70
readable_results = readable_results.filter(
    (F.col("year_of_birth").isNull()) |
    ((F.col("year") - F.col("year_of_birth") >= 10) &
     (F.col("year") - F.col("year_of_birth") < 80))
)

# Last-resort safety net remove any rows where the event name contradicts the athlete's gender
readable_results = readable_results.filter(
    ~(
        (F.lower(F.col("event_name")).rlike("\\bwomen\\b") & (F.col("gender") != "F")) |
        (F.lower(F.col("event_name")).rlike("\\bmen\\b")   & (F.col("gender") != "M"))
    )
)

# Align to original schemas right before writing
athlete_event_clean_full = athlete_event_clean_full.select(
    "athlete_id", "event_id", "time", "result"
)

athletes_clean = athletes_clean.select(
    "athlete_id", "first_name", "last_name", "middle_name", 
    "height(cm)", "weight(kg)", "nationality_id", "year_of_birth",
    "nationality_id_at_birth", "gender"
)

events_clean = events_clean.select(
    "event_id", "gold_id", "silver_id", "bronze_id",
    "olympics_id", "event_name", "event_type"
)

olympics_clean = olympics_clean.select(
    "olympics_id", "year", "summer", "host_city_name"
)

# =============================================================================
# WRITE CLEAN & CANONICAL CSVs
# =============================================================================
def write_single_csv(df, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        if os.path.isdir(path): shutil.rmtree(path)
        else: os.remove(path)
    tmp_dir = tempfile.mkdtemp()
    df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")
    part_file = next((os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                      if f.startswith("part-") and f.endswith(".csv")), None)
    if part_file is None:
        raise RuntimeError("No part file found in " + tmp_dir)
    shutil.move(part_file, path)
    shutil.rmtree(tmp_dir)

write_single_csv(events_clean, "./clean_datasets/events_clean.csv")
write_single_csv(athlete_event_clean_full, "./clean_datasets/athlete_event_clean.csv")
write_single_csv(athletes_clean, "./clean_datasets/athletes_clean.csv")
write_single_csv(olympics_clean, "./clean_datasets/olympics_clean.csv")
write_single_csv(city_country_code, "./clean_datasets/city_country_clean.csv")
write_single_csv(countries, "./clean_datasets/countries_clean.csv")

write_single_csv(athletes_final, "./canonical/athletes.csv")
write_single_csv(events_final, "./canonical/events.csv")
write_single_csv(results_final, "./canonical/results.csv")
write_single_csv(readable_results, "./readable_results.csv")

print("All done!")
exit(0)