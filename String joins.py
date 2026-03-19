from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

# Create spark session and dataframe
spark = SparkSession.builder.appName("olympic_abb").getOrCreate()
df = spark.read.csv("./mock_olympics.csv", header=True)

# 1 Preprocessing text / Normalizing
punctuation = r'[^\w\s]' # Get rid of punctuation regex

for col_name in df.columns:
    if isinstance(df.schema[col_name].dataType, StringType): # If col is string
        df = df.withColumn(col_name, F.lower(F.col(col_name))) # Lowercase for everything
        df = df.withColumn(col_name, F.regexp_replace(F.col(col_name), punctuation, '')) # Remove puncuation
        df = df.withColumn(col_name, F.trim(F.regexp_replace(F.col(col_name), r'\s+', ' '))) # Remove whitespace

# TODO: handle edge cases like Null and empty
token_cols = []

# Create token columns
for col_name in df.columns:
    token_cols.append(F.col(col_name)) # Keep the original column
    if isinstance(df.schema[col_name].dataType, StringType): # if column is string type
        token_cols.append(F.split(F.col(col_name), " ").alias(f"{col_name}_tokens")) # tokenize by whitespace split
df = df.select(*token_cols) # Finalize changes to df

# =================== 2. Build Abbreviation Sets ====================

# ======== per-column abbreviation dictionaries ==========

# --- Athlete table columns ---
athlete_id_abbr = {}
first_name_abbr = {}
middle_name_abbr = {}
last_name_abbr = {}
height_abbr = {} # was height(cm)
weight_abbr = {} # was weight(kg)
nationality_id_abbr = {}
year_of_birth_abbr = {}
nationality_at_birth_abbr = {}
gender_abbr = {}


# --- Olympics table columns ---
olympics_id_abbr = {}
# year_abbr = {}
summer_abbr = {}
host_city_name_abbr = {}

# --- City / country lookup columns ---
city_id_abbr = {}
# country_id_abbr = {}

# --- GDP / economic table columns ---
country_id_abbr = {}
year_abbr = {}
gdp_per_capita_abbr = {}

# --- Events table columns ---
events_id_abbr = {}
gold_id_abbr = {}
silver_id_abbr = {}
bronze_id_abbr = {}
olympics_id_abbr = {}
event_name_abbr = {}
event_type_abbr = {}

# --- Results table columns ---
# athlete_id_abbr = {}
event_id_abbr = {}
time_abbr = {}
result_abbr = {}



# ===== start building up early pairs ======

# Give each row a unique id so can form ordered pairs
# and avoid comparing row with itself or counting each pair twice
df = df.withColumn("row_id", F.monotonically_increasing_id()) 

# Create specific keys that will likely provide a match
# Target col   - Column for token mining for abbreviations
# blocking keys   - columns that must match exactly to form block
# dict_target  - which named abbreviation dictionary this belongs to
discovery_tasks = [
    {
        "target_col": "event_name",
        "blocking_keys": ["event_type"], # Rows sharing event type
        "dict_target": "event_name_abbr"
    },
    {
        "target_col": "event_name",
        "blocking_keys": ["year"], # Rows sharing year
        "dict_target": "event_name_abbr"
    },
    {
        "target_col": "event_name",
        "blocking_keys": ["country_id"], # Rows with same country id
        "dict_target": "event_name_abbr"
    }
]

# Global candidate set to accumulate candidate rules from every task
all_candidates= None 

for task in discovery_tasks:
    col = task["target_col"]
    keys = task["blocking_keys"]
    token_col = f"{col}_tokens"

    # Join to find pairs in this block
    # so only rows sharing same key values are paired together.
    # Row id makes sure each unordered pair appears exactly once
    # and so a row never pairs with itself
    #
    pairs = df.alias("a").join(
        df.alias("b"),
        on=keys,
        how="inner"
    ).filter(F.col("a.row_id") < F.col("b.row_id"))

    # Mismatches are pairs (not the same thing joined to itself) where text isnt the same.
    # Find tokens that are in 'a' but not in 'b' as potential abbreviations
    # and tokens in 'b' but not in 'a' that would be potential full forms

    # only_in_a    - tokens exclusive to row a taht are candidate abbreviations
    # only_in_b    - tokens exclusive to row b that are candidate full forms
    mismatches = pairs.filter(F.col(f"a.{col}") != F.col(f"b.{col}")) \
        .select(
            "a.row_id", "b.row_id",
            F.array_except(F.col(f"a.{token_col}"), F.col(f"b.{token_col}")).alias("only_in_a"),
            F.array_except(F.col(f"b.{token_col}"), F.col(f"a.{token_col}")).alias("only_in_b")
        ).filter((F.size("only_in_a") > 0) & (F.size("only_in_b") > 0))
    # Size filter drops pairs where one side has no unique tokens ^


    # List of candidate map
    # Flatten so theres only one row per candidate table
    # explode turns the only_in_a array into individual rows so theres one abbreviation per row
    # Array join collapses only_in_b back into single string as full form
    # Category label represents which dictionary this rule targets
    task_rules = mismatches.select(
        F.explode("only_in_a").alias("abbr"),
        F.array_join("only_in_b", " ").alias("full_form"),
        F.lit(task["dict_target"]).alias("category")
    )

    # Build up rules from every task into one df
    if all_candidates is None:
        all_candidates = task_rules
    else:
        all_candidates = all_candidates.union(task_rules)

# Count frequencies to filter low occurrences (LCS)
# group identical (category, abbr, full_form) triples and count
# how oftn each appears across all blocks. Rules seen only once are
# prob noise / bad so we require count > 1.
final_suggestions = all_candidates.groupBy("category", "abbr", "full_form") \
    .count() \
    .filter(F.col("count") > 1) \
    .orderBy(F.desc("count"))

# List to hold (category, abbr, full_form) tuples from manual dicts
manual_rules_list = []

# Map dictionary names to their actual python dict objects
manual_dicts = {
    "athlete_id_abbr": athlete_id_abbr,
    "first_name_abbr": first_name_abbr,
    "height_abbr": height_abbr,
    "weight_abbr": weight_abbr,
    "nationality_abbr": nationality_id_abbr,
    "gender_abbr": gender_abbr,
    "host_city_abbr": host_city_name_abbr,
    "event_name_abbr": event_name_abbr,
    "result_abbr": result_abbr
}

# Flatten each manual dict into shared lsit
for category, d in manual_dicts.items():
    for abbr, full in d.items():
        manual_rules_list.append((category, abbr, full))

# Create dataframe based off manual rule set
manual_df = spark.createDataFrame(manual_rules_list, ["category", "abbr", "full_form"])

# Union with auto suggestions, drop dups on category and abbr so if mapping
# in both sources its kept only once.
# Manual rules override autosuggested rules
all_rules_df = final_suggestions.select("category", "abbr", "full_form") \
    .union(manual_df) \
    .drop_duplicates(["category", "abbr"])

# =================== 3. multipass blocking casts a wider net for abbreviations and full forms =================

# Each inner list is a blocking pass.
# Blocking pass = rows must match on all listed columns to be placed in the same block for that pass.
blocking_passes = [
    ["last_name", "nationality_id"],
    ["first_name", "year_of_birth"],
    ["event_name", "year"]
]

all_candidate_pairs = None

for keys in blocking_passes:
    # self join on this pass's blocking keys. Same row_id trick from above to get
    # each unordered pair exactly once.
    pass_pairs = df.alias("a").join(
        df.alias("b"),
        on=keys,
        how="inner"
    ).filter(F.col("a.row_id") < F.col("b.row_id")) \
    .select(F.col("a.row_id").alias("id_a"), F.col("b.row_id").alias("id_b"))

    if all_candidate_pairs is None:
        all_candidate_pairs = pass_pairs
    else:
        all_candidate_pairs = all_candidate_pairs.union(pass_pairs)

# Drop dups
final_pairs_to_compare = all_candidate_pairs.distinct()

#  =========== 4. Record Comparison ==============

# Apply abbreviation expansion for every column
# replace tokens using abbreviation sets
# TODO: HARDCODING THIS FOR RIGHT NOW
def expand_tokens(tokens):
    if tokens is None:
        return []
    # look up each token in event_name abbr dict.
    # if its not found then keep original token
    return [event_name_abbr.get(token, token) for token in tokens]


expand_udf = F.udf(expand_tokens,  ArrayType(StringType()))

# TODO: HARDCODED FOR NOW
# Call expansion on event_name token array. store the result in new col
# so both orig and exp versions exist.
df = df.withColumn(
    "event_name_expanded",
    expand_udf(F.col("event_name_tokens"))
)

# Create expanded abbreviation dictionaries based off of 'abbreviation tokens'


# Join candidate pair id's back to full df twice (once per side)
# so each result row contains all cols from both row A and row B
pairs = final_pairs_to_compare \
    .join(df.alias("a"), F.col("id_a") == F.col("a.row_id")) \
    .join(df.alias("b"), F.col("id_b") == F.col("b.row_id"))

# Similarity with Jaccard
# Get intersection, union size, and final jaccard score
# F.when covers union size == 0
pairs = pairs.withColumn(
    "intersection",
    F.size(F.array_intersect(
        F.col("a.event_name_expanded"),
        F.col("b.event_name_expanded")
    ))
).withColumn(
    "union",
    F.size(F.array_union(
        F.col("a.event_name_expanded"),
        F.col("b.event_name_expanded")
        ))
).withColumn(
    "similarity",
    F.when(F.col("union") == 0, 0)
    .otherwise(F.col("intersection") / F.col("union"))
)

# Only keep pairs with jaccard similarity above threshold
# so at least 80% of combined token vocab is shared.
j_threshold = 0.8
matches = pairs. filter(F.col("similarity") > j_threshold)

# TODO: Maybe do fuzzy?


# 5. TODO: de duplication and entity resolution


# Make similarity threshold
# if similarity > threshold then its a match

# Merge matched records
#  combine attributes
#  resolve conflicts


# 6. Eval vs ground truth kaggle database

# Compare predicted matches to ground truth

# get evaluation metrics
#  precision
#  recall
#  F1 score

#  Look at errors
#  - false positives
#  - false negatives


# 7. Optimization with tweaks 
# thresholds, similarity, different block sizes,
# increase recall, abbr rules could be less noisy potentially