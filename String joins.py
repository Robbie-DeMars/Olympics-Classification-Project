from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType
from itertools import combinations

# Create spark session and dataframe
spark = SparkSession.builder.appName("olympic_abb").getOrCreate()
df = spark.read.csv("./mock_olympics.csv", header=True)


# ================================
# 1 Preprocessing text / Normalizing
# =================================
punctuation = r'[^\w\s]' # Get rid of punctuation regex

for col in df.columns:
    if isinstance(df.schema[col].dataType, StringType): # If col is string
        df = df.withColumn(col, F.lower(F.col(col))) # Lowercase for everything
        df = df.withColumn(col, F.regexp_replace(F.col(col), punctuation, '')) # Remove puncuation
        df = df.withColumn(col, F.trim(F.regexp_replace(F.col(col), r'\s+', ' '))) # Remove whitespace



df = df.withColumn("row_id", F.monotonically_increasing_id()) 

string_cols = [c for c in df.columns if isinstance(df.schema[c].dataType, StringType)]

# =====================================
# 2. Tokenization
# ======================================

# Create token columns
for col in string_cols:
    df = df.withColumn(f"{col}_tokens", F.split(F.col(col), " "))

base_cols = [c for c in df.columns if isinstance(df.schema[c].dataType, StringType)]
string_cols = [f"{c}_tokens" for c in base_cols]

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


# Create specific keys that will likely provide a match
# Target col   - Column for token mining for abbreviations
# blocking keys   - columns that must match exactly to form block
# dict_target  - which named abbreviation dictionary this belongs to

#abbr_columns = [c for c in df.columns if isinstance(df.schema[c].dataType, StringType)]

#discovery_tasks = []

all_candidates = None





for col in base_cols:
    token_col = f"{col}_tokens"

    blocking_keys = [c for c in df.columns if c != col and c != "row_id"]

    pairs = df.alias("a").join(
        df.alias("b"),
        on=blocking_keys,
        how="inner"
    ).filter(F.col("a.row_id") < F.col("b.row_id"))

    mismatches = pairs.filter(F.col(f"a.{col}") != F.col(f"b.{col}")) \
    .select(
        "a.row_id", "b.row_id",
        F.array_except(F.col(f"a.{token_col}"), F.col(f"b.{token_col}")).alias("only_a"),
        F.array_except(F.col(f"b.{token_col}"), F.col(f"a.{token_col}")).alias("only_b")
    ).filter((F.size("only_a") > 0) & (F.size("only_b") > 0))

    rules = mismatches.select(
        F.explode("only_a").alias("abbr"),
        F.array_join("only_b", " ").alias("full_form"),
        F.lit(f"{col}_abbr").alias("category")
    )

    if all_candidates is None:
        all_candidates = rules
    else:
        all_candidates = all_candidates.union(rules)


# Count frequencies to filter low occurrences (LCS)
# group identical (category, abbr, full_form) triples and count
# how oftn each appears across all blocks. Rules seen only once are
# prob noise / bad so we require count > 1.
final_suggestions = all_candidates.groupBy("category", "abbr", "full_form") \
    .count() \
    .filter(F.col("count") > 1) \
    .filter(F.length("abbr") <= 5) \
    .filter(F.size(F.split("full_form", " ")) <= 3) \
    .orderBy(F.desc("count"))



# ========================
# 4. Manual dicts
# ========================


# Map dictionary names to their actual python dict objects
manual_dicts = {
    f"{col}_abbr": {}
    for col in base_cols
}

manual_rules =[]

# Flatten each manual dict into shared lsit
for category, d in manual_dicts.items():
    for abbr, full in d.items():
        manual_rules.append((category, abbr, full))

# Create dataframe based off manual rule set
manual_df = spark.createDataFrame(manual_rules, ["category", "abbr", "full_form"])

# Union with auto suggestions, drop dups on category and abbr so if mapping
# in both sources its kept only once.
# Manual rules override autosuggested rules
all_rules_df = final_suggestions.select("category", "abbr", "full_form") \
    .union(manual_df) \
    .drop_duplicates(["category", "abbr"])

abbr_map = {
    (row["category"], row["abbr"]): row["full_form"]
    for row in all_rules_df.collect()
}
broadcast_map = spark.sparkContext.broadcast(abbr_map)

# ==============================
# 5 Token Expansion
# ==============================

# Each inner list is a blocking pass.
# Blocking pass = rows must match on all listed columns to be placed in the same block for that pass.

def expand_tokens(tokens, category):
    if tokens is None:
        return []
    expanded = []
    for t in tokens:
        val = broadcast_map.value.get((category, t), t)
        expanded.extend(val.split(" "))
    return expanded

expand_udf = F.udf(expand_tokens,  ArrayType(StringType()))

for col in string_cols:
    category = f"{col.replace('_tokens', '')}_abbr"
    df = df.withColumn(
        f"{col}_expanded",
        expand_udf(F.col(col), F.lit(category))
    )


# =====================
# 6. Blocking Passes
# =====================


blocking_passes = []

for k in [1]:
    for combo in combinations(base_cols, k):
        blocking_passes.append(list(combo))

all_pairs = None

for keys in blocking_passes:
    # self join on this pass's blocking keys. Same row_id trick from above to get
    # each unordered pair exactly once.
    pairs = df.alias("a").join(
        df.alias("b"),
        on=keys,
        how="inner"
    ).filter(F.col("a.row_id") < F.col("b.row_id")) \
    .select(
        F.col("a.row_id").alias("id_a"),
        F.col("b.row_id").alias("id_b")
    )

    if all_pairs is None:
        all_pairs = pairs
    else:
        all_pairs = all_pairs.union(pairs)

# Drop dups
candidate_pairs = all_pairs.distinct()



# ========================
# 7 Similarity
# ========================

# Similarity with Jaccard
# Get intersection, union size, and final jaccard score

#Jaccard
def jaccard(colA, colB):
    return F.when(
        F.size(F.array_union(colA, colB)) == 0,0
    ).otherwise(
        F.size(F.array_intersect(colA, colB)) /
        F.size(F.array_union(colA, colB))
    )

# Fuzzy jaccard
def fuzzy_jaccard(colA, colB, j_weight=0.7):
    base_j = jaccard(colA, colB)
    joinedA = F.array_join(colA, " ")
    joinedB = F.array_join(colB, " ")
    lev_sim = 1 - (F.levenshtein(joinedA, joinedB) / F.greatest(F.length(joinedA), F.length(joinedB)))
    return j_weight * base_j + (1-j_weight) * lev_sim

# Join candidate pair id's back to full df twice (once per side)
# so each result row contains all cols from both row A and row B
pairs = candidate_pairs \
    .join(df.alias("a"), F.col("id_a") == F.col("a.row_id")) \
    .join(df.alias("b"), F.col("id_b") == F.col("b.row_id"))

sim_cols = []

for col in base_cols:
    expanded_col = f"{col}_expanded"
    sim_name = f"sim_{col}"
    pairs = pairs.withColumn(
        sim_name,
        fuzzy_jaccard(F.col(f"a.{expanded_col}"), F.col(f"b.{expanded_col}"))
    )
    sim_cols.append(sim_name)




# Default dynamic
weight = 1.0 / len(sim_cols)

score_expr = sum(weight * F.col(c) for c in sim_cols)

pairs = pairs.withColumn("final_score", score_expr)

j_threshold = 0.8
matches = pairs.filter(F.col("final_score") > j_threshold)
matches.select("id_a", "id_b", "final_score").show(20, False)

# ===================
# Clustering
# ===================


# Do a union find
match_pairs = matches.select("id_a", "id_b").collect()

parent = {}

# Find node's parent
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x]) # Doing path compression to filter down parents of each node
    return parent[x]

# Union between two rows
def union(x,y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y: # Dont union similar pairs
        parent[root_y] = root_x

# Init parent pointers
for row in match_pairs:
    a = row["id_a"]
    b = row["id_b"]
    parent.setdefault(a, a)
    parent.setdefault(b,b)

for row in match_pairs:
    union(row["id_a"], row["id_b"])


all_ids = [row["row_id"] for row in df.select("row_id").collect()]
# If there are rows not in a cluster, put them in a cluster
for r_id in all_ids:
    parent.setdefault(r_id, r_id)

clusters = {}

for node in parent:
    root = find(node)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(node)

cluster_rows = []

for root, members in clusters.items():
    for m in members:
        cluster_rows.append((m, root))

cluster_df = spark.createDataFrame(cluster_rows, ["row_id", "cluster_id"])
df_w_clusters = df.join(cluster_df, on= "row_id", how="left")
# Group clusters for null values
df_w_clusters = df_w_clusters.withColumn(
    "cluster_id",
    F.coalesce(F.col("cluster_id"), F.col("row_id"))
)


# ===========================
# 9. Merging
# ===========================
agg_exprs = []

for col in df.columns:
    if col in ["row_id", "cluster_id"]:
        continue
    if isinstance(df.schema[col].dataType, StringType):
        agg_exprs.append(F.mode(col, ignorenulls=True).alias(col))
    else:
        agg_exprs.append(F.first(col, ignorenulls=True).alias(col))

merged = df_w_clusters.groupBy("cluster_id").agg(*agg_exprs)


# ==============================
# 10. Eval
# ==============================

# Time to start filtering the truth.
# Build truth pairs the same way
truth_cols = [c for c in df.columns if "id" in c.lower()]

truth_clusters = df.groupBy(*truth_cols).agg(
    F.collect_list("row_id").alias("members")
)

pairs_truth = truth_clusters \
    .withColumn("a", F.explode("members")) \
    .withColumn("b", F.explode("members")) \
    .filter(F.col("a") < F.col("b")) \
    .select(
        F.col("a").alias("id_a"),
        F.col("b").alias("id_b")
    )


clusters_grouped = cluster_df.groupBy("cluster_id").agg(
    F.collect_list("row_id").alias("members")
)

pairs_prediction = clusters_grouped \
.withColumn("a", F.explode("members")) \
.withColumn("b", F.explode("members")) \
.filter(F.col("a") < F.col("b")) \
.select(
    F.col("a").alias("id_a"),
    F.col("b").alias("id_b")
)

# True positives
tp = pairs_prediction.join(
    pairs_truth,
    on=["id_a", "id_b"],
    how="inner"
)

# False positives
fp = pairs_prediction.join(
    pairs_truth,
    on=["id_a", "id_b"],
    how="left_anti"
)

# False negatives
fn = pairs_truth.join(
    pairs_prediction,
    on=["id_a", "id_b"],
    how="left_anti"
)

tp_count = tp.count()
fp_count = fp.count()
fn_count = fn.count()

precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
f1 = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precisiton: {precision}, Recall: {recall}, F1: {f1}")