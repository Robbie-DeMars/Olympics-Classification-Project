from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from itertools import combinations
from graphframes import GraphFrame
from pyspark.sql.functions import broadcast


# Create spark session and dataframe
spark = SparkSession.builder.appName("olympic_abb").getOrCreate()
df = spark.read.csv("./mock_olympics.csv", header=True)


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

# ================================
# 1 Preprocessing text / Normalizing
# =================================
punctuation = r'[^\w\s]' # Get rid of punctuation regex

for col in df.columns:
    if isinstance(df.schema[col].dataType, StringType): # If col is string
        df = df.withColumn(
    col,
    F.trim(
        F.regexp_replace(
            F.regexp_replace(F.lower(F.col(col)), punctuation, ''),
            r'\s+', ' '
        )
    )
)

df = df.withColumn("row_id", F.monotonically_increasing_id()) 


# =====================================
# 2. Tokenization
# ======================================

# Create token columns
original_cols = df.columns

feature_cols = [
    "first_name",
    "last_name",
    "year_of_birth",
    "nationality_id",
    "gender"
]

base_cols = [c for c in feature_cols if c in df.columns]

token_cols = [f"{c}_tokens" for c in feature_cols]

for col in feature_cols:
    df = df.withColumn(f"{col}_tokens", F.split(F.col(col), " "))

# ===== start building up early pairs ======


# Create specific keys that will likely provide a match
# Target col   - Column for token mining for abbreviations
# blocking keys   - columns that must match exactly to form block
# dict_target  - which named abbreviation dictionary this belongs to

#abbr_columns = [c for c in df.columns if isinstance(df.schema[c].dataType, StringType)]

#discovery_tasks = []

all_candidates = None


for col in feature_cols:
    token_col = f"{col}_tokens"

    blocking_keys = ["nationality_id", "year_of_birth"]

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
    for col in feature_cols
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

# abbr_map = {
#     (row["category"], row["abbr"]): row["full_form"]
#     for row in all_rules_df.collect()
# }

rules_df = all_rules_df.select("category", "abbr", "full_form")

#broadcast_map = spark.sparkContext.broadcast(abbr_map)

# ==============================
# 5 Token Expansion
# ==============================

# Each inner list is a blocking pass.
# Blocking pass = rows must match on all listed columns to be placed in the same block for that pass.

# def expand_tokens(tokens, category):
#     if tokens is None:
#         return []
#     expanded = []
#     for t in tokens:
#         val = broadcast_map.value.get((category, t), t)
#         expanded.extend(val.split(" "))
#     return expanded

# expand_udf = F.udf(expand_tokens,  ArrayType(StringType()))

expanded_dfs = []

for col in feature_cols:
    token_col = f"{col}_tokens"
    category = f"{col}_abbr"

    exploded = df.select(
        "row_id",
        F.explode(F.col(token_col)).alias("token")
    )
    mapped = exploded.join(
        broadcast(rules_df),
        (rules_df.abbr ==exploded.token) &
        (rules_df.category == F.lit(category)),
        how= "left"
    ).withColumn(
        "expanded_token",
        F.coalesce(F.col("full_form"), F.col("token"))
    )

    expanded_dfs.append(
        mapped.groupBy("row_id")
        .agg(F.collect_list("expanded_token").alias(f"{col}_expanded"))
    )

    # df = df.join(
    #     expanded_df,
    #     on="row_id", 
    #     how="left"
    #     )

for expanded_df in expanded_dfs:
    df = df.join(expanded_df, on="row_id", how="left")
# =====================
# 6. Blocking Passes
# =====================

keywords = [
    "nationality_id",
    "year_of_birth",
    "last_name"
] # Biased search for common prio words


blocking_candidates = [
    "nationality_id",
    "year_of_birth",
    "last_name"
]
ranked_candidates = []

for col in blocking_candidates:
    prio_score = 0
    for keyword in keywords:
        if keyword in col:
            prio_score += 1
    ranked_candidates.append((col, prio_score))

# ranked_candidates = sorted(
#     ranked_candidates,
#     key = lambda item: item[1],
#     reverse = True
# )

blocking_candidates = [col for col, score in ranked_candidates if score > 0]

if not blocking_candidates:
    blocking_candidates = base_cols[:3]

df = df.withColumn(
    "block_last4",
    F.substring("last_name", 1, 4)
)

df = df.withColumn(
    "block_first2",
    F.substring("first_name", 1, 2)
)

blocking_candidates += ["block_last4", "block_first2"]

blocking_candidates = blocking_candidates[:5]

blocking_passes = [
    # extreme strict block
    ["first_name", "last_name", "year_of_birth"],

    # strict block
    ["last_name", "year_of_birth"],

    # medium block
    ["block_last4", "year_of_birth"],
    ["block_first2", "block_last4"],

    # loose block
    ["block_last4"],
    ["block_first2"],

    # wide-net block
    ["year_of_birth"]
]

all_pairs = None

for keys in blocking_passes:
    # self join on this pass's blocking keys. Same row_id trick from above to get
    # each unordered pair exactly once.
    pairs = df.alias("a").join(
        df.alias("b"),
        on=keys,
        how="inner"
    ).filter(
        (F.col("a.first_name")[0] == F.col("b.first_name")[0]) |
        (F.col("a.last_name")[0] == F.col("b.last_name")[0])
        ) \
    .select(
        F.col("a.row_id").alias("id_a"),
        F.col("b.row_id").alias("id_b")
    )

    if all_pairs is None:
        all_pairs = pairs
    else:
        all_pairs = all_pairs.unionByName(pairs)

# Drop dups
candidate_pairs = all_pairs.dropDuplicates(["id_a", "id_b"])

# Quick filter for garbage pairs

pairs = pairs.filter(
    (F.col("a.first_name")[0] == F.col("b.first_name")[0]) |
    (F.col("a.last_name")[0] == F.col("b.last_name")[0])
)

# ========================
# 7 Similarity
# ========================

# Similarity with Jaccard
# Get intersection, union size, and final jaccard score

for col in feature_cols:
    expanded_col = f"{col}_expanded"
    df = df.withColumn(
        expanded_col,
        F.coalesce(F.col(expanded_col), f"{col}_tokens"))

#Jaccard
def jaccard(colA, colB):
    return F.when(
        F.size(F.array_union(colA, colB)) == 0,0
    ).otherwise(
        F.size(F.array_intersect(colA, colB)) /
        F.size(F.array_union(colA, colB))
    )

def safe_join(x):
    return F.coalesce(F.array_join(x, " "), F.lit(""))

# Fuzzy jaccard
def fuzzy_jaccard(colA, colB, j_weight=0.7):
    base_j = jaccard(colA, colB)
    joinedA = safe_join(colA)
    joinedB = safe_join(colB)
    lev_sim = F.when(
        (F.length(joinedA) == 0) | (F.length(joinedB) == 0),0) \
    .otherwise(
    1 - (F.levenshtein(joinedA, joinedB) / 
         F.greatest(F.length(joinedA), F.length(joinedB)))
)    
    return j_weight * base_j + (1-j_weight) * lev_sim

# Join candidate pair id's back to full df twice (once per side)
# so each result row contains all cols from both row A and row B
pairs = candidate_pairs \
    .join(df.alias("a"), F.col("id_a") == F.col("a.row_id")) \
    .join(df.alias("b"), F.col("id_b") == F.col("b.row_id"))

sim_cols = []

for col in feature_cols:
    expanded_col = f"{col}_expanded"
    sim_name = f"sim_{col}"
    pairs = pairs.withColumn(
        sim_name,
        fuzzy_jaccard(F.col(f"a.{expanded_col}"), F.col(f"b.{expanded_col}"))
    )
    sim_cols.append(sim_name)




# Default dynamic
# weight = 1.0 / len(sim_cols)
# score_expr = sum(weight * F.col(c) for c in sim_cols)

# Biased hardcode
weights = {
    "first_name": 0.35,
    "last_name": 0.35,
    "year_of_birth": 0.15,
    "nationality_id": 0.1,
    "gender": 0.05
}

score_expr = 0
for col in feature_cols:
    sim_col = f"sim_{col}"
    w = weights.get(col, 0.05) # default weight
    score_expr += w * F.col(sim_col)

pairs = pairs.withColumn("final_score", score_expr)

j_threshold = 0.8
matches = pairs.filter(F.col("final_score") > j_threshold)
matches.select("id_a", "id_b", "final_score").show(20, False)

# ===================
# Clustering
# ===================


# Create graph with connected components based on id_a and id_b.
# Faster than union find.
vertices = df.select(F.col("row_id").alias("id"))
edges = matches.select(
    F.col("id_a").alias("src"),
    F.col("id_b").alias("dst")
)

edges = edges.filter(F.col("src") != F.col("dst")) # Self loop
edges = edges.dropDuplicates(["src", "dst"])

g = GraphFrame(vertices, edges)
components = g.connectedComponents()

# Do a union find
# match_pairs = matches.select("id_a", "id_b").collect()

# parent = {}

# # Find node's parent
# def find(x):
#     if parent[x] != x:
#         parent[x] = find(parent[x]) # Doing path compression to filter down parents of each node
#     return parent[x]

# # Union between two rows
# def union(x,y):
#     root_x = find(x)
#     root_y = find(y)
#     if root_x != root_y: # Dont union similar pairs
#         parent[root_y] = root_x

# # Init parent pointers
# for row in match_pairs:
#     a = row["id_a"]
#     b = row["id_b"]
#     parent.setdefault(a, a)
#     parent.setdefault(b,b)

# for row in match_pairs:
#     union(row["id_a"], row["id_b"])


# all_ids = [row["row_id"] for row in df.select("row_id").collect()]
# # If there are rows not in a cluster, put them in a cluster
# for r_id in all_ids:
#     parent.setdefault(r_id, r_id)

# clusters = {}

# for node in parent:
#     root = find(node)
#     if root not in clusters:
#         clusters[root] = []
#     clusters[root].append(node)

# cluster_rows = []

# for root, members in clusters.items():
#     for m in members:
#         cluster_rows.append((m, root))

# cluster_df = spark.createDataFrame(cluster_rows, ["row_id", "cluster_id"])
# df_w_clusters = df.join(cluster_df, on= "row_id", how="left")
# # Group clusters for null values
# df_w_clusters = df_w_clusters.withColumn(
#     "cluster_id",
#     F.coalesce(F.col("cluster_id"), F.col("row_id"))
# )

cluster_df = components.select(
    F.col("id").alias("row_id"),
    F.col("component").alias("cluster_id")
)

# ===========================
# 9. Merging
# ===========================

df_with_clusters = df.join(cluster_df, on = "row_id", how = "left")

df_with_clusters = df_with_clusters.withColumn(
    "cluster_id",
    F.coalesce(F.col("cluster_id"), F.col("row_id"))
)

agg_exprs = []

for col in df.columns:
    if col in ["row_id", "cluster_id"]:
        continue
    if isinstance(df.schema[col].dataType, StringType):
        agg_exprs.append(F.mode(col, ignorenulls=True).alias(col))
    else:
        agg_exprs.append(F.first(col, ignorenulls=True).alias(col))

clean_df = df_with_clusters.groupBy("cluster_id").agg(*agg_exprs)

# Athlete table for ML.
# TODO: potentially modify some of the paramaters / which are included
athletes = df_with_clusters.groupBy("cluster_id").agg(
    F.mode("first_name").alias("first_name"),
    F.mode("last_name").alias("last_name"),
    F.avg("height").alias("height"),
    F.avg("weight").alias("weight"),
    F.mode("nationality_id").alias("nationality_id"),
    F.mode("year_of_birth").alias("year_of_birth"),
    F.mode("gender").alias("gender")
).withColumnRenamed("cluster_id", "athlete_id")

events_df = df_with_clusters.select(
    "cluster_id",   # athelete cluster
    "event_id",
    "event_type",
    "olympics_id",
    "year",
    "gold_id",
    "silver_id",
    "bronze_id"
).withColumnRenamed("cluster_id", "athlete_id")

gdp_df = df.select(
    "country_id",
    "year",
    "gdp_per_capita"
).dropDuplicates()


# ==============================
# 10. Eval
# ==============================

# Time to start filtering the truth.
# Build truth pairs the same way
truth_cols = ["nationality_id", "year_of_birth"]

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


# BUILD ML TABLE

ml_df = events_df.join(
    athletes,
    on = "athlete_id",
    how = "inner"
)

ml_df = ml_df.join(
    gdp_df,
    on = ["country_id", "year"],
    how = "left"
)

ml_df = ml_df.withColumn("height", F.col("height").cast("double"))
ml_df = ml_df.withColumn("weight", F.col("weight").cast("double"))
ml_df = ml_df.withColumn("year_of_birth", F.col("year_of_birth").cast("int"))
ml_df = ml_df.withColumn("year", F.col("year").cast("int"))


# Binary medal label
ml_df = ml_df.withColumn(
    "label",
    F.when(
        (F.col("athlete_id") == F.col("gold_id")) |
        (F.col("athlete_id") == F.col("silver_id")) |
        (F.col("athlete_id") == F.col("bronze_id")),
        1  # 1 if medal
    ).otherwise(0) # 0 if no medal
)

# Drop the spoilers for training
ml_df = ml_df.drop("gold_id", "silver_id", "bronze_id")

# Some small feature engineering
ml_df = ml_df.withColumn(
    "age",
    F.col("year") - F.col("year_of_birth")
)

# Drop empty.
ml_df = ml_df.dropna(subset=[
    "height",
    "weight",
    "age"
])



ml_df.write.mode("overwrite").parquet("ml_data")

# basically lab3 classification for now

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE

# pdf = ml_df.toPandas()

# X = pdf.drop(columns=["label", "athlete_id"])
# y = pdf["label"]

# # Label encoders
# label_encoders = {}
# for col in X.select_dtypes(include=["object"]).columns:
#     label_encoder = LabelEncoder()
#     X[col] = label_encoder.fit_transform(X[col].astype(str))
#     label_encoders[col] = label_encoder

# X = X.fillna(0)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)

# y_pred_lr = lr.predict(X_test)

# print("LR Accuracy:", accuracy_score(y_test, y_pred_lr))
# print(classification_report(y_test, y_pred_lr))

# # RF as well to compare
# rf = RandomForestClassifier(n_estimators=100)
# rf.fit(X_train, y_train)

# y_pred_rf = rf.predict(X_test)

# print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
# print(classification_report(y_test, y_pred_rf))


# # Smote as well
# smote = SMOTE(random_state=42)

# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# rf_smote = RandomForestClassifier(n_estimators=100)
# rf_smote.fit(X_resampled, y_resampled)

# y_pred_smote = rf_smote.predict(X_test)

# print("Random Forest with SMOTE Accuracy:", accuracy_score(y_test, y_pred_smote))
# print(classification_report(y_test, y_pred_smote))
