import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Load csv into pandas dataframes
df_candidates = pd.read_csv("Final Data/athletes.csv") # The individual athletes
df_combined = pd.read_csv("Final Data/athlete_event.csv") # The bridge file that connects athletes to their events, and their performance in said event
df_combined["time"] = pd.to_timedelta(df_combined["time"]).dt.total_seconds()
df_events = pd.read_csv("Final Data/events.csv") # The individual events
df_countries_gdp = pd.read_csv("Final Data/countries.csv") # Country / year / gdp tuples
df_city_countries = pd.read_csv("Final Data/city_country.csv") # City / country pairings
df_olympics = pd.read_csv("Final Data/olympics.csv") # Olympics and the city they were hosted in

#sport, height, weight, age, country of origin, birthplace, gdp of birthplace, gdp of origin country, results?

#Host countries
df_host_countries = pd.merge(df_olympics, df_city_countries, on = "city_id")
df_host_countries = pd.merge(df_host_countries, df_countries_gdp,
                             left_on = ["country_id", "year"],
                             right_on = ["country_id", "year"],
                             how = "inner",
                             suffixes = ("", "_host"))

#Candidates
#Merge candidates to their nationality at birth and gdp of their nationality at birth
df_tables_merged = pd.merge(df_candidates, df_countries_gdp, 
                            left_on = ["nationality_id_at_birth", "year_of_birth"],
                            right_on = ["id", "year"],
                            how = "inner",
                            suffixes = ("", "_birth"))

#Merge to the intermediate events table that also contains that athletes time for that event
df_tables_merged = pd.merge(df_tables_merged, df_combined, on = "athlete_id")

#Merge to events proper
df_tables_merged = pd.merge(df_tables_merged, df_events, on = "event_id")

#Merge to olympics year
df_tables_merged = pd.merge(df_tables_merged, df_olympics[["olympics_id", "year"]], on = "olympics_id")
df_tables_merged = df_tables_merged.rename(columns = {
    "year_x": "gdp_birth_year",
    "year_y": "year"
})


#Merge to the nationality they are participating for and the gdp of that nationality at the time of the performance
df_tables_merged = pd.merge(df_tables_merged, df_countries_gdp,
                            left_on = ["nationality_id", "year"],
                            right_on = ["id", "year"],
                            how = "inner",
                            suffixes = ("_birth", "_competing"))

#Merge to host city
df_tables_merged = pd.merge(df_tables_merged, df_host_countries)

df_tables_merged.to_csv("test.csv", index = False)

print(df_tables_merged)

#Preprocessing

#Remove answers
X = df_tables_merged.drop(columns = ["gold_id", "silver_id", "bronze_id", "result"])
#Remove id noise
X = X.drop(columns = ["olympics_id", "athlete_id", "id", "id_birth", "id_competing", "nationality_id_at_birth", "nationality_id", "event_id", "city_id"])
#Remove names (ideally not relevant for your performance in the olympics)
X = X.drop(columns = ["first_name", "middle_name", "last_name"])

print(X.columns)

features = ['height(cm)', 'weight(kg)', 'year_of_birth', 'gender',
       'country_id_birth', 'gdp_birth_year', 'gdp_per_capita_birth', 'time',
       'event_name', 'event_type', 'year',
       'country_id_competing', 'gdp_per_capita_competing', 'summer',
       'country_id', 'gdp_per_capita']

#One hot encode label columns
categorical_cols = ["gender", "event_name", "event_type", "country_id_birth", "country_id_competing", "country_id"]
numerical_cols = ['height(cm)', 'weight(kg)', 'year_of_birth',
       'gdp_birth_year', 'gdp_per_capita_birth', 'time', 'year',
       'gdp_per_capita_competing', 'summer', 'gdp_per_capita']

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown = "ignore", sparse_output = False), categorical_cols),
    ("num", SimpleImputer(strategy = "median"), numerical_cols)
])

y = df_tables_merged["result"].apply(lambda x: x if x in [1.0, 2.0, 3.0] else 0).fillna(0).astype(int)

print(X)

#Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


rf = RandomForestClassifier(
    n_estimators = 300,
    class_weight = "balanced",
    random_state = 69,
    n_jobs = -1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names = ["No Medal", "Gold", "Silver", "Bronze"]))
print(confusion_matrix(y_test, y_pred))