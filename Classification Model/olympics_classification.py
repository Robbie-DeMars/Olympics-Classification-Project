import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer

#Load csv into pandas dataframes
df_candidates = pd.read_csv("Final Data/athletes_clean.csv") # The individual athletes
df_candidates['year_of_birth'] = pd.to_numeric(df_candidates['year_of_birth'], errors = "coerce")
print("Loaded candidates...")
df_combined = pd.read_csv("Final Data/athlete_event_clean_fixed_time.csv") # The bridge file that connects athletes to their events, and their performance in said event
df_combined["time"] = pd.to_timedelta(df_combined["time"], errors = 'coerce').dt.total_seconds()
print("Loaded combined tables...")
df_events = pd.read_csv("Final Data/events_clean.csv") # The individual events
print("Loaded events...")
df_countries_gdp = pd.read_csv("Final Data/countries_clean.csv") # Country / year / gdp tuples
print("Loaded countries gdp...")
df_city_countries = pd.read_csv("Final Data/city_country_clean.csv") # City / country pairings
print("Loaded country city pairs...")
df_olympics = pd.read_csv("Final Data/olympics_clean.csv") # Olympics and the city they were hosted in

country_list = df_countries_gdp['country_id'].unique() #Save for later
events_list = df_events["event_name"].unique() #Save for later

#Host countries
df_host_countries = pd.merge(df_olympics, df_city_countries, on = "city_id")
print(len(df_host_countries))
df_host_countries = pd.merge(df_host_countries, df_countries_gdp,
                             left_on = ["country_id", "year"],
                             right_on = ["country_id", "year"],
                             how = "inner",
                             suffixes = ("", "_host"))
print(len(df_host_countries))
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
                            right_on = ["country_id", "year"],
                            how = "inner",
                            suffixes = ("_birth", "_competing"))



#Merge to host city
df_tables_merged = pd.merge(df_tables_merged, df_host_countries, on = "olympics_id")
print(len(df_tables_merged))
print(len(df_host_countries))

df_tables_merged.to_csv("test.csv", index = False)

#Preprocessing

#Remove answers
X = df_tables_merged.drop(columns = ["gold_id", "silver_id", "bronze_id", "result"])
#Remove id noise
X = X.drop(columns = ["olympics_id", "athlete_id", "id", "id_birth", "id_competing", "nationality_id_at_birth", "nationality_id", "event_id", "city_id"])
#Remove names (ideally not relevant for your performance in the olympics)
X = X.drop(columns = ["first_name", "middle_name", "last_name"])
#Remove extra birth year
X = X.drop(columns = ["gdp_birth_year"])

features = ['height(cm)', 'weight(kg)', 'year_of_birth', 'gender',
       'country_id_birth', 'gdp_per_capita_birth', 'time',
       'event_name', 'event_type', 'year',
       'country_id_competing', 'gdp_per_capita_competing', 'summer',
       'country_id', 'gdp_per_capita']

#One hot encode label columns
categorical_cols = ["gender", "event_name", "event_type", "country_id_birth", "country_id_competing", "country_id"]
numerical_cols = ['height(cm)', 'weight(kg)', 'year_of_birth',
       'gdp_per_capita_birth', 'time', 'year',
       'gdp_per_capita_competing', 'summer', 'gdp_per_capita']

#Median imputation
#preprocessor = ColumnTransformer([
#    ("cat", OneHotEncoder(handle_unknown = "ignore", sparse_output = False), categorical_cols),
#    ("num", SimpleImputer(strategy = "median"), numerical_cols)
#])

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown = "ignore", sparse_output = False), categorical_cols),
    ("num", KNNImputer(n_neighbors = 5), numerical_cols)
])

y = df_tables_merged["result"].apply(lambda x: x if x in [1.0, 2.0, 3.0] else 0).fillna(0).astype(int)

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

#Evaluate mode

while (True):
    
    #Gender
    while (True):
        gender = input("Please enter gender (M/F):")
        if gender.lower() == 'm' or gender.lower() == 'f':
            break
        else:
            print("Invalid input, please try again.")
    
    #Height
    while (True):
        height = input("Please enter height (cm):")
        try:
            height = float(height)
            break
        except:
            print("Invalid input, please try again.")
    
    #Weight
    while (True):
        weight = input("Please enter weight (kg):")
        try:
            weight = float(weight)
            break
        except:
            print("Invalid input, please try again.")
    
    #Year of Birth
    while (True):
        yob = input("Please enter year of birth (ex. 2002):")
        try:
            yob = int(yob)
            break
        except:
            print("Invalid input, please try again.")
    
    #Country of Birth
    while (True):
        country_of_birth = input("Please enter 3 letter country of birth (ex. USA):")
        if country_of_birth in country_list:
            break
        else:
            print("Invalid input, please try again.")
    
    #Country Competing
    while (True):
        country_competing = input("Please enter 3 letter country team competing (ex. USA):")
        if country_competing in country_list:
            break
        else:
            print("Invalid input, please try again.")

    #Event Participating in
    while (True):
        for x, event in enumerate(events_list):
            print(f"{x}: {event}")
        event_participating_num = input("Please enter the above number of the event:")
        try:
            event_participating_num = int(event_participating_num)
            if event_participating_num < len(events_list) and event_participating_num >= 0:
                event_participating = events_list[event_participating_num]
                event_type = df_events[df_events['event_name'] == event_participating]['event_type'].iloc[0]
                break
            else:
                print("Invalid input, please try again.")
        except:
            print("Invalid input, please try again.")
    
    #Olympics year
    while (True):
        olympics_year = input("Please enter year the Olympics (ex. 2002):")
        try:
            olympics_year = int(olympics_year)
            break
        except:
            print("Invalid input, please try again.")

    #Host country
    while (True):
        country_hosting = input("Please enter 3 letter country hosting (ex. USA):")
        if country_hosting in country_list:
            break
        else:
            print("Invalid input, please try again.")

    #Season
    while (True):
        season = input("Summer or Winter Olympics (S/W):")
        if season.lower() == 's' or season.lower() == 'w':
            if season.lower() == 's':
                summer = 1
            else:
                summer = 0
            break
        else:
            print("Invalid input, please try again.")

    #Time
    while (True):
        time = input("Please enter recent time achieved for applicable sport (hh:mm:ss):")
        time = str(time).strip()
        segments = time.split(':')
        if len(segments) == 3:
            try:
                h = float(segments[0])
                m = float(segments[1])
                s = float(segments[2])

                #Convert it into seconds
                time_in_seconds = int(h) * 3600 + int(m) * 60 + float(s)

                break
            except:
                print("Invalid input, please try again.")

    #Get Relevant GDPs
    #GDP at birth
    r = df_countries_gdp[(df_countries_gdp['country_id'] == country_of_birth) & (df_countries_gdp['year'] == yob)]
    if len(r) > 0:
        gdp_birth = r['gdp_per_capita'].iloc[0]
    else:
        gdp_birth = None
    
    #GDP of competing country
    r = df_countries_gdp[(df_countries_gdp['country_id'] == country_competing) & (df_countries_gdp['year'] == olympics_year)]
    if len(r) > 0:
        gdp_competing = r['gdp_per_capita'].iloc[0]
    else:
        gdp_competing = None
    
    #GDP at birth
    r = df_countries_gdp[(df_countries_gdp['country_id'] == country_hosting) & (df_countries_gdp['year'] == olympics_year)]
    if len(r) > 0:
        gdp_host = r['gdp_per_capita'].iloc[0]
    else:
        gdp_host = None

    #Create single row dataframe to test model on using given input
    user_input = pd.DataFrame([{
        'height(cm)': height,
        'weight(kg)': weight,
        'year_of_birth': yob,
        'gender': gender,
        'country_id_birth': country_of_birth,
        'gdp_per_capita_birth': gdp_birth,
        'time': time_in_seconds,
        'event_name': event_participating,
        'event_type': event_type,
        'year': olympics_year,
        'country_id_competing': country_competing,
        'gdp_per_capita_competing': gdp_competing,
        'summer': summer,
        'country_id': country_hosting,
        'gdp_per_capita': gdp_host
    }])

    input_encoded = preprocessor.transform(user_input)
    prediction = rf.predict(input_encoded)
    print(prediction)
