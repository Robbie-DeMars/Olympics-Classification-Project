import pandas as pd
import numpy as np

def men_to_m_women_to_f(g):
    if pd.isna(g):
        return np.nan

    string_gender = str(g).strip()
    if g == "Men":
        return "M"
    else:
        return "F"


df = pd.read_csv("Smaller Tables/swim_candidates.csv")
df['gender'] = df['gender'].apply(men_to_m_women_to_f)
df.to_csv("Smaller Tables/swim_candidates_fixed_gender.csv", index = False)

df = pd.read_csv("Smaller Tables/track_candidates.csv")
df['gender'] = df['gender'].apply(men_to_m_women_to_f)
df.to_csv("Smaller Tables/track_candidates_fixed_gender.csv", index = False)

df = pd.read_csv("Smaller Tables/weight_height_candidates.csv")
df['gender'] = df['gender'].apply(men_to_m_women_to_f)
df.to_csv("Smaller Tables/weight_height_candidates_fixed_gender.csv", index = False)