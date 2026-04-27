'''
olympics classification project
Group:
Logan Hunter, u1342460
Quang Nguyen, u1470229
Robbie DeMars, u1411323

This script helps fix time formats that don't match the needs of the classification model.
'''

import pandas as pd
import numpy as np

def to_hh_mm_ss(t):
    if pd.isna(t):
        return np.nan
    string_time = str(t).strip()
    segments = string_time.split(':')

    #Just seconds
    if len(segments) == 1:
        return f"00:00:{segments[0]}"
    
    elif len(segments) == 2:
        return f"00:{segments[0]}:{segments[1]}"

    elif len(segments) == 3:
        return string_time
    
df = pd.read_csv("Final Data/athlete_event_clean.csv")
df['time'] = df['time'].apply(to_hh_mm_ss)
df.to_csv("Final Data/athlete_event_clean_fixed_time.csv", index = False)

df = pd.read_csv("Scraped Data/athlete_event.csv")
df['time'] = df['time'].apply(to_hh_mm_ss)
df.to_csv("Scraped Data/athlete_event_fixed_time.csv", index = False)