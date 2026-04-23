import pandas as pd
import numpy as np

def to_hhmmss(t):
    try:
        if pd.isna(t):
            return np.nan
        s = str(t).strip()
        parts = s.split(':')
        if len(parts) == 1:
            # Plain seconds e.g. '12.2' -> '0:00:12.2'
            return f"0:00:{parts[0]}"
        if len(parts) == 2:
            # MM:SS e.g. '4:33.2' -> '0:04:33.2'
            return f"0:{int(parts[0]):02}:{parts[1]}"
        if len(parts) == 3:
            # Already HH:MM:SS e.g. '2:58:50' -> '2:58:50'
            return s
    except (ValueError, AttributeError):
        return np.nan

df = pd.read_csv("Final Data/athlete_event.csv")
df['time'] = df['time'].apply(to_hhmmss)
df.to_csv("Final Data/athlete_event.csv", index=False)

print(df['time'].head(20))
print(f"Unparseable times: {df['time'].isna().sum()}")
