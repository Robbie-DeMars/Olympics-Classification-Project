import pandas as pd
df = pd.read_csv('../Scraped Data/events.csv')

df['gold_id'] = df['gold_id'].apply(lambda x: '' if pd.isna(x) else [int(x)])
df['silver_id'] = df['silver_id'].apply(lambda x: '' if pd.isna(x) else [int(x)])
df['bronze_id'] = df['bronze_id'].apply(lambda x: '' if pd.isna(x) else [int(x)])

df.to_csv('events.csv', index = False)