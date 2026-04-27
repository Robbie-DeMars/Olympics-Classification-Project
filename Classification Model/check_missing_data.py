'''
olympics classification project
Group:
Logan Hunter, u1342460
Quang Nguyen, u1470229
Robbie DeMars, u1411323

This script analyzes the completeness of each of our dataset, for the purpose of analyzing
how well our data aggregation was.
'''


import pandas as pd
import numpy as np

datasets = {
    'final': {
        'candidates': 'Final Data/athletes_clean.csv',
        'combined': 'Final Data/athlete_event_clean_fixed_time.csv',
        'events': 'Final Data/events_clean.csv',
        'countries_gdp': 'Final Data/countries_clean.csv',
        'city_country': 'Final Data/city_country_clean.csv',
        'olympics': 'Final Data/olympics_clean.csv'
    },
    'small_swim': {
        'candidates': 'Smaller Tables/swim_candidates_fixed_gender.csv',
        'combined': 'Smaller Tables/swim_combined.csv',
        'events': 'Smaller Tables/swim_events.csv',
        'countries_gdp': 'Smaller Tables/countries.csv',
        'city_country': 'Smaller Tables/city_country.csv',
        'olympics': 'Smaller Tables/swim_olympics.csv'
    },
    'small_track': {
        'candidates': 'Smaller Tables/track_candidates_fixed_gender.csv',
        'combined': 'Smaller Tables/track_combined.csv',
        'events': 'Smaller Tables/track_events.csv',
        'countries_gdp': 'Smaller Tables/countries.csv',
        'city_country': 'Smaller Tables/city_country.csv',
        'olympics': 'Smaller Tables/track_olympics.csv'
    },
    'small_weight_height': {
        'candidates': 'Smaller Tables/weight_height_candidates_fixed_gender.csv',
        'combined': 'Smaller Tables/weight_height_combined.csv',
        'events': 'Smaller Tables/weight_height_events.csv',
        'countries_gdp': 'Smaller Tables/countries.csv',
        'city_country': 'Smaller Tables/city_country.csv',
        'olympics': 'Smaller Tables/weight_height_olympics.csv'
    },
    'scraped': {
        'candidates': 'Scraped Data/athletes.csv',
        'combined': 'Scraped Data/athlete_event_fixed_time.csv',
        'events': 'Scraped Data/events.csv',
        'countries_gdp': 'Scraped Data/countries.csv',
        'city_country': 'Scraped Data/city_country.csv',
        'olympics': 'Scraped Data/olympics.csv'
    }
}

rows = []

for dataset, files in datasets.items():
    for table, path in files.items():
        df = pd.read_csv(path)
        for col in df.columns:
            missing_data_percentage = round(df[col].isnull().mean() * 100, 1)
            if missing_data_percentage > 0:
                rows.append({
                    'dataset': dataset,
                    'table': table,
                    'rows': len(df),
                    'column': col,
                    'percent missing':missing_data_percentage
                })
final_table_df = pd.DataFrame(rows)
print(final_table_df.to_string(index = False))

summary_stats = final_table_df.groupby('dataset')['percent missing'].agg(
    worst = 'max',
    avg = 'mean'
).round(1).reset_index()
print(f"\n{summary_stats.to_string(index = False)}")