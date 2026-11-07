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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
            rows.append({
                'dataset': dataset,
                'table': table,
                'rows': len(df),
                'column': col,
                'percent missing':missing_data_percentage
            })
final_table_df = pd.DataFrame(rows)
print(final_table_df.to_string(index = False))
final_table_df.to_csv("missing_data.csv", index = False)

summary_stats = final_table_df.groupby('dataset')['percent missing'].agg(
    worst = 'max',
    avg = 'mean'
).round(1).reset_index()
print(f"\n{summary_stats.to_string(index = False)}")

dataset_list = final_table_df['dataset'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_list)))
color_set = dict(zip(dataset_list, colors))

labels = final_table_df['table'] + ' | ' + final_table_df['column']
bar_color_set = final_table_df['dataset'].map(color_set)

figure, axis = plt.subplots(figsize = (24, 7))
axis.bar(range(len(final_table_df)), final_table_df['percent missing'], color = bar_color_set)

axis.set_xticks(range(len(final_table_df)))
axis.set_xticklabels(labels, rotation = 90, fontsize = 6)
axis.set_ylabel("Percent Missing")
axis.set_ylim(0, 100)
axis.set_title("Data Missing from Each Dataset")

legend = []
for d in dataset_list:
    legend.append(Patch(facecolor = color_set[d], label = d))
axis.legend(handles = legend)

plt.tight_layout()
plt.savefig("dataset_completeness.png", dpi = 150, bbox_inches = 'tight')
plt.show()
