import pandas as pd

gdp_df = pd.read_excel('mpd2023_web.xlsx', sheet_name = 'Full data')
countries_df = pd.read_csv('../Extracted CSV Data/countries.csv')

merged = countries_df.merge(gdp_df[['countrycode', 'year', 'gdppc']], 
                            left_on = ['country_id', 'year'], 
                            right_on= ['countrycode', 'year'],
                            how = 'left')

merged['gdp_per_capita'] = merged['gdppc']
merged.drop(columns = ['gdppc', 'countrycode'], inplace = True)
merged['gdp_per_capita'] = merged['gdp_per_capita'].fillna(-1).astype(int)

merged.to_csv('countries.csv', index = False)