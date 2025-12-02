import folium
import pandas as pd
import re

'''
This file merges in the results of the API queries to the FCA.
It uses postcodes to confirm matches to the results from the FCA.
There is then some very basic mapping using folium.
'''

df1 = pd.read_csv('data/problem1_2021_finance_FCA.csv')
lookup = pd.read_csv('data/Geocode_lookup.csv')

df1 = pd.merge(df1, lookup, on='id')

df1['postcode_clean'] = df1['postcodes'].apply(lambda x: re.sub("\[\'", "", x))
df1['postcode_clean'] = df1['postcode_clean'].apply(lambda x: re.sub("\'\]", "", x))
df1['FCA_Postcode'] = df1['FCA_Postcode'].apply(lambda x: re.sub("\'", "", str(x)))

df1.loc[df1['FCA_Postcode'] == df1['postcode_clean'], 'FCA_CONFIRMED'] = 1

# adding to map 

m = folium.Map(location=(51.5, 0))

# matches
for index, row in df1[df1['FCA_CONFIRMED'] != 1].iterrows():
    folium.CircleMarker(
        location=[df1.loc[index, 'lat'], df1.loc[index, 'lon']],
        radius=3,
        color="red",
        stroke=False,
        fill=True,
        fill_opacity=0.6,
        opacity=1,
        popup=df1.loc[index, 'summary'],
        tooltip=df1.loc[index, 'parent_url'],
    ).add_to(m)

# matches
for index, row in df1[df1['FCA_CONFIRMED'] == 1].iterrows():
    folium.CircleMarker(
        location=[df1.loc[index, 'lat'], df1.loc[index, 'lon']],
        radius=6,
        color="cornflowerblue",
        stroke=False,
        fill=True,
        fill_opacity=0.6,
        opacity=1,
        popup=df1.loc[index, 'summary'],
        tooltip=df1.loc[index, 'Name'],
    ).add_to(m)

m.save("geocoding/fca_map.html")
