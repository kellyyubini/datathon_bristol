import pandas as pd
import numpy as np
import re

## This file runs some basic code to geo-code the postcodes in the main dataset
## It produces a lookup file that can be easily merged into any of the datathon files 

df = pd.read_csv('/Data/problem1_2021_finance.csv')

# UK postcode look up file, from: https://geoportal.statistics.gov.uk/search?q=PRD_NSPL%20NOV_2025&sort=Date%20Created%7Ccreated%7Cdesc
nspl = pd.read_csv('NSPL_NOV_2025_UK.csv')

# Postcodes per entry
df['postcode_count'] = df['postcodes'].apply(lambda x: len(re.split(',', x)))

# Keeping only entries with one postcode - this drops around 400 entries
df1 = df[df['postcode_count'] == 1]

# Cleaning post codes
df1['postcode_clean'] = df1['postcodes'].apply(lambda x: re.sub("\[\'", "", x))
df1['postcode_clean'] = df1['postcode_clean'].apply(lambda x: re.sub("\'\]", "", x))

# Postcode to long/lat lookup dicts
pc_lat = dict(zip(nspl['pcds'], nspl['lat']))
pc_lon = dict(zip(nspl['pcds'], nspl['long']))

df1['lat'] = df1['postcode_clean'].apply(lambda x: pc_lat.get(x))
df1['lon'] = df1['postcode_clean'].apply(lambda x: pc_lon.get(x))

df1[['id', 'lat', 'long']].to_csv('/Data/geocode_lookup.csv')
