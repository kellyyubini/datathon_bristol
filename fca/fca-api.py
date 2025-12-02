import pandas as pd
import numpy as np
import re
import requests

'''
This script sets up a quick and hacky way to query the data from the challenge against
FCA records. As we don't have entity names from the OG dataset, this lookup uses web
addresses, this either matches the website held by the FCA or as a 'good enough' entity name.

The script is not written with any redundancy or retries for connection issues etc in mind,
but does save after every request (which also slows things down a little, as a rate limiter)
It will take about 3-5 hours to run.
'''

key = ''
username = ''

headers = {'x-auth-email': username,
           'x-auth-key' : key,
        'Content-Type' : 'application/json'}


df = pd.read_csv('data/problem1_2021_.csv')

df['url_search'] = df['parent_url'].apply(lambda x: re.sub('www.', '', x))
df['url_search'] = df['url_search'].apply(lambda x: re.sub('.co.uk', '', x))

df['Searched'] = ''
df['json'] = ''
df['Name'] = ''
df['URL'] = ''
df['REF_NO'] = ''


for n in range(0,len(df)):
    name = df.loc[n, 'url_search']
    out = requests.get('https://register.fca.org.uk/services/V0.1/Search?q=' + name + '&type=firm', headers=headers)
    out_json = out.json()
    df.loc[n, 'json'] = str(out_json)
    if out_json['ResultInfo'] is None:
        print ('missing')
    if out_json['ResultInfo'] is not None:
        df.loc[n, 'json'] = str(out_json)
        df.loc[n, 'Name'] = out_json['Data'][0]['Name']
        df.loc[n, 'URL'] = out_json['Data'][0]['URL']
        df.loc[n, 'REF_NO'] = out_json['Data'][0]['Reference Number']
        print (str(n) + '  out of ' + str(len(df)))
        print (out_json['ResultInfo'])
        print (out_json['Data'][0]['Name'])

    df.to_csv('data/problem1_2021_finance_FCA.csv')

