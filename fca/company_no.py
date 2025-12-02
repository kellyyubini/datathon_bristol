import pandas as pd
import numpy as np
import re

df = pd.read_csv('data/problem1_2021_finance.csv')

'''
This very simple regex tries to find company numbers in the cleaned webdate. EDA showed
that the text 'company number/no:' shows up on a few thousand of the sample. Given that
company numbers follow a very standard format this regex looks to do just that!

It certainly found some of the company numbers, with positive matches to companies house
data, but with false positives, and overall not enough of a sample match be worth pursuing
much further, with the FCA API match turning out to be more effective.
'''

def company_no_finder(string):
    out = re.findall('[:| |;|-][0-9]{8}[\ |. \\\/]', string)
    if len(out) == 0:
        return np.nan
    if len(out) > 0:
        return out[0]

df['co_no_1'] = df['content'].apply(company_no_finder)

df[['id', 'co_no_1']].to_csv('data/company_number.csv')
