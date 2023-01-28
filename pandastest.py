import requests
import pandas as pd
import numpy as np
"""
download_url = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
target_csv_path = "nba_all_elo.csv"

response = requests.get(download_url)
response.raise_for_status()    # Check that the request was successful
with open(target_csv_path, "wb") as f:
    f.write(response.content)
print("Download ready.")
"""

import pandas as pd
df = pd.read_csv("Investment_Data_Train (1).csv")
new_df = df[['MSN', 'StateCode', 'Year', 'Amount',
       'CO2 Emissions (Mmt)', 'TotalNumberofInvestments',
       'TotalAmountofAssistance']]

states_df = new_df[~new_df['StateCode'].isin(['DC','US','X3','X5'])]
print(states_df)
states_df2 = states_df[states_df['Year'] == 2015]
print(states_df2)