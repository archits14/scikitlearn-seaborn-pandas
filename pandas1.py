import pandas as pd

data = pd.read_csv('Election Turnout Rates.csv', index_col=0)

print(data.head())
print(data.tail())

print(data.loc['Indiana'])